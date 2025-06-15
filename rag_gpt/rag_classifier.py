# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import asyncio
import faiss
import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv
from datasets import load_dataset
from sklearn.metrics import classification_report

# --- 1. Carregamento Robusto da Chave da API ---
print("1. Iniciando configuração e busca pelo arquivo .env...")

try:
    script_path = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_path, '..'))
    dotenv_path = os.path.join(project_root, '.env')

    if os.path.exists(dotenv_path):
        print(f"Arquivo .env encontrado em: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path)
    else:
        raise FileNotFoundError
except (FileNotFoundError, NameError):
    print("Caminho do script não encontrado, tentando carregar .env do diretório atual ou pais...")
    if not load_dotenv():
        print("ERRO: Não foi possível encontrar o arquivo .env.")
        sys.exit()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("\nERRO: A variável OPENAI_API_KEY não foi encontrada. Verifique o conteúdo do seu arquivo .env.")
    sys.exit()

client = AsyncOpenAI(api_key=api_key)
print("Chave da API carregada e cliente OpenAI inicializado com sucesso.")

# --- 2. Configuração do Dataset e Modelos ---
print("\n2. Carregando dataset e configurando modelos...")
dataset = load_dataset("fancyzhx/ag_news")
knowledge_base_df = pd.DataFrame(dataset['train']).sample(10000, random_state=42)
test_df = pd.DataFrame(dataset['test'])
sample_test_df = test_df.groupby('label', group_keys=False).apply(lambda x: x.sample(250, random_state=42))

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
print("Configuração e carregamento de dados concluídos.")


# --- 3. Funções para Construção da Base e Classificação RAG ---

# ######## INÍCIO DA MUDANÇA ##########
async def create_embeddings(texts, model, batch_size=1000):
    """Função para criar embeddings para uma lista de textos EM LOTES."""
    all_embeddings = []
    print(f"Iniciando criação de embeddings em lotes de {batch_size}...")
    
    # Itera sobre a lista de textos em "fatias" (batches)
    for i in range(0, len(texts), batch_size):
        # Pega o lote atual
        batch = texts[i:i + batch_size]
        print(f"Processando lote {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}...")
        
        # Faz a chamada da API para o lote
        response = await client.embeddings.create(input=batch, model=model)
        
        # Adiciona os embeddings do lote à lista principal
        all_embeddings.extend([embedding.embedding for embedding in response.data])
        
        # Uma pequena pausa para não sobrecarregar a API
        await asyncio.sleep(0.5)

    print("Criação de embeddings em lotes concluída.")
    return all_embeddings
# ######## FIM DA MUDANÇA ##########


async def build_or_load_knowledge_base():
    index_path = "agnews_faiss.index"
    kb_path = "agnews_kb.pkl"

    if os.path.exists(index_path) and os.path.exists(kb_path):
        print("\n3. Carregando Base de Conhecimento existente do disco...")
        index = faiss.read_index(index_path)
        kb_df = pd.read_pickle(kb_path)
        print("Base de conhecimento carregada com sucesso.")
        return index, kb_df

    print("\n3. Base de Conhecimento não encontrada. Construindo agora...")
    kb_texts = knowledge_base_df['text'].tolist()
    
    # A chamada aqui não muda, mas a função agora trabalha em lotes
    embeddings = await create_embeddings(kb_texts, EMBEDDING_MODEL)
    embeddings_array = np.array(embeddings).astype('float32')
    
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings_array)
    print(f"Índice FAISS criado com {index.ntotal} vetores.")

    faiss.write_index(index, index_path)
    knowledge_base_df.to_pickle(kb_path)
    print(f"Base de Conhecimento salva em '{index_path}' e '{kb_path}'.")
    return index, knowledge_base_df

def retrieve_examples(query_embedding, index, kb_df, k=3):
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), k)
    retrieved_docs = kb_df.iloc[indices[0]]
    context = ""
    for _, row in retrieved_docs.iterrows():
        label_name = LABEL_NAMES[row['label']]
        context += f"---\nExemplo (Categoria Correta: {label_name}):\n\"{row['text']}\"\n"
    return context

async def classify_with_rag(news_item, index, kb_df, semaphore):
    async with semaphore:
        try:
            query_embedding = (await create_embeddings([news_item['text']], EMBEDDING_MODEL, batch_size=1))[0]
            retrieved_context = retrieve_examples(query_embedding, index, kb_df, k=3)
            
            prompt = f"""Sua tarefa é classificar a notícia final. Para te ajudar, aqui estão alguns exemplos de notícias similares e suas categorias corretas:\n{retrieved_context}\n---\n\nAgora, com base nos exemplos, classifique a seguinte notícia em uma das quatro categorias (0: World, 1: Sports, 2: Business, 3: Sci/Tech) e retorne APENAS um objeto JSON com a chave "categoria_id".\n\nNotícia para classificar:\n"{news_item['text']}" """
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            json_response = json.loads(response.choices[0].message.content)
            # Adicionando um print de progresso na classificação final
            print(f"Notícia {news_item.name+1} classificada com sucesso.")
            return int(json_response["categoria_id"])

        except Exception as e:
            print(f"Erro no processamento RAG para a notícia {news_item.name+1}: {e}")
            return -1

async def main():
    index, kb_df = await build_or_load_knowledge_base()

    CONCURRENT_REQUESTS = 20
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    test_items = [row for _, row in sample_test_df.iterrows()]
    
    tasks = [classify_with_rag(item, index, kb_df, semaphore) for item in test_items]
    
    print(f"\n4. Iniciando classificação RAG para {len(tasks)} notícias...")
    start_time = time.time()
    predictions = await asyncio.gather(*tasks)
    end_time = time.time()
    print(f"Classificação RAG concluída em: {end_time - start_time:.2f} segundos.")
    
    y_true = sample_test_df['label'].tolist()
    valid_indices = [i for i, p in enumerate(predictions) if p != -1]
    y_test_filtered = [y_true[i] for i in valid_indices]
    predictions_filtered = [predictions[i] for i in valid_indices]

    if len(y_test_filtered) > 0:
        print("\n--- Relatório de Classificação Final (RAG com GPT-4o mini) ---")
        print(classification_report(y_test_filtered, predictions_filtered, target_names=LABEL_NAMES.values()))
    else:
        print("Nenhuma predição foi bem-sucedida.")

if __name__ == "__main__":
    asyncio.run(main())