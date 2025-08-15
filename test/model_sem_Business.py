# -*- coding: utf-8 -*-
"""
Script para realizar o Estudo de Ablação - Teste A (Sem a classe 'Sports')
no dataset AG News.

Este script executa o pipeline completo de comparação para um subconjunto
do dataset, permitindo analisar como a ausência de uma classe afeta o
desempenho dos modelos SVM, GPT (Zero-Shot) e GPT (com RAG).

Para realizar os outros testes (sem 'World', 'Business', 'Sci/Tech'),
basta alterar o valor da variável `CLASSE_A_REMOVER`.
"""
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
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# --- 1. CONFIGURAÇÃO DO EXPERIMENTO ---
print("--- INICIANDO ESTUDO DE ABLAÇÃO ---")

# Altere este valor para os outros testes:
# 0: World, 1: Sports, 2: Business, 3: Sci/Tech
CLASSE_A_REMOVER = 3 
NOME_CLASSE_REMOVIDA = "Sci/Tech"
print(f"CONFIGURAÇÃO: Removendo a classe '{NOME_CLASSE_REMOVIDA}' (ID: {CLASSE_A_REMOVER})")

# Carregamento da chave da API
try:
    script_path = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_path, '..'))
    dotenv_path = os.path.join(project_root, '.env')
    if not load_dotenv(dotenv_path=dotenv_path): raise FileNotFoundError
except (FileNotFoundError, NameError):
    if not load_dotenv(): sys.exit("ERRO: Arquivo .env não encontrado.")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key: sys.exit("ERRO: OPENAI_API_KEY não encontrada no .env.")
client = AsyncOpenAI(api_key=api_key)

# --- 2. PREPARAÇÃO DOS DADOS FILTRADOS ---
print("\n[FASE 1/3] Preparando os dados filtrados...")
dataset = load_dataset("fancyzhx/ag_news")

train_df_full = pd.DataFrame(dataset['train'])
train_df = train_df_full[train_df_full['label'] != CLASSE_A_REMOVER]
X_train = train_df['text']
y_train = train_df['label']

test_df_full = pd.DataFrame(dataset['test'])
sample_test_df_full = test_df_full.groupby('label', group_keys=False).apply(lambda x: x.sample(250, random_state=42))
sample_test_df = sample_test_df_full[sample_test_df_full['label'] != CLASSE_A_REMOVER]
X_test = sample_test_df['text']
y_test = sample_test_df['label']

print(f"Dados preparados. Treino: {len(train_df)} amostras. Teste: {len(sample_test_df)} amostras.")

# --- 3. EXECUÇÃO DOS MODELOS ---

def run_svm():
    print("\n--- Executando Modelo SVM ---")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
        ('svm', LinearSVC(C=1.0, random_state=42, max_iter=2000, dual=True))
    ])
    
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Treino do SVM concluído em {training_time:.2f} segundos.")
    
    predictions = pipeline.predict(X_test)
    
    print(f"\n--- Relatório de Classificação SVM (sem '{NOME_CLASSE_REMOVIDA}') ---")
    print(classification_report(y_test, predictions, zero_division=0))

async def run_gpt_models():
    print("\n--- Executando Modelos GPT (Zero-Shot e RAG) ---")
    
    all_labels = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    labels_restantes = {k: v for k, v in all_labels.items() if k != CLASSE_A_REMOVER}
    prompt_labels = ", ".join([f"{k}: {v}" for k,v in labels_restantes.items()])

    async def create_embeddings(texts, model="text-embedding-3-small", batch_size=500):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = await client.embeddings.create(input=batch, model=model)
            all_embeddings.extend([e.embedding for e in response.data])
            await asyncio.sleep(0.2)
        return all_embeddings

    print("Construindo base de conhecimento para RAG...")
    kb_df = train_df.sample(10000, random_state=42)
    kb_embeddings_list = await create_embeddings(kb_df['text'].tolist())
    kb_embeddings = np.array(kb_embeddings_list).astype('float32')
    index = faiss.IndexFlatL2(kb_embeddings.shape[1])
    index.add(kb_embeddings)
    
    async def classify_zero_shot(text, semaphore):
        async with semaphore:
            prompt = f"""Sua tarefa é classificar a notícia abaixo em uma das seguintes categorias: {prompt_labels}. Retorne APENAS um objeto JSON com a chave "categoria_id".\n\nNotícia:\n"{text}" """
            try:
                response = await client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0, response_format={"type": "json_object"})
                return int(json.loads(response.choices[0].message.content)["categoria_id"])
            except Exception: return -1

    async def classify_with_rag(text, semaphore):
        async with semaphore:
            try:
                query_embedding = (await create_embeddings([text]))[0]
                _, indices = index.search(np.array([query_embedding]).astype('float32'), 3)
                retrieved_docs = kb_df.iloc[indices[0]]
                context = ""
                for _, row in retrieved_docs.iterrows():
                    context += f"---\nExemplo (Categoria Correta: {all_labels[row['label']]}):\n\"{row['text']}\"\n"
                
                prompt = f"""Com base nos exemplos abaixo, classifique a notícia final em uma das seguintes categorias: {prompt_labels}. Retorne APENAS um objeto JSON com a chave "categoria_id".\n\n{context}\n---\n\nNotícia para classificar:\n"{text}" """
                response = await client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0, response_format={"type": "json_object"})
                return int(json.loads(response.choices[0].message.content)["categoria_id"])
            except Exception: return -1

    # ########## INÍCIO DA CORREÇÃO ############
    # Ajustando o número de requisições concorrentes para um valor mais seguro
    CONCURRENT_REQUESTS = 5
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    print(f"INFO: Limite de requisições concorrentes definido para {CONCURRENT_REQUESTS}.")
    # ########## FIM DA CORREÇÃO ############
    
    # --- Execução e Avaliação do Zero-Shot ---
    print("\nExecutando GPT Zero-Shot...")
    zero_shot_tasks = [classify_zero_shot(text, semaphore) for text in X_test]
    start_time_zs = time.time()
    zero_shot_predictions_raw = await asyncio.gather(*zero_shot_tasks)
    end_time_zs = time.time()
    print(f"Inferência Zero-Shot concluída em {end_time_zs - start_time_zs:.2f} segundos.")
    
    valid_indices_zs = [i for i, p in enumerate(zero_shot_predictions_raw) if p != -1]
    y_test_filtered_zs = y_test.iloc[valid_indices_zs]
    zero_shot_predictions_filtered = [p for p in zero_shot_predictions_raw if p != -1]
    
    print(f"\n--- Relatório de Classificação GPT Zero-Shot (sem '{NOME_CLASSE_REMOVIDA}') ---")
    print(f"Amostras válidas: {len(zero_shot_predictions_filtered)}/{len(y_test)}")
    print(classification_report(y_test_filtered_zs, zero_shot_predictions_filtered, zero_division=0))

    # --- Execução e Avaliação do RAG ---
    print("\nExecutando GPT com RAG...")
    rag_tasks = [classify_with_rag(text, semaphore) for text in X_test]
    start_time_rag = time.time()
    rag_predictions_raw = await asyncio.gather(*rag_tasks)
    end_time_rag = time.time()
    print(f"Inferência RAG concluída em {end_time_rag - start_time_rag:.2f} segundos.")

    valid_indices_rag = [i for i, p in enumerate(rag_predictions_raw) if p != -1]
    y_test_filtered_rag = y_test.iloc[valid_indices_rag]
    rag_predictions_filtered = [p for p in rag_predictions_raw if p != -1]

    print(f"\n--- Relatório de Classificação GPT com RAG (sem '{NOME_CLASSE_REMOVIDA}') ---")
    print(f"Amostras válidas: {len(rag_predictions_filtered)}/{len(y_test)}")
    print(classification_report(y_test_filtered_rag, rag_predictions_filtered, zero_division=0))


# --- PONTO DE ENTRADA PRINCIPAL ---
if __name__ == "__main__":
    run_svm()
    asyncio.run(run_gpt_models())
    print("\n--- ESTUDO DE ABLAÇÃO CONCLUÍDO ---")
