# -*- coding: utf-8 -*-
"""
Script para classificar notícias em PARALELO usando asyncio e a API da OpenAI.

Funcionalidades:
1. Usa 'asyncio' para fazer múltiplas chamadas de API concorrentemente,
   acelerando drasticamente o processo.
2. Usa um 'Semaphore' para limitar o número de requisições simultâneas e
   evitar erros de rate limit da API.
3. Carrega a chave da API de um arquivo .env.
4. Carrega o dataset AG News e cria a amostra de teste.
5. Coleta as respostas e calcula as métricas de performance.
"""

import os
import time
import json
import asyncio  
import pandas as pd
from openai import AsyncOpenAI  
from dotenv import load_dotenv
from datasets import load_dataset
from sklearn.metrics import classification_report

# --- 1. Configuração Inicial ---
print("Carregando configurações...")
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("\nERRO: A variável OPENAI_API_KEY não foi encontrada.")
    exit()


client = AsyncOpenAI(api_key=api_key)


dataset = load_dataset("fancyzhx/ag_news")
test_df = pd.DataFrame(dataset['test'])
sample_test_df = test_df.groupby('label', group_keys=False).apply(lambda x: x.sample(250, random_state=42))
news_to_classify = sample_test_df['text'].tolist()
y_test_labels = sample_test_df['label'].tolist()

# --- 2. Definição das Funções Assíncronas ---
def create_prompt(news_text):

    return f"""
    Sua tarefa é classificar a notícia abaixo em uma das quatro categorias a seguir:
    - 0: World, 1: Sports, 2: Business, 3: Sci/Tech
    Analise a notícia e retorne APENAS um objeto JSON com a chave "categoria_id".
    Notícia: "{news_text}"
    """

async def classify_article(news_text, index, total, semaphore):
   
    prompt = create_prompt(news_text)
    
    
    async with semaphore:
        try:
            print(f"Enviando notícia {index}/{total}...")
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            json_response = json.loads(response.choices[0].message.content)
            return int(json_response["categoria_id"])
        
        except Exception as e:
            print(f"Erro ao classificar notícia {index}/{total}: {e}")
            return -1 # Marcar erro

async def main():
    
    CONCURRENT_REQUESTS = 20
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    
    tasks = [classify_article(news, i + 1, len(news_to_classify), semaphore) 
             for i, news in enumerate(news_to_classify)]
    
    print(f"\nIniciando a classificação de {len(tasks)} notícias em paralelo (lotes de {CONCURRENT_REQUESTS})...")
    start_time = time.time()
    
    
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    print(f"\nClassificação em paralelo concluída em: {end_time - start_time:.2f} segundos.")
    
    return results

# --- 3. Execução e Avaliação ---
if __name__ == "__main__":
    
    gpt_predictions = asyncio.run(main())

    
    valid_indices = [i for i, p in enumerate(gpt_predictions) if p != -1]
    y_test_filtered = [y_test_labels[i] for i in valid_indices]
    gpt_predictions_filtered = [gpt_predictions[i] for i in valid_indices]

    if len(y_test_filtered) > 0:
        print("\n--- Relatório de Classificação do Modelo GPT-4o mini (Execução Paralela) ---")
        target_names = ['World', 'Sports', 'Business', 'Sci/Tech']
        print(classification_report(y_test_filtered, gpt_predictions_filtered, target_names=target_names))
    else:
        print("\nNenhuma predição foi bem-sucedida.")

    print("\nProcesso finalizado.")