# -*- coding: utf-8 -*-
"""
Script final para executar os testes de ablação para os modelos
SVM e GPT (Zero-Shot) no dataset AG News.

Este script está parametrizado para facilitar a execução de todos os
cenários restantes e completar a tabela de resultados da pesquisa.
"""
import os
import sys
import time
import json
import asyncio
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report


print("--- INICIANDO TESTE DE ABLAÇÃO (SVM & ZERO-SHOT) ---")


CLASSE_A_REMOVER = 3


ALL_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
NOME_CLASSE_REMOVIDA = ALL_LABELS.get(CLASSE_A_REMOVER, "Nenhuma (Baseline)")
print(f"CONFIGURAÇÃO: Removendo a classe '{NOME_CLASSE_REMOVIDA}'")


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


print("\n[FASE 1/2] Preparando os dados...")
dataset = load_dataset("fancyzhx/ag_news")

train_df_full = pd.DataFrame(dataset['train'])
test_df_full = pd.DataFrame(dataset['test'])

if CLASSE_A_REMOVER is not None:
    train_df = train_df_full[train_df_full['label'] != CLASSE_A_REMOVER]
    test_df = test_df_full[test_df_full['label'] != CLASSE_A_REMOVER]
else:
    train_df = train_df_full
    test_df = test_df_full

X_train = train_df['text']
y_train = train_df['label']

sample_test_df = test_df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(250, len(x)), random_state=42))
X_test = sample_test_df['text']
y_test = sample_test_df['label']

print(f"Dados preparados. Treino: {len(train_df)} amostras. Teste: {len(sample_test_df)} amostras.")



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
    
    print(f"\n--- Relatório de Classificação SVM (Cenário: sem '{NOME_CLASSE_REMOVIDA}') ---")
    print(classification_report(y_test, predictions, zero_division=0))

async def run_gpt_zero_shot():
    print("\n--- Executando Modelo GPT (Zero-Shot) ---")
    
    labels_restantes = {k: v for k, v in ALL_LABELS.items() if k != CLASSE_A_REMOVER} if CLASSE_A_REMOVER is not None else ALL_LABELS
    prompt_labels = ", ".join([f"{k}: {v}" for k,v in labels_restantes.items()])

    async def classify_zero_shot(text, semaphore):
        async with semaphore:
            prompt = f"""Sua tarefa é classificar a notícia abaixo em uma das seguintes categorias: {prompt_labels}. Retorne APENAS um objeto JSON com a chave "categoria_id".\n\nNotícia:\n"{text}" """
            try:
                response = await client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0, response_format={"type": "json_object"})
                return int(json.loads(response.choices[0].message.content)["categoria_id"])
            except Exception: return -1

    CONCURRENT_REQUESTS = 5
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    tasks = [classify_zero_shot(text, semaphore) for text in X_test]
    
    start_time = time.time()
    predictions_raw = await asyncio.gather(*tasks)
    end_time = time.time()
    print(f"Inferência Zero-Shot concluída em {end_time - start_time:.2f} segundos.")
    
    valid_indices = [i for i, p in enumerate(predictions_raw) if p != -1]
    y_test_filtered = y_test.iloc[valid_indices]
    predictions_filtered = [p for p in predictions_raw if p != -1]
    
    print(f"\n--- Relatório de Classificação GPT Zero-Shot (Cenário: sem '{NOME_CLASSE_REMOVIDA}') ---")
    print(f"Amostras válidas: {len(predictions_filtered)}/{len(y_test)}")
    print(classification_report(y_test_filtered, predictions_filtered, zero_division=0))

# --- PONTO DE ENTRADA PRINCIPAL ---
async def main():
    run_svm()
    await run_gpt_zero_shot()

if __name__ == "__main__":
    asyncio.run(main())
    print(f"\n--- TESTE DE ABLAÇÃO PARA O CENÁRIO 'sem {NOME_CLASSE_REMOVIDA}' CONCLUÍDO ---")
