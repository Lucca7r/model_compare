# -*- coding: utf-8 -*-
"""
Script para treinar e salvar um modelo SVM para classificação de notícias
do dataset AG News.

Este script realiza as seguintes etapas:
1. Carrega o dataset AG News.
2. Define um pipeline de Machine Learning usando TF-IDF para vetorização
   e um classificador LinearSVC, que é otimizado para velocidade.
3. Treina o pipeline com o conjunto de dados de treino completo (120.000 amostras).
4. Salva o pipeline treinado em um único arquivo ('pipeline_svm_agnews.joblib')
   usando a biblioteca joblib.
5. (Opcional) Avalia o modelo em uma amostra do conjunto de teste e exibe o relatório.
"""

import pandas as pd
import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC 
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# --- Bloco Principal de Execução ---
if __name__ == "__main__":
    print("Iniciando o processo de treinamento do modelo SVM...")

    # 1. Carregar o dataset
    print("Carregando o dataset AG News...")
    dataset = load_dataset("fancyzhx/ag_news")
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    print("Dataset carregado com sucesso.")

    # 2. Preparar os dados para treinamento e teste
    X_train = train_df['text']
    y_train = train_df['label']

    
    
    sample_test_df = test_df.groupby('label', group_keys=False).apply(lambda x: x.sample(250, random_state=42))
    X_test = sample_test_df['text']
    y_test = sample_test_df['label']

    # 3. Criar o Pipeline Otimizado
    svm_pipeline_otimizado = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english',
                                  ngram_range=(1, 2),
                                  max_features=150000)), 
        
        ('svm', LinearSVC(C=1.0,
                          random_state=42,
                          max_iter=2000,
                          dual=True)) 
    ])

    # 4. Treinar o modelo
    
    print("\nIniciando o treinamento do modelo LinearSVC (versão rápida)...")
    start_time = pd.Timestamp.now()
    svm_pipeline_otimizado.fit(X_train, y_train)
    end_time = pd.Timestamp.now()
    print(f"Treinamento concluído em: {end_time - start_time}")

    # 5. Salvar o pipeline treinado em um arquivo
    
    nome_arquivo_modelo = 'model/pipeline_svm_agnews.joblib'
    joblib.dump(svm_pipeline_otimizado, nome_arquivo_modelo)
    print(f"\n✅ Modelo salvo com sucesso como '{nome_arquivo_modelo}'")

    # 6. Avaliar e exibir os resultados
    print("\nAvaliando o modelo na amostra de teste...")
    predictions = svm_pipeline_otimizado.predict(X_test)
    
    print("\n--- Relatório de Classificação do Modelo SVM Otimizado ---")
    target_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    print(classification_report(y_test, predictions, target_names=target_names))

    print("\nProcesso finalizado.")