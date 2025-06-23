# Explicação Detalhada do Trabalho Prático: Classificadores Multiclasse Baseados em Algoritmos de Agrupamento

## 1. Desenvolvimento e Implementação

Você deve criar **três classificadores multiclasse** baseados em algoritmos de agrupamento (clustering) com **treinamento supervisionado**. Ou seja, após o agrupamento, cada cluster será associado a uma classe real, usando os rótulos durante o treinamento.

- **Linguagem:** Pode ser feito em **MATLAB** ou **Python**.
- **Método do Cotovelo:** Utilize o método do cotovelo (Elbow Method) para determinar o número ideal de clusters. Isso envolve rodar o algoritmo para diferentes valores de k e escolher aquele em que a redução do erro começa a diminuir lentamente, formando um “cotovelo” no gráfico.
- **Bases de Dados:** Avalie os classificadores em **dois conjuntos de dados**: Adult e Dry Bean (ambos do UCI).
- **Critérios de Avaliação:**
  - **Acurácia:** Percentual de acertos do classificador.
  - **Desvio Padrão:** Variação da acurácia ao longo das repetições.
  - **Matriz de Confusão:** Tabela que mostra a quantidade de acertos e erros para cada classe.
- **Repetição dos Experimentos:** Repita cada experimento **30 vezes**, usando sementes de 1 a 30 para embaralhar os dados.
- **Divisão dos Dados:** Em cada repetição, divida os dados em 70% para treinamento e 30% para teste.

## 2. Entrega

- **Relatório técnico em PDF** contendo:
  - **Pseudocódigo do classificador**
  - **Metodologia dos experimentos**
  - **Resultados experimentais** em tabelas e gráficos
  - **Análise detalhada dos resultados**
- **Código fonte implementado** (scripts, notebooks, etc)
- **Slides da apresentação**

## 3. Apresentação

- **Duração:** 5 minutos
- **Foco:** Metodologia dos experimentos e resultados (não precisa detalhar código ou teoria)

---

**Resumo:**
Implemente, teste e analise três classificadores baseados em agrupamento supervisionado, usando o método do cotovelo para definir o número de clusters, avaliando em duas bases de dados, repetindo 30 vezes, e entregue relatório técnico, código e slides, além de apresentar os resultados em 5 minutos.
