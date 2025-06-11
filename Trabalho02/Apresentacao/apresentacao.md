# Sistemas Fuzzy Mamdani e Takagi-Sugeno

## Objetivos

- Desenvolver um sistema fuzzy Mamdani para controle de ventilação em ambientes fechados.
- Implementar um sistema Takagi-Sugeno para aproximação de funções não lineares.
- Analisar o impacto de diferentes funções de pertinência, operadores e técnicas de defuzzificação/otimização.

---

## Sistema Mamdani: Controle de Ventilação

- **Entradas:** Temperatura, Umidade, Número de Pessoas
- **Saída:** Intensidade da Ventilação (fraca, moderada, forte)
- **Funções de pertinência:** Triangulares
- **Base de regras:** 9 regras baseadas em conhecimento de conforto térmico
- **Operadores:** AND (mínimo), OR (máximo, soma)
- **Defuzzificação:** Centroide, Média do Máximo, Bissetriz

### Resultados Mamdani

- Decisões qualitativas coerentes com o conforto térmico.
- Diferenças pequenas entre técnicas de defuzzificação.
- Sistema flexível e interpretável.

---

## Sistema Takagi-Sugeno: Aproximação de Função

- **Função alvo:** $f(x) = e^{-x/5} \cdot \sin(3x) + 0.5 \cdot \sin(x)$
- **Entrada:** $x \in [0, 10]$
- **Regras:** 3 (Baixo, Médio, Alto)
- **Funções de pertinência:** Gaussianas (centros 2, 5, 8), Triangulares, Trapezoidais
- **Consequentes:** Constantes (ordem zero) e lineares (primeira ordem)
- **Otimização:** Gradiente descendente para consequentes

### Resultados Takagi-Sugeno

- **RMSE (ordem zero, gaussiana):** ~0.46
- **RMSE (ordem zero, triangular/trapezoidal):** ~0.35
- **RMSE (primeira ordem):** ~0.41
- **RMSE otimizado:** ~0.41

---

## Conclusões

- O sistema Mamdani é eficaz para decisões qualitativas e interpretáveis.
- O Takagi-Sugeno aproxima funções não lineares com boa precisão, especialmente após otimização.
- A escolha das funções de pertinência e operadores influencia o desempenho.
- Sistemas fuzzy são flexíveis e podem ser adaptados para diferentes aplicações.

---

## Referências

- Zadeh, L. A. (1965). Fuzzy sets.
- Mamdani, E. H., & Assilian, S. (1975). An experiment in linguistic synthesis with a fuzzy logic controller.
- Takagi, T., & Sugeno, M. (1985). Fuzzy identification of systems and its applications to modeling and control.
- Ross, T. J. (2010). Fuzzy Logic with Engineering Applications.

---
