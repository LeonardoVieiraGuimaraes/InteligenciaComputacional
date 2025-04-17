# Funções de Pertinência na Lógica Fuzzy

As funções de pertinência são um dos elementos fundamentais da lógica fuzzy. Elas definem o grau de pertencimento de um elemento a um conjunto fuzzy, atribuindo a cada elemento um valor contínuo entre 0 e 1. Esse valor representa o quanto o elemento é compatível com o conjunto.

## Tipos Comuns de Funções de Pertinência

1. **Função Triangular**  
   É definida por uma forma triangular e é amplamente utilizada devido à sua simplicidade.  
   Fórmula:  
   ```
   μ(x) = max(min((x-a)/(b-a), (c-x)/(c-b)), 0)
   ```
   Onde `a`, `b` e `c` são os parâmetros que definem a base e o pico do triângulo.

2. **Função Trapezoidal**  
   Similar à triangular, mas com um topo plano.  
   Fórmula:  
   ```
   μ(x) = max(min((x-a)/(b-a), 1, (d-x)/(d-c)), 0)
   ```
   Onde `a`, `b`, `c` e `d` definem os limites do trapézio.

3. **Função Gaussiana**  
   Baseada na curva normal, é usada para modelar transições suaves.  
   Fórmula:  
   ```
   μ(x) = exp(-((x-c)^2) / (2*σ^2))
   ```
   Onde `c` é o centro e `σ` é o desvio padrão.

4. **Função Sigmoidal**  
   Utilizada para representar crescimento ou decaimento suave.  
   Fórmula:  
   ```
   μ(x) = 1 / (1 + exp(-α(x-c)))
   ```
   Onde `α` controla a inclinação e `c` é o ponto central.

## Aplicações

As funções de pertinência são amplamente utilizadas em sistemas de controle fuzzy, classificação, tomada de decisão e modelagem de incertezas em diversas áreas, como automação, inteligência artificial e engenharia.

