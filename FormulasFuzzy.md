# Fórmulas Fundamentais de Sistemas Fuzzy

## 1. Funções de Pertinência

- **Triangular:**  
  $\mu_A(x) = \max\left(\min\left(\frac{x-a}{b-a}, \frac{c-x}{c-b}\right), 0\right)$

- **Trapezoidal:**  
  $\mu_A(x) = \max\left(\min\left(\frac{x-a}{b-a}, 1, \frac{d-x}{d-c}\right), 0\right)$

- **Gaussiana:**  
  $\mu_A(x) = \exp\left(-\frac{1}{2}\left(\frac{x-c}{\sigma}\right)^2\right)$

- **Sigmoidal:**  
  $\mu_A(x) = \frac{1}{1 + e^{-a(x-c)}}$

- **Sino (bell-shaped):**  

  $\mu_A(x) = \frac{1}{1 + \left|\frac{x-c}{a}\right|^{2b}}$

- **S-shaped:**  
  $$
  S(x; a, b) =
  \begin{cases}
    0, & x \leq a \\
    2\left(\frac{x-a}{b-a}\right)^2, & a < x < \frac{a+b}{2} \\
    1 - 2\left(\frac{b-x}{b-a}\right)^2, & \frac{a+b}{2} \leq x < b \\
    1, & x \geq b
  \end{cases}
  $$

- **Z-shaped:**  
  $$
  Z(x; a, b) =
  \begin{cases}
    1, & x \leq a \\
    1 - 2\left(\frac{x-a}{b-a}\right)^2, & a < x < \frac{a+b}{2} \\
    2\left(\frac{b-x}{b-a}\right)^2, & \frac{a+b}{2} \leq x < b \\
    0, & x \geq b
  \end{cases}
  $$

---

## 2. Operações com Conjuntos Fuzzy

- **União:**  
  $\mu_{A \cup B}(x) = \max(\mu_A(x), \mu_B(x))$

- **Interseção:**  
  $\mu_{A \cap B}(x) = \min(\mu_A(x), \mu_B(x))$

- **Complemento:**  
  $\mu_{\overline{A}}(x) = 1 - \mu_A(x)$

---

## 3. α-corte, Suporte e Núcleo

- **α-corte:**  
  $A_\alpha = \{x \mid \mu_A(x) \geq \alpha\}$

- **Suporte:**  
  $\text{Supp}(A) = \{x \mid \mu_A(x) > 0\}$

- **Núcleo:**  
  $\text{Core}(A) = \{x \mid \mu_A(x) = 1\}$

---

## 4. Cardinalidade

- **Discreto:**  
  $|A| = \sum_x \mu_A(x)$

- **Contínuo:**  
  $|A| = \int \mu_A(x) dx$

---

## 5. T-normas (Interseção Generalizada)

- **Mínimo:**  
  $T(a, b) = \min(a, b)$

- **Produto:**  
  $T(a, b) = a \cdot b$

- **Lukasiewicz:**  
  $T(a, b) = \max(0, a + b - 1)$

- **Drástica:**  
  $T(a, b) = \begin{cases} a, & b = 1 \\ b, & a = 1 \\ 0, & \text{caso contrário} \end{cases}$

---

## 6. S-normas (União Generalizada)

- **Máximo:**  
  $S(a, b) = \max(a, b)$

- **Soma Probabilística:**  
  $S(a, b) = a + b - a b$

- **Lukasiewicz:**  
  $S(a, b) = \min(1, a + b)$

- **Drástica:**  
  $S(a, b) = \begin{cases} a, & b = 0 \\ b, & a = 0 \\ 1, & \text{caso contrário} \end{cases}$

---

## 7. Leis de De Morgan (Fuzzy)

- $\overline{A \cup B} = \overline{A} \cap \overline{B}$
- $\overline{A \cap B} = \overline{A} \cup \overline{B}$

---

## 8. Relações Fuzzy e Composição

- **Relação fuzzy:**  
  $R(U, V) = \{((x, y), \mu_R(x, y)) \mid x \in U, y \in V\}$

- **Composição Max-Min:**  
  $\mu_{R \circ S}(u, w) = \max_v [\min(\mu_R(u, v), \mu_S(v, w))]$

- **Composição Max-Produto:**  
  $\mu_{R \circ S}(u, w) = \max_v [\mu_R(u, v) \cdot \mu_S(v, w)]$

---

## 9. Inferência Fuzzy (Modus Ponens Generalizado)

- **Para regra "Se $x$ é $A$ então $y$ é $B$" e fato "$x$ é $A'$":**
  $$
  \mu_{B'}(y) = \max_x \min[\mu_{A'}(x), \mu_R(x, y)]
  $$
  onde $\mu_R(x, y) = \min[\mu_A(x), \mu_B(y)]$ (implicação de Mamdani).

---

## 10. Métodos de Inferência

- **Mamdani (Max-Min):**
  $$
  B'_i(y) = w_i \wedge B_i(y)
  $$
  $$
  B'(y) = \max_i B'_i(y)
  $$

- **Larsen (Max-Produto):**
  $$
  B'_i(y) = w_i \cdot B_i(y)
  $$
  $$
  B'(y) = \max_i B'_i(y)
  $$

- **Kosko/Mizumoto (MIN + Soma):**
  $$
  C(z) = \sum_{i=1}^n \min(w_i, \mu_{C_i}(z))
  $$

---

## 11. Defuzzificação

- **Centro da Área (CoA) / Centroide:**
  $$
  x^* = \frac{\int x \mu_A(x) dx}{\int \mu_A(x) dx}
  $$
  (discreto: $x^* = \frac{\sum x_i \mu_A(x_i)}{\sum \mu_A(x_i)}$)

- **Centro das Somas (CoS):**
  $$
  \text{CoS}(Z) = \frac{\sum_{i=1}^n z_i \cdot \sum_{k=1}^{N'} C'_k(z_i)}{\sum_{i=1}^n \sum_{k=1}^{N'} C'_k(z_i)}
  $$

- **Mínimo/Máximo/Média dos Máximos:**
  - Mínimo dos máximos: menor $z$ com $C(z)$ máximo.
  - Máximo dos máximos: maior $z$ com $C(z)$ máximo.
  - Média dos máximos: média dos $z$ com $C(z)$ máximo.

---

## 12. Outras Fórmulas Úteis

- **Dualidade T-norma/S-norma:**
  $$
  T(a, b) = N(S(N(a), N(b))) \\
  S(a, b) = N(T(N(a), N(b)))
  $$
  onde $N(a) = 1 - a$

- **Partição fuzzy:**  
  Para cada $x$ do universo, $\sum_{i=1}^n \mu_{A_i}(x) > 0$

---

## 13. Resumo de Notação

- $\mu_A(x)$: grau de pertinência de $x$ no conjunto fuzzy $A$
- $A'$, $B'$: conjuntos fuzzy resultantes de inferência
- $T$, $S$: operadores t-norma e s-norma
- $w_i$: grau de ativação da regra $i$
- $C(z)$: conjunto fuzzy de saída
- $x^*$: valor defuzzificado (crisp)

---
