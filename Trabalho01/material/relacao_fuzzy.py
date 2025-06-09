import numpy as np
import matplotlib.pyplot as plt

# T-normas
def tnorm_min(a, b):
    return np.minimum(a, b)

def tnorm_prod(a, b):
    return a * b

# S-normas
def snorm_max(a, b):
    return np.maximum(a, b)

def snorm_prob(a, b):
    return a + b - a * b

# Calcula matriz de relação fuzzy usando t-norma
def matriz_relacao_fuzzy(A, B, tnorm):
    m = len(A)
    n = len(B)
    R = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            R[i, j] = tnorm(A[i], B[j])
    return R

# Exemplo de conjuntos fuzzy
A = np.array([0.1, 0.5, 0.8, 1.0])
B = np.array([0.2, 0.4, 0.7, 0.9])

# Matrizes de relação usando t-normas
R_min = matriz_relacao_fuzzy(A, B, tnorm_min)
R_prod = matriz_relacao_fuzzy(A, B, tnorm_prod)

# Matrizes de relação usando s-normas (para comparação)
R_max = matriz_relacao_fuzzy(A, B, snorm_max)
R_prob = matriz_relacao_fuzzy(A, B, snorm_prob)

# Plotando os resultados
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
im1 = axs[0, 0].imshow(R_min, cmap='Blues', vmin=0, vmax=1)
axs[0, 0].set_title('T-norma: Mínimo')
plt.colorbar(im1, ax=axs[0, 0])

im2 = axs[0, 1].imshow(R_prod, cmap='Blues', vmin=0, vmax=1)
axs[0, 1].set_title('T-norma: Produto')
plt.colorbar(im2, ax=axs[0, 1])

im3 = axs[1, 0].imshow(R_max, cmap='Oranges', vmin=0, vmax=1)
axs[1, 0].set_title('S-norma: Máximo')
plt.colorbar(im3, ax=axs[1, 0])

im4 = axs[1, 1].imshow(R_prob, cmap='Oranges', vmin=0, vmax=1)
axs[1, 1].set_title('S-norma: Soma Probabilística')
plt.colorbar(im4, ax=axs[1, 1])

for ax in axs.flat:
    ax.set_xlabel('B')
    ax.set_ylabel('A')
    ax.set_xticks(range(len(B)))
    ax.set_yticks(range(len(A)))

plt.tight_layout()
plt.show()

# Análise comparativa
print("Análise comparativa das matrizes de relação fuzzy:")
print("T-norma mínimo gera valores mais conservadores (menores), refletindo interseção forte.")
print("T-norma produto suaviza a relação, permitindo valores intermediários.")
print("S-norma máximo destaca a união dos conjuntos, sempre puxando para o maior grau.")
print("S-norma soma probabilística também reflete união, mas com suavização, nunca ultrapassando 1.")
