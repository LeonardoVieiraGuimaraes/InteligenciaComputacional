#!/usr/bin/env python
# coding: utf-8

# # 1. Funções de Pertinência para Lógica Fuzzy

# ## 1.1. Implementac¸ao de Funcões de Pertinência

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os

# Cria o diretório para salvar os gráficos, se não existir
output_dir = 'relatorio/img'
os.makedirs(output_dir, exist_ok=True)


# ### Plot 

# In[121]:


def plot_results(x, y, params, type):
    # Plotando a função
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f"{type.capitalize()} {params}")
    plt.title(f"Função {type.capitalize()}")
    plt.xlabel("x")
    plt.ylabel("Grau de Pertinência")
    plt.legend()
    plt.grid()
    plt.savefig(f'{output_dir}/{type}.png')
    plt.show()


# ### 1. Função Triangular

# In[122]:


def triangular(x, a, b, c):
    if a <= x < b:
        return (x - a) / (b - a)
    elif b <= x < c:
        return (c - x) / (c - b)
    else:
        return 0


# In[123]:


# Exemplo de plotagem para a função triangular
a, b, c = 5, 10, 15  # Parâmetros da função triangular
x = np.linspace(0, 20, 100)  # Valores de x no intervalo [0, 20]

# Calcula os graus de pertinência
y = [triangular(val, a, b, c) for val in x]

# Plotando a função triangular
plot_results(x, y, [a, b, c], "triangular")


# ### 2. Função Trapezoidal

# In[124]:


def trapezoidal(x, a, b, c, d):
    if a <= x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1
    elif c < x <= d:
        return (d - x) / (d - c)
    else:
        return 0


# In[125]:


# Exemplo de plotagem para a função trapezoidal
a, b, c, d = 5, 10, 15, 20  # Parâmetros da função trapezoidal
x = np.linspace(0, 25, 100)  # Valores de x no intervalo [0, 25]

# Calcula os graus de pertinência
y = [trapezoidal(val, a, b, c, d) for val in x]

# Plotando a função trapezoidal
plot_results(x, y, [a, b, c, d], "trapezoidal")


# ### 3. Função Gaussiana

# In[126]:


def gaussian(x, c, sigma):
    return np.exp(-0.5 * ((x - c) / sigma) ** 2)


# In[127]:


# Exemplo de uso
x = np.linspace(0, 20, 100)  # Valores de x no intervalo [0, 20]
c = 10  # Centro da curva (onde a pertinência é máxima)
sigma = 3  # Largura da curva

# Calcula os graus de pertinência
y = [gaussian(val, c, sigma) for val in x]

# Plotando a função gaussiana
plot_results(x, y, [c, sigma], "gaussian")


# ### 4. Função Sigmoidal

# In[128]:


def sigmoidal(x, a, c):
    return 1 / (1 + np.exp(-a * (x - c)))


# In[129]:


# Exemplo de plotagem para a função sigmoidal
a, c = 1, 10  # Parâmetros da função sigmoidal
x = np.linspace(0, 20, 100)  # Valores de x no intervalo [0, 20]

# Calcula os graus de pertinência
y = [sigmoidal(val, a, c) for val in x]

# Plotando a função sigmoidal
plot_results(x, y, [a, c], "sigmoidal")


# ### 5. Função Sinoidal (Bell)

# In[130]:


def bell_function(x, a, b, c):
    """
    Função de pertinência em forma de sino.
    :param x: Valor de entrada.
    :param a: Controla a largura do sino.
    :param b: Controla a inclinação.
    :param c: Centro do sino.
    :return: Grau de pertinência.
    """
    return 1 / (1 + abs((x - c) / a) ** (2 * b))


# In[131]:


# Exemplo de plotagem para a função Bell
a, b, c = 2, 4, 10  # Parâmetros da função Bell
x = np.linspace(0, 20, 100)  # Valores de x no intervalo [0, 20]

# Calcula os graus de pertinência
y = [bell_function(val, a, b, c) for val in x]

# Plotando a função Bell
plot_results(x, y, [a, b, c], "bell")


# ### 6. Função S

# In[132]:


def s_function(x, a, b):
    if x <= a:
        return 0
    elif a < x < b:
        return 2 * ((x - a) / (b - a)) ** 2
    elif x >= b:
        return 1


# In[133]:


# Exemplo de plotagem para a função S
a, b = 5, 15  # Parâmetros da função S
x = np.linspace(0, 20, 100)  # Valores de x no intervalo [0, 20]

# Calcula os graus de pertinência
y = [s_function(val, a, b) for val in x]

# Plotando a função S
plot_results(x, y, [a, b], "s")


# ### 7. Função Z

# In[134]:


def z_function(x, a, b):
    if x <= a:
        return 1
    elif a < x < b:
        return 1 - 2 * ((x - a) / (b - a)) ** 2
    elif x >= b:
        return 0


# In[135]:


# Exemplo de plotagem para a função Z
a, b = 5, 15  # Parâmetros da função Z
x = np.linspace(0, 20, 100)  # Valores de x no intervalo [0, 20]

# Calcula os graus de pertinência
y = [z_function(val, a, b) for val in x]

# Plotando a função Z
plot_results(x, y, [a, b], "z")


# ### 8. Função Cauchy

# In[136]:


def cauchy_function(x, c, gamma):
    """
    Função de pertinência Cauchy.
    :param x: Valor de entrada.
    :param c: Centro da função.
    :param gamma: Largura da função.
    :return: Grau de pertinência.
    """
    return 1 / (1 + ((x - c) / gamma) ** 2)


# In[137]:


# Exemplo de plotagem para a função Cauchy
c, gamma = 10, 3  # Parâmetros da função Cauchy
x = np.linspace(0, 20, 100)  # Valores de x no intervalo [0, 20]

# Calcula os graus de pertinência
y = [cauchy_function(val, c, gamma) for val in x]

# Plotando a função Cauchy
plot_results(x, y, [c, gamma], "cauchy")


# ### 9. Função Gaussiana Dupla

# In[138]:


def double_gaussian(x, c1, sigma1, c2, sigma2):
    """
    Função de pertinência Gaussiana Dupla.
    :param x: Valor de entrada.
    :param c1: Centro da primeira gaussiana.
    :param sigma1: Largura da primeira gaussiana.
    :param c2: Centro da segunda gaussiana.
    :param sigma2: Largura da segunda gaussiana.
    :return: Grau de pertinência combinado.
    """
    return np.maximum(
        np.exp(-0.5 * ((x - c1) / sigma1) ** 2),
        np.exp(-0.5 * ((x - c2) / sigma2) ** 2)
    )


# In[139]:


# Exemplo de plotagem para a função Gaussiana Dupla
c1, sigma1 = 8, 2  # Parâmetros da primeira gaussiana
c2, sigma2 = 14, 3  # Parâmetros da segunda gaussiana
x = np.linspace(0, 20, 100)  # Valores de x no intervalo [0, 20]

# Calcula os graus de pertinência
y = [double_gaussian(val, c1, sigma1, c2, sigma2) for val in x]

# Plotando a função Gaussiana Dupla
plot_results(x, y, [c1, sigma1, c2, sigma2], "double_gaussian")


# ### 10. Função Retangular

# In[140]:


def rectangular_function(x, a, b):
    """
    Função de pertinência retangular.
    :param x: Valor de entrada.
    :param a: Início do intervalo.
    :param b: Fim do intervalo.
    :return: Grau de pertinência.
    """
    return 1 if a <= x <= b else 0


# In[141]:


# Exemplo de plotagem para a função Retangular
a, b = 3, 7  # Parâmetros da função Retangular
x = np.linspace(0, 10, 100)  # Valores de x no intervalo [0, 10]

# Calcula os graus de pertinência
y = [rectangular_function(val, a, b) for val in x]

# Plotando a função Retangular
plot_results(x, y, [a, b], "rectangular")


# ### 11. Função logarítmica

# In[142]:


def logarithmic_function(x, a, b):
    """
    Função de pertinência logarítmica.
    :param x: Valor de entrada.
    :param a: Base do logaritmo.
    :param b: Escala do logaritmo.
    :return: Grau de pertinência.
    """
    if x > 0:
        return min(1, max(0, b * np.log(x) / np.log(a)))
    return 0


# In[143]:


# Exemplo de plotagem para a função Logarítmica
a, b = 2, 1  # Parâmetros da função Logarítmica
x = np.linspace(0.1, 10, 100)  # Valores de x no intervalo [0.1, 10] (evitando zero)

# Calcula os graus de pertinência
y = [logarithmic_function(val, a, b) for val in x]

# Plotando a função Logarítmica
plot_results(x, y, [a, b], "logarithmic")


# ### 12. Função Pi

# In[144]:


def pi_function(x, a, b, c):
    if x < b:
        return s_function(x, a, b)
    else:
        return z_function(x, b, c)


# In[145]:


# Exemplo de plotagem para a função Pi
a, b, c = 5, 10, 15  # Parâmetros da função Pi
x = np.linspace(0, 20, 100)  # Valores de x no intervalo [0, 20]

# Calcula os graus de pertinência
y = [pi_function(val, a, b, c) for val in x]

# Plotando a função Pi
plot_results(x, y, [a, b, c], "pi")


# ### 13. Função Singleton

# In[146]:


def singleton_function(x, c):
    """
    Função de pertinência Singleton.
    :param x: Valor de entrada.
    :param c: Ponto onde a pertinência é máxima (1).
    :return: Grau de pertinência.
    """
    return 1 if x == c else 0


# In[147]:


# Exemplo de plotagem para a função Singleton
c = 10  # Parâmetro da função Singleton
x = np.linspace(0, 20, 100)  # Valores de x no intervalo [0, 20]

# Calcula os graus de pertinência
y = [singleton_function(val, c) for val in x]

# Plotando a função Singleton
plot_results(x, y, [c], "singleton")


# ### 14. Função Linear

# In[148]:


def linear_function(x, a, b):
    """
    Função de pertinência linear crescente ou decrescente.
    :param x: Valor de entrada.
    :param a: Início do intervalo.
    :param b: Fim do intervalo.
    :return: Grau de pertinência.
    """
    if x <= a:
        return 0
    elif x >= b:
        return 1
    else:
        return (x - a) / (b - a)


# In[149]:


# Exemplo de plotagem para a função Linear
a, b = 5, 15  # Parâmetros da função Linear
x = np.linspace(0, 20, 100)  # Valores de x no intervalo [0, 20]

# Calcula os graus de pertinência
y = [linear_function(val, a, b) for val in x]

# Plotando a função Linear
plot_results(x, y, [a, b], "linear")


# ### Garu de Pertinência

# In[150]:


### Função Membership
def calculate_membership(x, types, params):
    """
    Calcula os graus de pertinência para os valores em intervalos com base nos tipos e parâmetros fornecidos.

    :param dominio: Tupla (min, max) definindo o intervalo do domínio.
    :param types: Lista de strings indicando os tipos de funções de pertinência.
    :param params: Lista de listas com os parâmetros para cada função de pertinência.
    :return: Lista de listas com os graus de pertinência para cada tipo.
    """
    # x = np.linspace(dominio[0], dominio[1], 100)  # Gera os valores de x no domínio
    # Gera os valores de x no domínio
    results = []
    for func_type, func_params in zip(types, params):

        # Calcula os graus de pertinência com base no tipo de função
        if func_type == 'linear':
            results.append([linear_function(val, *func_params) for val in x])  
        elif func_type == 'triangular':
            results.append([triangular(val, *func_params) for val in x])
        elif func_type == 'trapezoidal':
            results.append([trapezoidal(val, *func_params) for val in x])
        elif func_type == 'gaussian':
            results.append([gaussian(val, *func_params) for val in x])
        elif func_type == 'sigmoidal':
            results.append([sigmoidal(val, *func_params) for val in x])
        elif func_type == 'z':
            results.append([z_function(val, *func_params) for val in x])
        elif func_type == 's':
            results.append([s_function(val, *func_params) for val in x])
        elif func_type == 'pi':
            results.append([pi_function(val, *func_params) for val in x])
        elif func_type == 'bell':
            results.append([bell_function(val, *func_params) for val in x])
        elif func_type == 'singleton':
            results.append([singleton_function(val, *func_params) for val in x])
        elif func_type == 'cauchy':
            results.append([cauchy_function(val, *func_params) for val in x])
        elif func_type == 'double_gaussian':
            results.append([double_gaussian(val, *func_params) for val in x])
        elif func_type == 'retangular':
            results.append([rectangular_function(val, *func_params) for val in x])
        elif func_type == 'logaritmica':
            results.append([logarithmic_function(val, *func_params) for val in x])
        else:
            raise ValueError(f"Tipo de função desconhecido: {func_type}")

    return results


# ## 1.2. Fuzzificação e Análise Comporativa

# ### Gera os Parametros

# In[151]:


def generate_params(X, types, n):
    """
    Gera os parâmetros para diferentes tipos de funções de pertinência.

    Args:
        X (tuple): Intervalo do domínio (mínimo, máximo).
        types (list): Lista de tipos de funções ('gaussian', 'triangular', etc.).
        n (int): Número de funções de pertinência.

    Returns:
        list: Lista de parâmetros para cada tipo de função.
    """
    centers = np.linspace(X[0], X[1], n)  # Centros uniformemente distribuídos
    step = (X[1] - X[0]) / (n - 1) if n > 1 else (X[1] - X[0])  # Espaçamento entre os centros
    params = []

    for i, t in enumerate(types):
        if t == 'gaussian':
            sigma = step / 2  # Sigma proporcional ao espaçamento
            params.append([centers[i], sigma])
        elif t == 'triangular':
            a = max(X[0], centers[i] - step)  # Início da base
            b = centers[i]                   # Pico
            c = min(X[1], centers[i] + step)  # Fim da base
            params.append([a, b, c])
        elif t == 'trapezoidal':
            a = max(X[0], centers[i] - step)  # Início da base
            b = max(X[0], centers[i] - step / 2)  # Início do topo
            c = min(X[1], centers[i] + step / 2)  # Fim do topo
            d = min(X[1], centers[i] + step)  # Fim da base
            params.append([a, b, c, d])
        elif t == 'sigmoidal':
            a = 1  # Inclinação padrão
            c = centers[i]  # Centro
            params.append([a, c])
        elif t == 'bell':
            a = step / 2  # Largura do sino
            b = 2  # Inclinação padrão
            c = centers[i]  # Centro
            params.append([a, b, c])
        elif t == 'z':
            a = max(X[0], centers[i] - step)  # Início do decaimento
            b = centers[i]  # Fim do decaimento
            params.append([a, b])
        elif t == 's':
            a = centers[i]  # Início do crescimento
            b = min(X[1], centers[i] + step)  # Fim do crescimento
            params.append([a, b])
        elif t == 'pi':
            a = max(X[0], centers[i] - step)  # Início do crescimento
            b = centers[i]  # Pico
            c = min(X[1], centers[i] + step)  # Fim do decaimento
            params.append([a, b, c])
        elif t == 'singleton':
            c = centers[i]  # Ponto único
            params.append([c])
        elif t == 'cauchy':
            c = centers[i]  # Centro
            gamma = step / 2  # Largura
            params.append([c, gamma])
        elif t == 'double_gaussian':
            c1 = max(X[0], centers[i] - step / 2)  # Centro da primeira gaussiana
            sigma1 = step / 4  # Largura da primeira gaussiana
            c2 = min(X[1], centers[i] + step / 2)  # Centro da segunda gaussiana
            sigma2 = step / 4  # Largura da segunda gaussiana
            params.append([c1, sigma1, c2, sigma2])
        elif t == 'retangular':
            a = max(X[0], centers[i] - step / 2)  # Início do intervalo
            b = min(X[1], centers[i] + step / 2)  # Fim do intervalo
            params.append([a, b])
        elif t == 'logaritmica':
            a = 2  # Base do logaritmo
            b = 1  # Escala
            params.append([a, b])
        elif t == 'linear':  # Adicionando suporte para 'linear'
            a = max(X[0], centers[i] - step)  # Início do intervalo
            b = min(X[1], centers[i] + step)  # Fim do intervalo
            params.append([a, b])
        else:
            raise ValueError(f"Tipo de função '{t}' não suportado!")

    return params


# In[152]:


def plot_membership(x, u, label):

    plt.figure(figsize=(10, 6))

    for i, result in enumerate(u):
        plt.plot(x, result, label=f"{label[i]}")

    plt.legend()
    plt.title("Funções de Pertinência")
    plt.xlabel("x")
    plt.ylabel("Grau de Pertinência")
    plt.grid()
    plt.show()


# In[153]:


def plot_membership_with_samples(x, results, labels, samples, title, activations):

    plt.figure(figsize=(10, 6))

    # Plota cada função de pertinência
    for i, result in enumerate(results):
        plt.plot(x, result, label=f"{labels[i]}")

    # Adiciona as amostras e seus graus de ativação
    for j, sample in enumerate(samples):
        plt.axvline(sample, color="gray", linestyle="--", alpha=0.7, label=f"Sample: {sample}")
        for i, activation in enumerate(activations):
            plt.scatter(sample, activation[j], color="red", zorder=5)

    # Configurações do gráfico
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("Grau de Pertinência")
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1), title="Funções Fuzzy")
    plt.grid(True)
    plt.tight_layout()
    print(f'{title.replace("- ", "").replace(" ", "_").lower()}_fuzzificado.png')
    plt.savefig(f'{output_dir}/{title.replace("- ", "").replace(" ", "_").lower()}_fuzzificado.png')  # Salva o gráfico
    plt.show()


# ### Função com exemplos

# In[154]:


def fuzzificacao(x, n, type, dominio, samples, labels, liguistica):
    """
    Realiza a fuzzificação e plota as funções de pertinência com destaque para as amostras.

    :param n: Número de funções de pertinência.
    :param type: Tipo de função de pertinência.
    :param dominio: Intervalo do domínio (mínimo, máximo).
    :param samples: Lista de amostras a serem destacadas.
    :param labels: Lista de rótulos para cada função de pertinência.
    """
    # Número de funções de pertinência para cada atributo
    types = [type] * n

    # Geração dos parâmetros para cada tipo de função
    params = generate_params(dominio, types, n)

    # Cálculo dos graus de pertinência para cada atributo
    results = calculate_membership(x, types, params)


    for i in range(n):
        v = dominio[1] - dominio[0]  # Tamanho do intervalo
        p = v / n  # Espaçamento correto entre os pontos
        intervalo = [round(dominio[0] + p * i, 2), round(dominio[0] + p * (i + 1), 2)]
        parametros_formatados = [float(round(val, 2)) for val in params[i]]

        print(f'{labels[i]}: {liguistica} - {intervalo} - Parâmetros: {parametros_formatados}')
    # Cálculo do grau de ativação para cada amostra
    activations = []
    for i, result in enumerate(results):
        sample_activations = []
        for sample in samples:
            # Encontra o índice mais próximo do valor da amostra no domínio
            idx = np.abs(x - sample).argmin()
            activation = result[idx]
            sample_activations.append(activation)
        activations.append(sample_activations)

    # Imprime as ativações calculadas com melhor formatação
    print(f"\nAtivações para o tipo {type}:")
    print(f"Samples: {samples}")
    for i, sample_activations in enumerate(activations):
         # Arredonda os valores para duas casas decimais
        rounded_activations = np.round(sample_activations, 2)
        print(f"{labels[i]}: {rounded_activations.tolist()}")

    # Plotagem das funções de pertinência para o atributo com as amostras
    plot_membership_with_samples(
        x, results, labels, samples, f"Funções de Pertinência - {type}", activations
    )


# In[155]:


# Definição do universo de discurso
dominio = [0, 100]  # Intervalo do universo de discurso
samples = [25, 75]  # Amostras para fuzzificação
x = np.linspace(dominio[0], dominio[1], 100)
n = 5
funcao_pertinecia = [
    'triangular', 'trapezoidal', 'gaussian', 'sigmoidal', 'bell', 's', 'z', 'pi', 'singleton', 'cauchy', 'double_gaussian', 'logaritmica', 'retangular'
]

labels = ['Muita Fria', 'Fria', 'Morno', 'Quente', 'Muito quente']

for type in funcao_pertinecia:

    fuzzificacao(x, n, type  , dominio, samples, [labels[i] for i in range(n)], liguistica='Temperatura')


# # 2. Operações Básicas no Contexto Fuzzy

# ## 2.1. Complemento União e Intersecção

# ### Complemento

# #### Zadeh

# In[156]:


def complemento_zadeh(u):
    return 1 - np.array(u)


# #### Sugeno

# In[157]:


def complemento_sugeno(u, lamb=0.5):
    u = np.array(u)  # Converte u para um array NumPy, se ainda não for
    return (1 - u) / (1 + lamb * u)


# #### Yager

# In[158]:


def complemento_yager(u, w=2):
    return (1 - u**w)**(1/w)


# In[159]:


def aplicar_complemento(results, tipo):
    results_complement = []

    for result in results:
        result = np.array(result)  # Garante que cada resultado seja um array NumPy
        if tipo == "zadeh":
            result_comp = complemento_zadeh(result)
        elif tipo == "sugeno":
            result_comp = complemento_sugeno(result)
        elif tipo == "yager":
            result_comp = complemento_yager(result)
        else:
            raise ValueError("Complemento não reconhecido.")
        results_complement.append(result_comp)

    return results_complement


# In[160]:


def fuzzificacao(x, n, type, dominio, samples, labels, liguistica, comp):
    """
    Realiza a fuzzificação e plota as funções de pertinência com destaque para as amostras.

    :param n: Número de funções de pertinência.
    :param type: Tipo de função de pertinência.
    :param dominio: Intervalo do domínio (mínimo, máximo).
    :param samples: Lista de amostras a serem destacadas.
    :param labels: Lista de rótulos para cada função de pertinência.
    """
    # Número de funções de pertinência para cada atributo
    types = [type] * n

    # Geração dos parâmetros para cada tipo de função
    params = generate_params(dominio, types, n)

    # Cálculo dos graus de pertinência para cada atributo
    results = calculate_membership(x, types, params)

    results = aplicar_complemento(results, comp)

    for i in range(n):
        v = dominio[1] - dominio[0]  # Tamanho do intervalo
        p = v / n  # Espaçamento correto entre os pontos
        intervalo = [round(dominio[0] + p * i, 2), round(dominio[0] + p * (i + 1), 2)]
        parametros_formatados = [float(round(val, 2)) for val in params[i]]

        print(f'{labels[i]}: {liguistica} - {intervalo} - Parâmetros: {parametros_formatados}')
    # Cálculo do grau de ativação para cada amostra
    activations = []
    for i, result in enumerate(results):
        sample_activations = []
        for sample in samples:
            # Encontra o índice mais próximo do valor da amostra no domínio
            idx = np.abs(x - sample).argmin()
            activation = result[idx]
            sample_activations.append(activation)
        activations.append(sample_activations)

    # Imprime as ativações calculadas com melhor formatação
    print(f"\nAtivações para o tipo {type}:")
    print(f"Samples: {samples}")
    for i, sample_activations in enumerate(activations):
         # Arredonda os valores para duas casas decimais
        rounded_activations = np.round(sample_activations, 2)
        print(f"{labels[i]}: {rounded_activations.tolist()}")

    # Plotagem das funções de pertinência para o atributo com as amostras
    plot_membership_with_samples(
        x, results, labels, samples, f"Funções de Pertinência Operador Complemento {type} {comp}", activations
    )


# In[161]:


# Definição do universo de discurso e execução da fuzzificação
dominio = [0, 100]  # Intervalo do universo de discurso
samples = [25, 75]  # Amostras para fuzzificação
x = np.linspace(dominio[0], dominio[1], 100)  # Valores de x no intervalo
n = 5  # Número de funções de pertinência
funcao_pertinencia = [
    'triangular', 'trapezoidal', 'gaussian', 'sigmoidal', 'bell', 's', 'z', 'pi', 'singleton', 'cauchy', 'double_gaussian', 'logaritmica', 'retangular'
]
labels = ['Muita Fria', 'Fria', 'Morno', 'Quente', 'Muito quente']  # Rótulos para as funções
complemento = ['zadeh', 'sugeno', 'yager']
liguistica = "Temperatura"
# Executa a fuzzificação para cada tipo de função de pertinência
for type in funcao_pertinencia:
    for comp in complemento:
        fuzzificacao(x, n, type, dominio, samples, [labels[i] for i in range(n)], liguistica, comp)


# ### União (t-conormas)

# #### Máximo:

# In[162]:


# Define a função uniao_maximo
def uniao_maximo(u1, u2):

    return np.maximum(u1, u2)


# #### Soma Probabilística:

# In[163]:


def uniao_soma_probabilistica(u1, u2):
    u1 = np.array(u1)
    u2 = np.array(u2)
    return u1 + u2 - u1 * u2


# #### Soma Limitada:

# In[164]:


def uniao_soma_limitada(u1, u2):
    u1 = np.array(u1)
    u2 = np.array(u2)
    return np.minimum(1, u1 + u2)


# #### Soma Drástica:

# In[165]:


def uniao_soma_drastica(u1, u2):
    u1 = np.array(u1)
    u2 = np.array(u2)
    return np.where((u1 == 0) & (u2 == 0), 0, np.maximum(u1, u2))


# In[166]:


def fuzzificacao(x, n, type, dominio, samples, labels, liguistica, uniao=None):
    """
    Realiza a fuzzificação e plota as funções de pertinência com destaque para as amostras.

    :param n: Número de funções de pertinência.
    :param type: Tipo de função de pertinência.
    :param dominio: Intervalo do domínio (mínimo, máximo).
    :param samples: Lista de amostras a serem destacadas.
    :param labels: Lista de rótulos para cada função de pertinência.
    :param uniao: Tipo de operador de união (t-conorma) a ser aplicado.
    """
    # Número de funções de pertinência para cada atributo
    types = [type] * n

    # Geração dos parâmetros para cada tipo de função
    params = generate_params(dominio, types, n)

    # Cálculo dos graus de pertinência para cada atributo
    results = calculate_membership(x, types, params)

    # Aplicação da união (t-conormas), se especificado
    if uniao:
        if len(results) < 2:
            raise ValueError("São necessários pelo menos dois conjuntos fuzzy para aplicar a união.")
        u1, u2 = results[:2]  # Considera os dois primeiros conjuntos para a união
        if uniao == "maximo":
            results_uniao = uniao_maximo(u1, u2)
        elif uniao == "soma_probabilistica":
            results_uniao = uniao_soma_probabilistica(u1, u2)
        elif uniao == "soma_limitada":
            results_uniao = uniao_soma_limitada(u1, u2)
        elif uniao == "soma_drastica":
            results_uniao = uniao_soma_drastica(u1, u2)
        else:
            raise ValueError(f"Tipo de união desconhecido: {uniao}")
        results.append(results_uniao)
        labels.append(f"União ({uniao.capitalize()})")

    for i in range(n):
        v = dominio[1] - dominio[0]  # Tamanho do intervalo
        p = v / n  # Espaçamento correto entre os pontos
        intervalo = [round(dominio[0] + p * i, 2), round(dominio[0] + p * (i + 1), 2)]
        parametros_formatados = [float(round(val, 2)) for val in params[i]]

        print(f'{labels[i]}: {liguistica} - {intervalo} - Parâmetros: {parametros_formatados}')

    # Cálculo do grau de ativação para cada amostra
    activations = []
    for i, result in enumerate(results):
        sample_activations = []
        for sample in samples:
            # Encontra o índice mais próximo do valor da amostra no domínio
            idx = np.abs(x - sample).argmin()
            activation = result[idx]
            sample_activations.append(activation)
        activations.append(sample_activations)

    # Imprime as ativações calculadas com melhor formatação
    print(f"\nAtivações para o tipo {type} (união: {uniao}):")
    print(f"Samples: {samples}")
    for i, sample_activations in enumerate(activations):
        # Arredonda os valores para duas casas decimais
        rounded_activations = np.round(sample_activations, 2)
        print(f"{labels[i]}: {rounded_activations.tolist()}")

    # Plotagem das funções de pertinência para o atributo com as amostras
    plot_membership_with_samples(
        x, results, labels, samples, f"Funções de Pertinência - {type} - União", activations
    )


# In[167]:


# Definição do universo de discurso e execução da fuzzificação
dominio = [0, 100]  # Intervalo do universo de discurso
samples = [25, 75]  # Amostras para fuzzificação
x = np.linspace(dominio[0], dominio[1], 100)  # Valores de x no intervalo
n = 4  # Número de funções de pertinência
funcao_pertinencia = [
    'triangular', 'trapezoidal', 'gaussian', 'sigmoidal', 'bell', 's', 'z', 'pi', 'singleton', 'cauchy', 'double_gaussian', 'logaritmica', 'retangular'
]
labels = ['Fria', 'Morno', 'Quente', 'Muito quente']  # Rótulos para as funções
unioes = ['maximo', 'soma_probabilistica', 'soma_limitada', 'soma_drastica']  # Tipos de união
liguistica = "Temperatura"

# Executa a fuzzificação para cada tipo de função de pertinência e união
for type in funcao_pertinencia:
    for uniao in unioes:
        fuzzificacao(x, n, type, dominio, samples, [labels[i] for i in range(n)], liguistica, uniao)


# ### Interseção (t-normas)

# #### Mínimo

# In[168]:


def intersecao_minimo(u1, u2):
    return np.minimum(u1, u2)


# #### Produto

# In[169]:


def intersecao_produto(u1, u2):
    u1 = np.array(u1)  # Converte u1 para array NumPy, se necessário
    u2 = np.array(u2)  # Converte u2 para array NumPy, se necessário
    return u1 * u2


# #### Produto Limitado:

# In[170]:


def intersecao_produto_limitado(u1, u2):
    u1 = np.array(u1)
    u2 = np.array(u2)
    return np.maximum(0, u1 + u2 - 1)


# #### Produto Drástico:

# In[171]:


def intersecao_produto_drastico(u1, u2):
    return np.where((u1 == 1) & (u2 == 1), np.minimum(u1, u2), 0)


# In[172]:


def fuzzificacao(x, n, type, dominio, samples, labels, liguistica, intersecao=None):
    # Número de funções de pertinência para cada atributo
    types = [type] * n

    # Geração dos parâmetros para cada tipo de função
    params = generate_params(dominio, types, n)

    # Cálculo dos graus de pertinência para cada atributo
    results = calculate_membership(x, types, params)

    # Aplicação da interseção (t-normas), se especificado
    if intersecao:
        if len(results) < 2:
            raise ValueError("São necessários pelo menos dois conjuntos fuzzy para aplicar a interseção.")
        u1, u2 = results[:2]  # Considera os dois primeiros conjuntos para a interseção
        if intersecao == "minimo":
            results_intersecao = intersecao_minimo(u1, u2)
        elif intersecao == "produto":
            results_intersecao = intersecao_produto(u1, u2)
        elif intersecao == "produto_limitado":
            results_intersecao = intersecao_produto_limitado(u1, u2)
        elif intersecao == "produto_drastico":
            results_intersecao = intersecao_produto_drastico(u1, u2)
        else:
            raise ValueError(f"Tipo de interseção desconhecido: {intersecao}")
        results.append(results_intersecao)
        labels.append(f"Interseção ({intersecao.capitalize()})")

    for i in range(n):
        v = dominio[1] - dominio[0]  # Tamanho do intervalo
        p = v / n  # Espaçamento correto entre os pontos
        intervalo = [round(dominio[0] + p * i, 2), round(dominio[0] + p * (i + 1), 2)]
        parametros_formatados = [float(round(val, 2)) for val in params[i]]

        print(f'{labels[i]}: {liguistica} - {intervalo} - Parâmetros: {parametros_formatados}')

    # Cálculo do grau de ativação para cada amostra
    activations = []
    for i, result in enumerate(results):
        sample_activations = []
        for sample in samples:
            # Encontra o índice mais próximo do valor da amostra no domínio
            idx = np.abs(x - sample).argmin()
            activation = result[idx]
            sample_activations.append(activation)
        activations.append(sample_activations)

    # Imprime as ativações calculadas com melhor formatação
    print(f"\nAtivações para o tipo {type} (interseção: {intersecao}):")
    print(f"Samples: {samples}")
    for i, sample_activations in enumerate(activations):
        # Arredonda os valores para duas casas decimais
        rounded_activations = np.round(sample_activations, 2)
        print(f"{labels[i]}: {rounded_activations.tolist()}")

    # Plotagem das funções de pertinência para o atributo com as amostras
    plot_membership_with_samples(
        x, results, labels, samples, f"Funções de Pertinência - {type} - Interseção", activations
    )

# Definição do universo de discurso e execução da fuzzificação
dominio = [0, 100]  # Intervalo do universo de discurso
samples = [25, 75]  # Amostras para fuzzificação
x = np.linspace(dominio[0], dominio[1], 100)  # Valores de x no intervalo
n = 4  # Número de funções de pertinência
funcao_pertinencia = [
    'triangular', 'trapezoidal', 'gaussian', 'sigmoidal', 'bell', 's', 'z', 'pi', 'singleton', 'cauchy', 'double_gaussian', 'logaritmica', 'retangular'
]
labels = ['Fria', 'Morno', 'Quente', 'Muito quente']  # Rótulos para as funções
intersecoes = ['minimo', 'produto', 'produto_limitado', 'produto_drastico']  # Tipos de interseção
liguistica = "Temperatura"

# Executa a fuzzificação para cada tipo de função de pertinência e interseção
for type in funcao_pertinencia:
    for intersecao in intersecoes:
        fuzzificacao(x, n, type, dominio, samples, [labels[i] for i in range(n)], liguistica, intersecao=intersecao)


# ### Analise Gráfica e textual

# ## 2.2. Relações Fuzzy

# ### Matriz de Relação Fuzzy

# In[179]:


import numpy as np
import matplotlib.pyplot as plt

# Conjuntos fuzzy representando graus de pertinência
temperatura = np.array([0.1, 0.5, 0.8, 1.0])  # Fria, Morna, Quente, Muito Quente
umidade = np.array([0.2, 0.4, 0.7, 0.9])      # Baixa, Moderada, Alta, Muito Alta

# Operadores t-norma
def tnorm_min(a, b):
    return np.minimum(a, b)

def tnorm_prod(a, b):
    return a * b

# Operadores s-norma
def snorm_max(a, b):
    return np.maximum(a, b)

def snorm_prob(a, b):
    return a + b - a * b

# Função para calcular a matriz de relação fuzzy
def matriz_relacao_fuzzy(A, B, operador):
    m, n = len(A), len(B)
    R = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            R[i, j] = operador(A[i], B[j])
    return R

# Função para plotar a matriz
def plot_matriz(matriz, titulo):
    plt.figure(figsize=(6, 5))
    plt.imshow(matriz, cmap="Blues", interpolation="nearest")
    plt.colorbar(label="Grau de Pertinência")
    plt.title(titulo)
    plt.xlabel("Umidade")
    plt.ylabel("Temperatura")
    plt.xticks(range(len(umidade)), ["Baixa", "Moderada", "Alta", "Muito Alta"])
    plt.yticks(range(len(temperatura)), ["Fria", "Morna", "Quente", "Muito Quente"])
    plt.show()

# Matrizes de relação fuzzy usando t-normas
R_min = matriz_relacao_fuzzy(temperatura, umidade, tnorm_min)
print("Matriz T-norma (Mínimo):")
print(R_min)

R_prod = matriz_relacao_fuzzy(temperatura, umidade, tnorm_prod)
print("\nMatriz T-norma (Produto):")
print(R_prod)

# Matrizes de relação fuzzy usando s-normas
R_max = matriz_relacao_fuzzy(temperatura, umidade, snorm_max)
print("\nMatriz S-norma (Máximo):")
print(R_max)

R_prob = matriz_relacao_fuzzy(temperatura, umidade, snorm_prob)
print("\nMatriz S-norma (Soma Probabilística):")
print(R_prob)

# Plotando os resultados
plot_matriz(R_min, "Matriz T-norma (Mínimo)")
plot_matriz(R_prod, "Matriz T-norma (Produto)")
plot_matriz(R_max, "Matriz S-norma (Máximo)")
plot_matriz(R_prob, "Matriz S-norma (Soma Probabilística)")

# Análise textual
print("\nAnálise comparativa das matrizes de relação fuzzy:")
print("1. T-norma (Mínimo): Gera valores mais conservadores, refletindo interseção forte.")
print("2. T-norma (Produto): Suaviza a relação, permitindo valores intermediários.")
print("3. S-norma (Máximo): Destaca a união dos conjuntos, sempre puxando para o maior grau.")
print("4. S-norma (Soma Probabilística): Reflete união com suavização, nunca ultrapassando 1.")


# ## 2.3. Composição de Relação Fuzzy

# #### Máximo-Mínimo: $\mu_R(x, z) = \max_y \min(\mu_A(x, y), \mu_B(y, z))$
# 
# 

# In[174]:


import numpy as np

# Função Máximo-Mínimo
def maximo_minimo(A, B):
    m, n = A.shape[0], B.shape[1]
    R = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            R[i, j] = np.max(np.minimum(A[i, :], B[:, j]))
    return R


# #### Mínimo-Máximo: $\mu_R(x, z) = \min_y \max(\mu_A(x, y), \mu_B(y, z))$

# In[175]:


# Função Mínimo-Máximo
def minimo_maximo(A, B):
    m, n = A.shape[0], B.shape[1]
    R = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            R[i, j] = np.min(np.maximum(A[i, :], B[:, j]))
    return R


# #### Máximo-Produto: $\mu_R(x, z) = \max_y (\mu_A(x, y) \cdot \mu_B(y, z))$

# In[176]:


# Função Máximo-Produto
def maximo_produto(A, B):
    m, n = A.shape[0], B.shape[1]
    R = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            R[i, j] = np.max(A[i, :] * B[:, j])
    return R


# ### Exemplo conjuntos fuzzy

# In[180]:


import numpy as np
import matplotlib.pyplot as plt

# Função Máximo-Mínimo
def maximo_minimo(A, B):
    m, n = A.shape[0], B.shape[1]
    R = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            R[i, j] = np.max(np.minimum(A[i, :], B[:, j]))
    return R

# Função Mínimo-Máximo
def minimo_maximo(A, B):
    m, n = A.shape[0], B.shape[1]
    R = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            R[i, j] = np.min(np.maximum(A[i, :], B[:, j]))
    return R

# Função Máximo-Produto
def maximo_produto(A, B):
    m, n = A.shape[0], B.shape[1]
    R = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            R[i, j] = np.max(A[i, :] * B[:, j])
    return R

# Exemplos de matrizes fuzzy para temperatura e umidade
temperatura = np.array([[0.1, 0.5, 0.8],  # Fria
                         [0.2, 0.6, 0.9],  # Morna
                         [0.3, 0.7, 1.0]]) # Quente

umidade = np.array([[0.2, 0.4, 0.7],  # Baixa
                    [0.3, 0.5, 0.8],  # Moderada
                    [0.4, 0.6, 0.9]]) # Alta

# Aplicação das composições
R_max_min = maximo_minimo(temperatura, umidade)
R_min_max = minimo_maximo(temperatura, umidade)
R_max_prod = maximo_produto(temperatura, umidade)

# Exibindo os valores das matrizes
print("Matriz Máximo-Mínimo:")
print(R_max_min.round(2))

print("\nMatriz Mínimo-Máximo:")
print(R_min_max.round(2))

print("\nMatriz Máximo-Produto:")
print(R_max_prod.round(2))

# Função para plotar as matrizes
def plot_matriz(matriz, titulo):
    plt.figure(figsize=(6, 5))
    plt.imshow(matriz, cmap="Blues", interpolation="nearest")
    plt.colorbar(label="Grau de Pertinência")
    plt.title(titulo)
    plt.xlabel("Umidade")
    plt.ylabel("Temperatura")
    plt.xticks(range(3), ["Baixa", "Moderada", "Alta"])
    plt.yticks(range(3), ["Fria", "Morna", "Quente"])
    plt.show()

# Plotando as matrizes
plot_matriz(R_max_min, "Matriz Máximo-Mínimo")
plot_matriz(R_min_max, "Matriz Mínimo-Máximo")
plot_matriz(R_max_prod, "Matriz Máximo-Produto")

# Análise textual
print("\nAnálise comparativa das composições de relação fuzzy:")
print("1. Máximo-Mínimo: Reflete a interseção mais conservadora entre os conjuntos.")
print("2. Mínimo-Máximo: Reflete a união mais conservadora entre os conjuntos.")
print("3. Máximo-Produto: Permite suavização, considerando o produto dos graus de pertinência.")


# In[181]:


# Função para plotar as matrizes
def plot_matriz(matriz, titulo):
    plt.figure(figsize=(6, 5))
    plt.imshow(matriz, cmap="Blues", interpolation="nearest")
    plt.colorbar(label="Grau de Pertinência")
    plt.title(titulo)
    plt.xlabel("Colunas")
    plt.ylabel("Linhas")
    plt.savefig(f'{output_dir}/{titulo.replace(" ", "_").lower()}_matriz.png')  # Salva o gráfico
    plt.show()


# Exibindo os valores das matrizes
print("Matriz Máximo-Mínimo:")
print(R_max_min.round(2))

print("\nMatriz Mínimo-Máximo:")
print(R_min_max.round(2))

print("\nMatriz Máximo-Produto:")
print(R_max_prod.round(2))
# Plotando as matrizes
plot_matriz(R_max_min, "Matriz Máximo-Mínimo")
plot_matriz(R_min_max, "Matriz Mínimo-Máximo")
plot_matriz(R_max_prod, "Matriz Máximo-Produto")

# Análise textual
print("\nAnálise comparativa das composições de relação fuzzy:")
print("1. Máximo-Mínimo: Reflete a interseção mais conservadora entre os conjuntos.")
print("2. Mínimo-Máximo: Reflete a união mais conservadora entre os conjuntos.")
print("3. Máximo-Produto: Permite suavização, considerando o produto dos graus de pertinência.")


# In[ ]:




