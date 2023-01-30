# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:50:53 2023

@author: REGIS CARDOSO
"""


######################################################################################################
## ANÁLISE DE VIBRAÇÃO REAL DE UM MOTOR - MUITO UTILIZADO PARA MANUTENÇÃO PREDITIVA ###
######################################################################################################

## IMPORTAR AS BIBLIOTECAS UTILIZADAS ###

import pandas as pd
import numpy as np
import statistics
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import timedelta, datetime
from scipy.fftpack import fft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler


## FUNÇÕES

# FUNÇÃO PARA FEATURE ENGINEERING VIA TRANSFORMADA RÁPIDA DE FOURIER - FFT


def FFT(df):

    df_Final = df

    graf_x = df_Final['Tempo'].values
    graf_y = df_Final['Valor'].values

    x = graf_x

    y = graf_y

    N = len(graf_x)

    T = x[1] - x[0]

    Fs = 1 / T

    yf = 2.0 / N * np.abs(fft(y)[0:N // 2])

    xf = fftfreq(N, T)[:N // 2]

    verX = []
    verY = []

    obs = len(yf)

    for i in range(1, obs, 1):
        verX.insert(i, xf[i])
        verY.insert(i, yf[i])
        
    df_FFT = []
    df_FFT = pd.DataFrame(df_FFT)

    df_FFT['Frequencia'] = verX
    df_FFT['Amplitude'] = verY

    return (df_FFT)


# FUNÇÃO PARA ORGANIZAR / CRIAR O DATASET DE VIBRAÇÃO COM E SEM DEFEITOS

def organiza_df(df):

    df.columns = ['Tempo', 'Valor', 'ValorY', 'ValorXZ']

    df_final = []
    df_final = pd.DataFrame(df_final)
    
    df_final['Frequencia'] = 0
    
    df_tempo = df.loc[0:1000]
    
    tempo_sequencial = []
    
    for i in range(len(df_tempo)):
        tempo_sequencial.append(round(i * (1/10000),5))
          
    
    df_tempo['Tempo'] = tempo_sequencial
    
    df_FFT = FFT(df_tempo)
    
    df_final['Frequencia'] = df_FFT['Frequencia']

   
    
        
    for j in range(0,50,1):
        
        
        df_final['Data'+str(j)] = 0
        
        df_calculo_fft = df.loc[j*1000:(j+1)*1000]
        
        tempo_sequencial = []
    
        for i in range(len(df_calculo_fft)):
            tempo_sequencial.append(round(i * (1/10000),5))
              
        
        df_calculo_fft['Tempo'] = tempo_sequencial
            
        df_FFT = FFT(df_calculo_fft)
        
            
        df_final['Data'+str(j)] = df_FFT['Amplitude']
    
    
    return (df_final)
    
        
## IMPORTAR OS ARQUIVOS DE DADOS ###
## Dados utilizados do link: https://data.mendeley.com/datasets/fm6xzxnf36/2
        
df_inicial_1 = pd.read_csv('healthy without pulley.csv', sep=',')

df_inicial_2 = pd.read_csv('1.1inner-100watt.csv', sep=',')

df_inicial_3 = pd.read_csv('1.7inner-300watt.csv', sep=',')


## ORGANIZAR / CRIAR O DATASET DE VIBRAÇÃO COM E SEM DEFEITOS, COMO RETORNO TEMOS A FFT DOS SINAIS ###

df_FFT_1 = organiza_df(df_inicial_1)
    
df_FFT_2 = organiza_df(df_inicial_2)

df_FFT_3 = organiza_df(df_inicial_3)


## JUNTAR OS DADOS ORGANIZADOS PARA CRIAR UM ÚNICO DATASET COM DEFEITOS E SEM DEFEITOS PELA ANÁLISE DE VIBRAÇÃO ###

df_novo_parcial = pd.merge(df_FFT_1, df_FFT_2, how = 'inner', on = 'Frequencia')

df_novo = pd.merge(df_novo_parcial, df_FFT_3, how = 'inner', on = 'Frequencia')


## ORGANIZAR / CRIAR O DATASET DE VIBRAÇÃO COM E SEM DEFEITOS ###

# É NECESSÁRIO FAZER COM QUE AS FREQUENCIA DA FFT SEJAM AS FEATURES DO SINAL

# PARA ISSO, FAZEMOS A MATRIZ TRANSPOSTA DOS DADOS, INVERTENDO AS LINHAS PELAS COLUNAS


df_novo_transposto = np.transpose(df_novo)


# DEFININDO AS FREQUENCIA COMO NOMES PARA AS FEATURES / COLUNAS

df_novo_transposto.columns = round(df_FFT_1['Frequencia'],5)

df_novo_transposto = df_novo_transposto.reset_index()

df_novo_transposto = df_novo_transposto.drop(columns=['index'])


# ELIMINANDO A PRIMEIRA LINHA, POIS SE TRATA DAS FREQUENCIAS CALCULADAS PELA FFT

df_novo_transposto_final = df_novo_transposto.loc[1:]


## NORMALIZANDO OS DADOS ###

scaler = MaxAbsScaler()  
scaler.fit(df_novo_transposto_final)
df_novo_transposto_final_NORMALIZADO = scaler.transform(df_novo_transposto_final)


## TREINANDO O MODELO, AQUI FOI UTILIZADO O MODELO DE KMEANS, MÉTODO NÃO SUPERVISIONADO ###

from sklearn.cluster import KMeans

## VERIFICANDO A QUANTIDADE DE CLUSTER PARA O ALGORITMO DE KMEANS, ATRAVÉS DO MÉTODO DO COTOVELO ###

## Metodo Baseado em: https://medium.com/pizzadedados/kmeans-e-metodo-do-cotovelo-94ded9fdf3a9#:~:text=Se%20voc%C3%AA%20n%C3%A3o%20est%C3%A1%20familiarizado,tentar%C3%A1%20encontrar%208%20agrupamentos%20distintos.


def calculate_wcss(data):
    wcss = []
    for n in range(2, 21):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X=data)
        wcss.append(kmeans.inertia_)

    return wcss


def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2


wcss = calculate_wcss(df_novo_transposto_final_NORMALIZADO)

clusters = optimal_number_of_clusters(wcss)


#clusters = 3

## TREINANDO O ALGORITMO DE KMEANS ###

kmeans = KMeans(n_clusters=clusters)

kmeans.fit(df_novo_transposto_final_NORMALIZADO)

y_kmeans = kmeans.predict(df_novo_transposto_final_NORMALIZADO)

# y_kmens tratam-se da separação que o modelo de kmeans realizou

# para melhor vizualizarmos, vamos fazer algumas interações
# vamos rotular os dados apenas para verificar como ficou a acurácia do kmeans
# lembrando que kmeans é um algoritmo não supervisionado, logo, não saberiamos os rótulos


## ROTULANDO OS DADOS COMO 0, 1 E 3 ###
# verificar se a ordem está correta após rodar a primeira vez o algoritmo

semdefeito = []
for i in range(int(len(df_novo_transposto_final_NORMALIZADO)/3)):
    semdefeito.append(1)
    
comdefeito1 = []
for i in range(int(len(df_novo_transposto_final_NORMALIZADO)/3)):
    comdefeito1.append(2)


comdefeito2 = []
for i in range(int(len(df_novo_transposto_final_NORMALIZADO)/3)):
    comdefeito2.append(0)


## JUNTANDO OS DADOS EM UMA MESMA LISTA ###
    
y_teste = semdefeito + comdefeito1 + comdefeito2


## AVALIANDO A ACURÁCIA, COM A MATRIZ DE CONFUSÃO ###

from sklearn.metrics import confusion_matrix
import seaborn as sns


fig = plt.figure(figsize=(6,6))
cm = confusion_matrix(y_kmeans, y_teste)


ax = plt.subplot()
sns.set(font_scale=2.0) # Adjust to fit
sns.heatmap(cm, annot=True, ax=ax, cmap="Blues", fmt="g");  

# Labels, title and ticks
label_font = {'size':'12'}  # Adjust to fit
ax.set_xlabel('Previsto', fontdict=label_font);
ax.set_ylabel('Observado', fontdict=label_font);

title_font = {'size':'12'}  # Adjust to fit
ax.set_title('Confusion Matrix', fontdict=title_font);

ax.tick_params(axis='both', which='major', labelsize=12)  # Adjust to fit
ax.xaxis.set_ticklabels(['Sem Defeito', 'Com Defeito 1', 'Com Defeito 2']);
ax.yaxis.set_ticklabels(['Sem Defeito', 'Com Defeito 1', 'Com Defeito 2']);
plt.show()


## REALIZANDO A ANÁLISE DAS COMPONENTES PRINCIPAIS ###

# O MÉTODO DE KMEANS UTILIZA A ANÁLISE DAS COMPONENTES PRINCIPAIS PARA AGRUPAR OS SEUS DADOS 
# PARA VERIFICAR COMO FICOU A DISTRIBUIÇÃO, VAMOS REALIZAR MANUALMENTE A PCA
# DEPOIS VAMOS PLOTAR CONFORME A CLASSIFICAÇÃO FEITA PELO KMEANS


pca_componentes = PCA(n_components=2)

principalComponents = pca_componentes.fit_transform(df_novo_transposto_final_NORMALIZADO)

principal_df = pd.DataFrame(data = principalComponents, columns = ['pca1', 'pca2'])


## PLOTANDO OS DADOS DE PCA E A SEPARAÇÃO FEITA POR KMEANS ###

plt.figure(figsize=(10, 8))
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.scatter(principal_df['pca1'].loc[0:49], principal_df['pca2'].loc[0:49], c='g', s=50)
plt.scatter(principal_df['pca1'].loc[50:98], principal_df['pca2'].loc[50:98], c='r', s=50)
plt.scatter(principal_df['pca1'].loc[99:49*3], principal_df['pca2'].loc[99:49*3], c='b', s=50)
plt.legend(['Sem Defeito', 'Com defeito 1', 'Com defeito 2'], loc='best')
plt.show()

