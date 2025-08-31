
import pandas as pd
import numpy as np
from math import log2

cols_excluir = ["fecha"]  # columnas a excluir

def entropia(S, objetivo):
    col_objetivo = S[objetivo].dropna()  #! TEMPORAL borro NaN
    filas = len(col_objetivo)
    C = col_objetivo.unique() #caso binario [1,0]
    val = 0
    ocurrs = col_objetivo.value_counts() #cuenta ocurrencias de cada ci 
    for ci in C: #form explicita
        p_ci = ocurrs[ci] / filas
        val += - p_ci * log2(p_ci)
    return val



def ganancia(S, A, objetivo):
    # S: dataframe completo
    # A: nombre de la columna a evaluar
    S = S.dropna(subset=[A])  #! TEMPORAL borro NaN
    entropia_S = entropia(S, objetivo)
    col_A = S[A]
    filas = len(col_A)
    C = col_A.unique()
    val = 0
    ocurrs = col_A.value_counts()
    for ci in C:
        sv_s = ocurrs[ci] / filas #|Sv|/|S|
        entropia_S_ci = entropia(S[S[A] == ci], objetivo)
        val += sv_s * entropia_S_ci
    ganancia = entropia_S - val
    return ganancia


def ganancias(S, objetivo):
    atributos = list(S.columns)
    g = {}
    for a in atributos:
        if a != objetivo or a not in cols_excluir:
            g[a] = ganancia(S, a, objetivo)
    return g




