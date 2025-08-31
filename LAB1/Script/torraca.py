
import pandas as pd
import numpy as np
from math import log2

dataframes = {}
dataframes["data_dia"] = pd.read_csv("../Dataset_INUMET/data.csv", sep = ",", encoding='utf-8')

def cargarDataset(nombre):
    if nombre in dataframes:
        return dataframes[nombre]
    else:
        raise ValueError(f"Dataset '{nombre}' no encontrado.")

cols_excluir = ["dia_lluvioso", "fecha"]  # columnas a excluir

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

def ganancia(S, A, objetivo, entropia_S):
    # S: dataframe completo
    # A: nombre de la columna a evaluar
    S = S.dropna(subset=[A])  #! TEMPORAL borro NaN
    #entropia_S = entropia(S, objetivo)
    col_A = S[A]
    filas = len(col_A)
    C = col_A.unique()
    val = 0
    ocurrs = col_A.value_counts() #cuenta ocurrencias de cada ci
    for ci in C:
        sv_s = ocurrs[ci] / filas #|Sv|/|S|
        entropia_S_ci = entropia(S[S[A] == ci], objetivo) #filtro por ci, trae todo el dataset que cumple con ci para atributo A
        val += sv_s * entropia_S_ci
    ganancia = entropia_S - val
    return ganancia


def ganancias(S, objetivo):
    entropia_S = entropia(S, objetivo) #ionce
    atributos = list(S.columns)
    for col in cols_excluir:
        if col in atributos:
            atributos.remove(col)
    g = {}
    for a in atributos:
        g[a] = ganancia(S, a, objetivo, entropia_S)
    return g

def obtener_max_gan(S, A, objetivo):
    entropia_S = entropia(S, objetivo) #ionce
    g = {}
    for a in A:
        g[a] = ganancia(S, a, objetivo, entropia_S)
    mayor = max(g, key=g.get) #mayor por valor no por entrada
    return mayor

## def arbol
class Nodo:
    def __init__(self, atributo=None, valor=None, hijos=None, clasif=None):
        self.atributo = atributo  
        self.valor = valor        
        self.hijos = hijos if hijos is not None else [] 
        self.clasif = clasif     

    def es_hoja(self):
        return self.clasif is not None

## ID3
# Crear raiz
# Si todos los ej. tienen el mismo valor → etiquetar con ese valor
# • Si no me quedan atributos → etiquetar con el valor más común
# • En caso contrario:
# ‣ La raíz pregunta por A, atributo que mejor clasifica los ejemplos
# ‣ Para cada valor vi de A
# ๏ Genero una rama
# ๏ Ejemplosvi={ejemplos en los cuales A=vi }
# ๏ Si Ejemplosvi es vacío → etiquetar con el valor más probable
# ๏ En caso contrario → ID3(Ejemplosvi, Atributos -{A})
def generacion_ID3(S, A, objetivo):
    raiz = Nodo()
    valores_iguales = S[objetivo].nunique() == 1 # todos los ejemplos tienen el mismo valor
    if valores_iguales:
        raiz.clase = S[objetivo].iloc[0] #etiquetar con ese valor
        return raiz
    elif len(A) == 0: # no me quedan atributos
        #etiquetar con el valor más común
        clase_mas_comun = S[objetivo].mode()[0] #valor mas comun, usa moda, si hay empate devuelve el primero pos [0]
        raiz.clase = clase_mas_comun
        return raiz
    else:
        mejor_atr = obtener_max_gan(S, A, objetivo) #atributo que mejor clasifica los ejemplos
        raiz.atributo = mejor_atr 
        vals_atr = S[mejor_atr].value_counts().index.tolist() #valores del atributo
        for vi in vals_atr: #para cada valor vi de A 
            ej_vi = S[S[mejor_atr] == vi] 
            if ej_vi.empty:
                clase_mas_comun = S[objetivo].mode()[0]
                nodo_hijo = Nodo(clasif=clase_mas_comun)
                raiz.children[vi] = nodo_hijo
                continue
                #valor mas probable, de los q restan
            else:
                A_nuevos = A.copy()
                A_nuevos.remove(mejor_atr)
                nodo_hijo = generacion_ID3(ej_vi, A_nuevos, objetivo)
                raiz.children[vi] = nodo_hijo
    return raiz



        


