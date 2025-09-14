
import pandas as pd
import numpy as np
from math import log2
from graphviz import Digraph
from sklearn.base import BaseEstimator, ClassifierMixin
#---------------------------------
# Carga de datos
dataframes = {}
dataframes["data_dia"] = pd.read_csv("../Dataset_INUMET/data.csv", sep = ",", encoding='utf-8')
dataframes["data_dia_rf"] = pd.read_csv("../Dataset_INUMET/dataRandomForest.csv", sep = ",", encoding='utf-8')

DOMINIOS = {col: dataframes["data_dia"][col].unique().tolist() for col in dataframes["data_dia"].columns if col != "fecha"}

def cargarDataset(nombre):
    if nombre in dataframes:
        return dataframes[nombre]
    else:
        raise ValueError(f"Dataset '{nombre}' no encontrado.")


## Método 1 : Ganancia 
#S: dataset
#objetivo: clasificacion
def entropia(S, objetivo):
    col_objetivo = S[objetivo] 
    filas = len(S[objetivo])
    C = col_objetivo.unique() #caso binario [1,0]
    val = 0
    ocurrs = col_objetivo.value_counts() #cuenta ocurrencias de cada ci 
    for ci in C: #form explicita
        p_ci = ocurrs[ci] / filas
        val += - p_ci * log2(p_ci)
    return val

#A: Atributo 
#S: dataset
#objetivo: clasificacion
def ganancia(S, A, objetivo, entropia_S):
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

#sA: Set de atributos 
#S: dataset
#objetivo: clasificacion
def get_gains(S, sA, objetivo):
    entropia_S = entropia(S, objetivo) #ionce
    g = {}
    for a in sA:
        g[a] = ganancia(S, a, objetivo, entropia_S)
    return g
#---------------------------------

## Gain Ratio y SplitInfo

#A: Atributo
#S: dataset
def split_info(S, A):
    col_A = S[A]
    filas = len(col_A)
    C = col_A.unique() #dominio de A(registrado)
    val = 0
    ocurrs = col_A.value_counts() #cuenta ocurrencias de cada ci
    for ci in C:
        sv_s = ocurrs[ci] / filas #|Sv|/|S|
        val += - sv_s * log2(sv_s)
    return val

def gain_ratio(S, A, objetivo, entropia_S): 
    g = ganancia(S, A, objetivo, entropia_S)
    si = split_info(S, A)
    if si == 0:
        return 0
    else:
        return g / si

def get_gain_ratios(S, sA, objetivo):
    entropia_S = entropia(S, objetivo) #once
    gr = {}
    for a in sA:
        gr[a] = gain_ratio(S, a, objetivo, entropia_S)
    return gr

#--------------------------------

## Impurity Reduction (Gini)
#S: dataset
#objetivo: clasificacion
def gini(S, objetivo):
    col_objetivo = S[objetivo] 
    filas = len(S[objetivo])
    C = col_objetivo.unique() #caso binario [1,0]
    val = 1
    ocurrs = col_objetivo.value_counts() #cuenta ocurrencias de cada ci 
    for ci in C: #general, caso binario (-p+^2 - p-^2)
        p_ci = ocurrs[ci] / filas
        val -= p_ci ** 2
    return val

#A: Atributo
#S: dataset
#objetivo: clasificacion
def impurity_reduction(S, A, objetivo, gini_S):
    col_A = S[A]
    filas = len(col_A)
    C = col_A.unique()
    val = 0
    ocurrs = col_A.value_counts() #cuenta ocurrencias de cada ci
    for ci in C:
        sv_s = ocurrs[ci] / filas #|Sv|/|S|
        gini_S_ci = gini(S[S[A] == ci], objetivo) #filtro por ci, trae todo el dataset que cumple con ci para atributo A
        val += sv_s * gini_S_ci
    ir = gini_S - val
    return ir

#sA: Set de atributos
#S: dataset
#objetivo: clasificacion
def get_max_irs(S, sA, objetivo):
    gini_S = gini(S, objetivo) #once
    ir = {}
    for a in sA:
        ir[a] = impurity_reduction(S, a, objetivo, gini_S)
    return ir


## def arbol
class Nodo:
    def __init__(self, atributo=None, valor=None, hijos=None, clasif=None):
        self.atributo = atributo  
        self.valor = valor        
        self.hijos = hijos if hijos is not None else {}
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
def generacion_ID3(S, A, objetivo, min_info_gain, criterio):
    raiz = Nodo()
    valores_iguales = S[objetivo].nunique() == 1 # todos los ejemplos tienen el mismo valor
    if valores_iguales:
        raiz.clasif = S[objetivo].iloc[0] #etiquetar con ese valor
        return raiz
    elif len(A) == 0: # no me quedan atributos
        #etiquetar con el valor más común
        clase_mas_comun = S[objetivo].mode()[0] #valor mas comun, usa moda, si hay empate devuelve el primero pos [0]
        raiz.clasif = clase_mas_comun
        return raiz
    else:
        match criterio:
            case "gain":
                ranking_atrs = get_gains(S, A, objetivo)
            case "gini":
                ranking_atrs = get_max_irs(S, A, objetivo)
            case "gain_ratio":
                ranking_atrs = get_gain_ratios(S, A, objetivo)
            case _:
                raise ValueError("Criterio no válido. Usa 'gain', 'gini' o 'gain_ratio'.")
        mejor_atr = max(ranking_atrs, key=ranking_atrs.get) #mejor atributo
        ganancia_max = ranking_atrs[mejor_atr]
        if ganancia_max < min_info_gain:
            clase_mas_comun = S[objetivo].mode()[0]
            raiz.clasif = clase_mas_comun
            return raiz
        raiz.atributo = mejor_atr 
        #vals_atr = S[mejor_atr].value_counts().index.tolist() #valores del atributo, !cambiar por diccionario de Dominios!
        vals_atr = DOMINIOS[mejor_atr]
        for vi in vals_atr: #para cada valor vi de A 
            ej_vi = S[S[mejor_atr] == vi] 
            if ej_vi.empty:
                clase_mas_comun = S[objetivo].mode()[0]
                nodo_hijo = Nodo(clasif=clase_mas_comun)
                raiz.hijos[vi] = nodo_hijo
                continue
                #valor mas probable, de los q restan
            else:
                A_nuevos = A.copy()
                A_nuevos.remove(mejor_atr)
                nodo_hijo = generacion_ID3(ej_vi, A_nuevos, objetivo, min_info_gain, criterio)
                raiz.hijos[vi] = nodo_hijo
    return raiz


#Imprime el árbol usando graphviz
def graficar_arbol(nodo, dot=None, nombre="root"):
    if dot is None:
        dot = Digraph()
    label = str(nodo.clasif).upper()
    if nodo.es_hoja():
        if nodo.clasif == "si":
            color = "lightgreen"
        else:
            color = "lightcoral"  
        dot.node(nombre, label, shape="box", style="filled", color=color)
    else:
        dot.node(nombre, f"Atributo: {nodo.atributo}")
        for valor, hijo in nodo.hijos.items():
            hijo_nombre = f"{nombre}_{valor}"
            dot.edge(nombre, hijo_nombre, label=str(valor))
            graficar_arbol(hijo, dot, hijo_nombre)
    return dot

#Prediccion/Clasificacion
def predecir(nodo, ejemplo):
    if nodo.es_hoja():
        return nodo.clasif
    valor = ejemplo[nodo.atributo]
    if valor in nodo.hijos:
        return predecir(nodo.hijos[valor], ejemplo)
    else: 
        raise ValueError(f"Valor '{valor}' del atributo '{nodo.atributo}' no encontrado en el árbol.")
    
class ID3Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, min_info_gain=0.0, criterio="gain_ratio"):
        self.min_info_gain = min_info_gain
        self.criterio = criterio
        self.arbol = None
        self.atributos = None
        self.objetivo = None

    def fit(self, X, y): #par generar valores y depositarlos en los atributos de la clase, que sea compatible con sklearn
        # Combino X e y en un solo DataFrame (como espera tu generacion_ID3)
        S = X.copy() 
        self.objetivo = y.name
        S[self.objetivo] = y
        self.atributos = list(X.columns) 

        # Genero árbol con tu función
        self.arbol = generacion_ID3(
            S,
            self.atributos.copy(),
            self.objetivo,
            self.min_info_gain,
            self.criterio
        )
        return self

    def predict(self, X):
        preds = []
        for _, row in X.iterrows(): #iterar sobre los valores
            preds.append(predecir(self.arbol, row))
        return preds



class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, m = 1, nd = 3):
        self.m = m 
        self.nd = nd
        self.probs_clase = {}
        self.probs_cond = {}
        self.classes = None


    def fit(self, X, y):
        self.classes = np.unique(y)
        n = len(y)

        self.probs_clase = {c: (y == c).sum() / n for c in self.classes}
        self.probs_cond = {c: {} for c in self.classes}
        for col in X.columns:
            valores = X[col].unique()
            for c in self.classes:
                X_c = X[y == c][col]
                n_c = len(X_c)
                self.probs_cond[c][col] = {}
                for v in valores:
                    count = (X_c == v).sum()
                    prob = (count + self.m / len(valores)) / (n_c + self.m)
                    self.probs_cond[c][col][v] = prob  
        return self

    def predict(self, X):
        preds = []
        for _, fila in X.iterrows():
            p = {}
            for c in self.classes:
                log_prob = np.log(self.probs_clase[c])
                for col, val in fila.items():
                    if val in self.probs_cond[c][col]:
                        prob = self.probs_cond[c][col][val]
                    else:
                        prob = 1 / (len(self.probs_cond[c][col]) + self.m)

                    log_prob += np.log(prob)
                p[c] = log_prob
            preds.append(max(p, key=p.get))
        return preds
    
    def predict_proba(self, X):
        probs = []
        for _, fila in X.iterrows():
            p = {}
            for c in self.classes:
                log_prob = np.log(self.probs_clase[c])
                for col, val in fila.items():
                    prob = self.probs_cond[c][col].get(
                        val, 1 / (len(self.probs_cond[c][col]) + self.m)
                    )
                    log_prob += np.log(prob)
                p[c] = np.exp(log_prob)
            total = sum(p.values())
            probs.append([p[c] / total for c in self.classes])
        return np.array(probs)

    