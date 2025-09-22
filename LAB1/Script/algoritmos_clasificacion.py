
import pandas as pd
import numpy as np
from math import log2
from graphviz import Digraph
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.metrics import (
    make_scorer, f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
import algoritmos_clasificacion as t
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#---------------------------------
# Carga de datos
dataframes = {}
dataframes["data_dia"] = pd.read_csv("../Datos_preprocesados/modelo1_vent1.csv", sep=",", encoding="utf-8")
dataframes["data_mes"] = pd.read_csv("../Datos_preprocesados/modelo1_vent30.csv", sep=",", encoding="utf-8")
dataframes["data_dia_c"] = pd.read_csv("../Datos_preprocesados/data_dia_c.csv", sep=",", encoding="utf-8")
dataframes["data_mes_c"] = pd.read_csv("../Datos_preprocesados/data_mes_c.csv", sep=",", encoding="utf-8")

# Inicializamos como vacío y lo llenamos cuando se carga un dataset
DOMINIOS = {}

def cargarDataset(nombre):
    global DOMINIOS  
    
    if nombre in dataframes:
        dataset = dataframes[nombre]
        # recalculamos los dominios dinámicamente
        DOMINIOS = {
            col: dataset[col].unique().tolist()
            for col in dataset.columns if col != "fecha"
        }
        return dataset
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
        else:   
            raiz.atributo = mejor_atr 
            #vals_atr = S[mejor_atr].value_counts().index.tolist() #valores del atributo, !cambiar por diccionario de Dominios!
            vals_atr = DOMINIOS[mejor_atr]
            for vi in vals_atr: #para cada valor vi de A 
                ej_vi = S[S[mejor_atr] == vi] 
                if ej_vi.empty:
                    clase_mas_comun = S[objetivo].mode()[0]
                    nodo_hijo = Nodo(clasif=clase_mas_comun)
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

class ID3Classifier(BaseEstimator, ClassifierMixin): #hereda de sklearn, para que fitee con las metricas y cv.
    def __init__(self, min_info_gain=0.0, criterio="gain_ratio"):
        self.min_info_gain = min_info_gain
        self.criterio = criterio
        self.arbol = None
        self.atributos = None
        self.objetivo = None

    def fit(self, X, y):
        S = X.copy() 
        self.objetivo = y.name
        S[self.objetivo] = y
        self.atributos = list(X.columns) 

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
    def __init__(self, m=1):
        """
        Clasificador Naive Bayes categórico con suavización m-estimate.

        Parámetros:
        -----------
        m : float
            Parámetro de suavización. Controla cuánto peso se da a la probabilidad a priori
            cuando los datos observados son escasos.
        """
        self.m = m
        self.probs_clase = {}
        self.probs_cond = {}
        self.classes_ = None

    def fit(self, X, y):
        """
        Ajusta el modelo con los datos de entrenamiento.

        Parámetros:
        -----------
        X : DataFrame de pandas
            Variables predictoras (ya preprocesadas con lags/ventanas si corresponde).
        y : Series o array
            Variable objetivo (clases).
        """
        self.classes_ = np.unique(y)
        n = len(y)

        # Probabilidad a priori de cada clase
        self.probs_clase = {c: (y == c).sum() / n for c in self.classes_}
        self.probs_cond = {c: {} for c in self.classes_}

        for col in X.columns:
            valores = X[col].unique()
            for c in self.classes_:
                X_c = X[y == c][col]
                n_c = len(X_c)
                self.probs_cond[c][col] = {}
                for v in valores:
                    count = (X_c == v).sum()
                    # suavización m-estimate
                    prob = (count + self.m / len(valores)) / (n_c + self.m)
                    self.probs_cond[c][col][v] = prob

        return self

    def predict(self, X):
        """
        Predice la clase más probable para cada fila de X.
        """
        preds = []
        for _, fila in X.iterrows():
            p = {}
            for c in self.classes_:
                log_prob = np.log(self.probs_clase[c])
                for col, val in fila.items():
                    # usa prob condicional si existe, si no aplica suavización mínima
                    prob = self.probs_cond[c][col].get(
                        val, 1 / (len(self.probs_cond[c][col]) + self.m)
                    )
                    log_prob += np.log(prob)
                p[c] = log_prob
            preds.append(max(p, key=p.get))
        return preds

    def predict_proba(self, X):
        """
        Devuelve las probabilidades normalizadas de cada clase.
        """
        probs = []
        for _, fila in X.iterrows():
            p = {}
            for c in self.classes_:
                log_prob = np.log(self.probs_clase[c])
                for col, val in fila.items():
                    prob = self.probs_cond[c][col].get(
                        val, 1 / (len(self.probs_cond[c][col]) + self.m)
                    )
                    log_prob += np.log(prob)
                p[c] = np.exp(log_prob)
            total = sum(p.values())
            probs.append([p[c] / total for c in self.classes_])
        return np.array(probs)

#Validar MOdel
def to_iter(x): 
    if isinstance(x, int):   
        return range(x)      
    return x 

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=20
    )
    
    return X_train, X_test, y_train, y_test

# def split_data(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=20
#     )

#     # Aplicar oversampling en el train. RES mejora recall, precision, pero baja entre un 20% el accuracy en testing
#     ros = RandomOverSampler(random_state=20)
#     X_train, y_train = ros.fit_resample(X_train, y_train)

#     return X_train, X_test, y_train, y_test

def validarModelo(X_train, y_train, random_state=10, n_splits=5, hiperparams=None, criterios=1):
    # Definir validación cruzada 
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []
    if criterios != 1:
        pos_label = 'si'
    else:
        pos_label = 1.0

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score, pos_label=pos_label),
        'recall': make_scorer(recall_score, pos_label=pos_label),
        'precision': make_scorer(precision_score, pos_label=pos_label)
    }
    for i in to_iter(criterios):
        for j in hiperparams:
            if criterios != 1:
                #DEBUG
                #print(f"--- Evaluando: Criterio={i}, Min_Info_Gain={j} ---")
                model = ID3Classifier(min_info_gain=j, criterio=i)
                cv_results = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring)
                row = { 'criterio': i, 'min_info_gain': j, 
                        'accuracy_mean': np.mean(cv_results['test_accuracy']),
                        'accuracy_std': np.std(cv_results['test_accuracy']),
                        'f1_mean': np.mean(cv_results['test_f1']),
                        'f1_std': np.std(cv_results['test_f1']),
                        'recall_mean': np.mean(cv_results['test_recall']),
                        'recall_std': np.std(cv_results['test_recall']),
                        'precision_mean': np.mean(cv_results['test_precision']),
                        'precision_std': np.std(cv_results['test_precision'])}
            else:
                #print(f"--- Evaluando: nro arboles={j} ---")
                model = RandomForestClassifier(n_estimators=j, random_state=random_state, class_weight="balanced")
                cv_results = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring)
                row = { 'n_estimators': j, 
                        'accuracy_mean': np.mean(cv_results['test_accuracy']),
                        'accuracy_std': np.std(cv_results['test_accuracy']),
                        'f1_mean': np.mean(cv_results['test_f1']),
                        'f1_std': np.std(cv_results['test_f1']),
                        'recall_mean': np.mean(cv_results['test_recall']),
                        'recall_std': np.std(cv_results['test_recall']),
                        'precision_mean': np.mean(cv_results['test_precision']),
                        'precision_std': np.std(cv_results['test_precision'])}
                
            results.append(row)

    df_results = pd.DataFrame(results)

    best_cfg_F = df_results.loc[df_results['f1_mean'].idxmax()]
    best_cfg_A = df_results.loc[df_results['accuracy_mean'].idxmax()]
    best_cfg_P = df_results.loc[df_results['precision_mean'].idxmax()]
    best_cfg_R = df_results.loc[df_results['recall_mean'].idxmax()]
    
    return df_results, best_cfg_F, best_cfg_A, best_cfg_P, best_cfg_R
    

def testearModelo(typeModel, best_cfg, X_train, y_train, X_test):
    match typeModel:
        case "ID3":
            model = ID3Classifier(
            min_info_gain=best_cfg['min_info_gain'],
            criterio=best_cfg['criterio']
            )
        case "RF":
            model = RandomForestClassifier(
                n_estimators=int(best_cfg['n_estimators']), 
            )
        case _:
            raise ValueError("Tipo de modelo no válido. Usa 'ID3' o 'RF")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def imprimirMetricas(typeModel, best_cfg, y_test, y_pred):
    match typeModel:
        case "ID3":
            print("mejor criterio: ", best_cfg['criterio'], "mejor min_info_gain: ", best_cfg['min_info_gain'])
            print("Accuracy :", accuracy_score(y_test, y_pred))
            print("F1       :", f1_score(y_test, y_pred, pos_label='si'))
            print("Recall   :", recall_score(y_test, y_pred, pos_label='si'))
            print("Precision:", precision_score(y_test, y_pred, pos_label='si'))
        case "RF":
            print("nro arboles: ", int(best_cfg['n_estimators']))
            print("Accuracy :", accuracy_score(y_test, y_pred))
            print("F1       :", f1_score(y_test, y_pred, pos_label=1.0))
            print("Recall   :", recall_score(y_test, y_pred, pos_label=1.0))
            print("Precision:", precision_score(y_test, y_pred, pos_label=1.0))
        case _:
            raise ValueError("Tipo de modelo no válido. Usa 'ID3' o 'RF'")

def print_CM(typeModel, y_test, y_pred):
    if typeModel == "ID3":
        labels = ['si', 'no']
        index = ["Actual: si", "Actual: no"]
        columns = ["Pred: si", "Pred: no"]
    elif typeModel == "RF":
        labels = [1.0, 0.0]  
        index = ["Actual: 1", "Actual: 0"]
        columns = ["Pred: 1", "Pred: 0"]
    else:
        raise ValueError("Tipo de modelo no válido")
    
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=index, columns=columns)
    print(cm_df)

def graficar_matriz_confusion(y_test, y_pred, labels=None, title="Matriz de Confusión"):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()



def funcGraficaError(df_results, metric="accuracy", modelo="ID3", ax=None):
    if metric in ["accuracy", "f1", "precision", "recall"]:
        values = 1 - df_results[f"{metric}_mean"]
        yerr = df_results[f"{metric}_std"]
        label = f"{metric.capitalize()} error (1 - {metric})"
    else:
        raise ValueError("Métrica no soportada, usa 'accuracy', 'f1', 'precision' o 'recall'")
    
    # Usar el eje si lo pasan
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,5))
    
    if modelo == "ID3":
        for crit in df_results["criterio"].unique():
            subset = df_results[df_results["criterio"] == crit]
            ax.errorbar(
                subset["min_info_gain"], 1 - subset[f"{metric}_mean"], 
                yerr=subset[f"{metric}_std"], 
                label=f"Criterio={crit}", marker="o", capsize=3
            )
        ax.set_xlabel("min_info_gain")
        
    elif modelo == "RF":  
        ax.errorbar(
            df_results["n_estimators"], values,
            yerr=yerr, 
            label="RandomForest", marker="o", capsize=3
        )
        ax.set_xlabel("n_estimators")
    
    ax.set_ylabel(label)
    ax.set_title(f"Curva de error vs hiperparámetros ({modelo})")
    ax.legend()
    ax.grid(True)


## Bayes version:
from sklearn.naive_bayes import MultinomialNB

def split_bayes_sklearn(ruta_dataset,obj):
    # Cargar dataset
    S = pd.read_csv(ruta_dataset)

    # Target y features
    y = S[obj]
    X = S.drop(columns=["dia_lluvioso","mes_lluvioso","fecha","amplitud_carrasco","amplitud_salto",
                        "tprom_carrasco","tprom_salto","fase_lunar",
                        "int_viento_carrasco_max","int_racha_viento_carrasco_max",
                        "int_viento_salto_max","int_racha_viento_salto_max"])

    # Convertir categorías a dummies (one-hot)
    X_enc = pd.get_dummies(X.astype(str))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

def validarNaiveBayes_sklearn(X_train, y_train, hiperparams=None, random_state=42, n_splits=5, pos_label='si', nd=1):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score, pos_label=pos_label),
        'recall': make_scorer(recall_score, pos_label=pos_label),
        'precision': make_scorer(precision_score, pos_label=pos_label)
    }

    for alpha in hiperparams:
        # Crear modelo
        model = MultinomialNB(alpha=alpha)

        # Validación cruzada
        cv_results = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring)

        # Guardar resultados
        row = {
            'alpha': alpha,
            'nd': nd,
            'accuracy_mean': np.mean(cv_results['test_accuracy']),
            'accuracy_std': np.std(cv_results['test_accuracy']),
            'f1_mean': np.mean(cv_results['test_f1']),
            'f1_std': np.std(cv_results['test_f1']),
            'recall_mean': np.mean(cv_results['test_recall']),
            'recall_std': np.std(cv_results['test_recall']),
            'precision_mean': np.mean(cv_results['test_precision']),
            'precision_std': np.std(cv_results['test_precision']),
        }
        results.append(row)

    df_results = pd.DataFrame(results)


    return df_results

def testearBayesSickit(best_cfg, X_train, y_train, X_test, y_test):
    model = MultinomialNB(alpha=best_cfg['alpha'])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calcular métricas
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label="si")
    rec = recall_score(y_test, y_pred, pos_label="si")
    prec = precision_score(y_test, y_pred, pos_label="si")

    
    return {
        "y_pred": y_pred,
        "accuracy": acc,
        "f1": f1,
        "recall": rec,
        "precision": prec,
    }


def split_bayes(ruta_dataset, obj):

    # Cargar dataset
    S = pd.read_csv(ruta_dataset)

    # Features y target
    
    y = S[obj]
    X = S.drop(columns=["dia_lluvioso","mes_lluvioso","fecha", "amplitud_carrasco","amplitud_salto","tprom_carrasco","tprom_salto", "fase_lunar", 
                        "int_viento_carrasco_max", "int_racha_viento_carrasco_max", "int_viento_salto_max", "int_racha_viento_salto_max"
                        ])

    # Train/test split
    X_train, X_test, y_train, y_test = t.split_data(X, y)

    return X_train, X_test, y_train, y_test

def validarNaiveBayes(X_train, y_train, hiperparams=None, random_state=42, n_splits=5, pos_label='si', nd=1):

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score, pos_label=pos_label),
        'recall': make_scorer(recall_score, pos_label=pos_label),
        'precision': make_scorer(precision_score, pos_label=pos_label)
    }

    for m in hiperparams:
        # Crear modelo
        model = t.NaiveBayesClassifier(m=m)
        
        # CV
        cv_results = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring)

        # Guardar resultados
        row = {
            'm': m,
            'nd': nd,
            'accuracy_mean': np.mean(cv_results['test_accuracy']),
            'accuracy_std': np.std(cv_results['test_accuracy']),
            'f1_mean': np.mean(cv_results['test_f1']),
            'f1_std': np.std(cv_results['test_f1']),
            'recall_mean': np.mean(cv_results['test_recall']),
            'recall_std': np.std(cv_results['test_recall']),
            'precision_mean': np.mean(cv_results['test_precision']),
            'precision_std': np.std(cv_results['test_precision']),
        }
        results.append(row)

    df_results = pd.DataFrame(results)

    return df_results

def testearBayes(best_cfg, X_train, y_train, X_test, y_test):
    
    model = t.NaiveBayesClassifier(m=best_cfg['m'])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calcular métricas
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label="si")
    rec = recall_score(y_test, y_pred, pos_label="si")
    prec = precision_score(y_test, y_pred, pos_label="si")


    return {
        "y_pred": y_pred,
        "accuracy": acc,
        "f1": f1,
        "recall": rec,
        "precision": prec,
    }

