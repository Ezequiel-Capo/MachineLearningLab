import pandas as pd
import numpy as np

#====PREPROCESAMIENTO======

#=======Lluvia.csv=========
df_lluvia = pd.read_csv("Lluvia.csv", sep = ";") 

df_lluvia = df_lluvia.rename(columns = {
  "Estación" : "estacion",
  "Cod. Pluvio" : "cod_pluvio",
  "Fecha" : "fecha",
  "[mm]" : "precip_mm",
  "Comentario" : "coment_lluv"
}) 

#Arreglo los tipos
df_lluvia["fecha"] = pd.to_datetime(df_lluvia["fecha"], format = "%d/%m/%Y")
df_lluvia["precip_mm"] = (
  df_lluvia["precip_mm"].replace("TRAZA",0.0)
)
df_lluvia["precip_mm"] = df_lluvia["precip_mm"].astype(float)


#Criterio para dia lluvioso
precip_minima = 1.0
df_lluvia["dia_lluvioso"] = (df_lluvia["precip_mm"] >= precip_minima).astype(int)

#Criterio para mes lluvioso?
precip_mes_minima = 100.0
df_lluvia["mes"] = df_lluvia["fecha"].dt.month
df_lluvia["anio"] = df_lluvia["fecha"].dt.year #lo uso para identificar el mes según el año para diferenciarlos

group_by_cols = ["anio", "mes"]
df_mes = (
    df_lluvia
    .groupby(group_by_cols, as_index=False)["precip_mm"]
    .sum()
    .rename(columns = {"precip_mm": "precip_mm_mes"})
)

df_mes["mes_lluvioso"] = (df_mes["precip_mm_mes"] >= precip_mes_minima).astype(int)
df_lluvia = df_lluvia.merge(df_mes[group_by_cols + ["precip_mm_mes","mes_lluvioso"]], on=group_by_cols, how="left")

df_global = df_lluvia.copy()


#=====Humedad==========

df_hum = pd.read_csv("Humedad.csv", sep=";")
df_hum = df_hum.rename(columns = {
  "Fecha/hora" : "fecha",
  "HumedadCarrasco " : "hum_carrasco",
  "HumedadSalto " : "hum_salto",
})
df_hum["fecha"] = pd.to_datetime(df_hum["fecha"], errors="coerce")
df_hum["fecha"] = df_hum["fecha"].dt.date
df_hum["fecha"] = pd.to_datetime(df_hum["fecha"])
df_hum = df_hum.drop(["Unnamed: 3","Unnamed: 4"], axis=1)

df_hum = df_hum.groupby(df_hum["fecha"]).agg({
  "hum_carrasco" : "mean",
  "hum_salto" : "mean"
})

df_global = pd.merge(df_global, df_hum, on="fecha", how="left")

#debug
print(df_lluvia.head())
print(df_lluvia.dtypes)

print(df_hum.head())
print(df_hum.dtypes)

print(df_global.head)
print(df_hum.dtypes)