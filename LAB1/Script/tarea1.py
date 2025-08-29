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


df_global = df_lluvia

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

#Genero a partir de los datos un solo dato por dia, usando el promedio de las medidas
df_hum = df_hum.groupby(df_hum["fecha"]).agg({
  "hum_carrasco" : "mean",
  "hum_salto" : "mean"
})

df_global = pd.merge(df_global, df_hum, on="fecha", how="left")

#=================

#debug
print(df_lluvia.head())
print(df_lluvia.dtypes)

print(df_hum.head())
print(df_hum.dtypes)

print(df_global.head)
print(df_hum.dtypes)