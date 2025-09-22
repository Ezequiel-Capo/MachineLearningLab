# MachineLearningLab

## Laboratorio 1: TorrAAca – Predicción Climática con Aprendizaje Automático

### Descripción

El objetivo de este laboratorio es aplicar técnicas de **aprendizaje automático supervisado** para predecir eventos climáticos en Uruguay, concretamente si un **día** o un **mes** será lluvioso, utilizando datos reales del **Instituto Nacional de Meteorología (INUMET)**.

El proyecto, llamado **TorrAAca**, integra:

- **Preprocesamiento de datos**: limpieza y filtrado de información histórica.
- **Ingeniería de atributos**: inclusión de fases de la luna, valores históricos y variables discretizadas.
- **Implementación de clasificadores**: tanto desde cero como utilizando `scikit-learn`.
- **Evaluación de modelos**: mediante validación cruzada, matrices de confusión y análisis de hiperparámetros.

---

### Ejecución

1. **Ubicación del dataset**: La ubicación está por defecto, pero puede alterarse en `LAB1/Script/preprocesamiento.ipynb`, en la línea ruta.

   - Por defecto relativa: `../Dataset_INUMET/` (llamado desde `LAB1/Script/`)

2. **Preprocesamiento**:

   - Ejecutar `preprocesamiento.ipynb` para generar los datasets filtrados y listos para entrenamiento.
   - Se debe disponer una carpeta: `LAB1/Datos_preprocesados/`, donde se almacenaran los datos ya preprocesados

3. **Evaluación de modelos**:  
   Existen cuatro scripts principales para evaluar los modelos según el tipo de clasificador y objetivo:

   - `eval_bayes_dia` y `eval_bayes_mes`: Evaluación de modelos **Naive Bayes** (implementación propia y `scikit-learn`) para los objetivos `dia_lluvioso` y `mes_lluvioso`.
   - `eval_arboles_dia` y `eval_arboles_mes`: Evaluación de modelos **Árbol de Decisión** y **Random Forest**, para los mismos objetivos.

4. **Código principal**:

   - `algoritmos_clasificacion.py` contiene toda la lógica de los algoritmos de clasificación, selección de atributos, validación de modelos por CV, testing, generación de gráficas y funciones auxiliares.

5. **Librerias**
   - Sickit Learn
   - Astral
   - Numpy
   - Pandas
