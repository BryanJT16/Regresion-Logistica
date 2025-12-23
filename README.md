# Campaña de Marketing Bancario

## 📘 Descripción del Proyecto

Este proyecto tiene como objetivo analizar una campaña de marketing telefónica realizada por un banco portugués, con el fin de identificar a los clientes con mayor probabilidad de suscribir un **depósito a largo plazo**.  
A través de técnicas de **análisis exploratorio de datos (EDA)** y **modelado predictivo**, se busca optimizar los esfuerzos de marketing y mejorar la rentabilidad de las campañas.

---

## 🧩 Contexto Empresarial

Los depósitos a largo plazo permiten a las instituciones bancarias retener fondos de los clientes durante un período determinado, lo que mejora la capacidad de inversión del banco.  
Las campañas de marketing de este tipo de producto se llevan a cabo principalmente mediante llamadas telefónicas. Si un cliente no se encuentra disponible en el momento del contacto, se le vuelve a llamar posteriormente.

Debido a una disminución en los ingresos, el banco desea enfocar sus recursos en aquellos clientes con **mayor probabilidad de aceptar la oferta**, evitando pérdidas de tiempo y dinero.

---

## 🎯 Objetivos

- Analizar los datos históricos de campañas de marketing anteriores.  
- Identificar patrones y variables más relevantes que influyen en la suscripción del producto.  
- Construir un modelo predictivo que permita estimar la probabilidad de éxito de futuras campañas.  
- Presentar conclusiones y recomendaciones basadas en los resultados obtenidos.
- Crear un ejecutable que pruebe el modelo

---

### Resumen de Características

| Columna | Tipo de Dato | Recuento No Nulo | Descripción |
| :--- | :--- | :--- | :--- |
| **age** | `int64` | 41188 | Edad del cliente. |
| **job** | `object` | 41188 | Tipo de trabajo (e.g., 'admin.', 'blue-collar'). |
| **marital** | `object` | 41188 | Estado civil. |
| **education** | `object` | 41188 | Nivel educativo. |
| **default** | `object` | 41188 | ¿Tiene crédito en mora? ('yes', 'no', 'unknown'). |
| **housing** | `object` | 41188 | ¿Tiene préstamo hipotecario? ('yes', 'no', 'unknown'). |
| **loan** | `object` | 41188 | ¿Tiene préstamo personal? ('yes', 'no', 'unknown'). |
| **contact** | `object` | 41188 | Tipo de comunicación de contacto ('cellular', 'telephone'). |
| **month** | `object` | 41188 | Último mes de contacto del año. |
| **day_of_week** | `object` | 41188 | Último día de contacto de la semana. |
| **duration** | `int64` | 41188 | Duración del último contacto, en segundos (variable muy importante y debe eliminarse después del contacto). |
| **campaign** | `int64` | 41188 | Número de contactos realizados durante esta campaña para este cliente. |
| **pdays** | `int64` | 41188 | Número de días transcurridos desde el último contacto de la campaña anterior (999 significa que el cliente no fue contactado previamente). |
| **previous** | `int64` | 41188 | Número de contactos realizados antes de esta campaña para este cliente. |
| **poutcome** | `object` | 41188 | Resultado de la campaña de marketing anterior ('failure', 'nonexistent', 'success'). |
| **emp.var.rate** | `float64` | 41188 | Tasa de variación del empleo (indicador trimestral). |
| **cons.price.idx** | `float64` | 41188 | Índice de precios al consumidor (indicador mensual). |
| **cons.conf.idx** | `float64` | 41188 | Índice de confianza del consumidor (indicador mensual). |
| **euribor3m** | `float64` | 41188 | Tasa Euribor a 3 meses (indicador diario). |
| **nr.employed** | `float64` | 41188 | Número de empleados (indicador trimestral). |
| **y** | `object` | 41188 | **Variable objetivo:** ¿Se suscribió a un depósito a largo plazo? ('yes' o 'no'). |

---

## 🚀 Metodología

### 1. Preprocesamiento de Datos
* **Eliminación de Duplicados:** Se eliminaron **12** filas duplicadas encontradas en el conjunto de datos.
* **Codificación de Variables Categóricas:** Las 11 columnas de tipo `object` (categóricas), incluyendo la variable objetivo `y`, fueron convertidas a formato numérico utilizando **Label Encoding** para hacerlas aptas para el modelo de Regresión Logística.
* **Escalado de Características:** Las características continuas y numéricas (`age`, `duration`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`) fueron **escaladas** para estandarizar su rango de valores.

### 2. Entrenamiento del Modelo
* **Algoritmo:** **Regresión Logística** (`LogisticRegression`).
* **Búsqueda de Hiperparámetros:** Se utilizó `GridSearchCV` para encontrar la combinación óptima de hiperparámetros que maximice el *accuracy* del modelo en el conjunto de entrenamiento.
    * **Hiperparámetros explorados:**
        ```python
        hyperparams = {
            "penalty": ['l1', 'l2', 'elasticnet', None],
            "dual": [True, False],
            "C": [1.0, 0.5, 0.05, 0.10, 1.5, 2.0]
        }
        ```
    * **Mejores Hiperparámetros encontrados:** `{'C': 1.0, 'dual': False, 'penalty': 'l1'}` (utilizando `solver='liblinear'`).

## 📚 ¿Qué es la Regresión Logística y cómo se usa en este proyecto?

La Regresión Logística es un modelo estadístico y de aprendizaje automático utilizado para clasificar observaciones en dos o más categorías.
A diferencia de la Regresión Lineal, que predice valores continuos, la Regresión Logística predice probabilidades que luego se convierten en clases (por ejemplo, sí o no) utilizando un umbral, normalmente de 0.5.

### 🧠 Concepto Básico

El modelo estima la probabilidad de que ocurra un evento (por ejemplo, que un cliente acepte un depósito a largo plazo) a partir de una combinación lineal de variables de entrada, como la edad, la ocupación o la duración de la llamada.

Matemáticamente se expresa como:

P(y = 1 | X) = 1 / (1 + e^-(β0 + β1x1 + β2x2 + ... + βnxn))

Donde:

* P(y=1∣X) → es la probabilidad de que el cliente acepte la oferta.

* xᵢ → son las variables predictoras (características del cliente).

* βᵢ →son los coeficientes aprendidos por el modelo.

Si la probabilidad resultante es mayor o igual a 0.5, el modelo clasifica el resultado como “Sí” (el cliente acepta); en caso contrario, como “No”.

### 🎯 ¿Por qué se usa?

* Permite modelar relaciones entre variables numéricas y categóricas.

* Ofrece resultados fáciles de interpretar, ya que cada coeficiente indica el peso o influencia de una variable sobre el resultado.

* Proporciona probabilidades, no solo clasificaciones, lo que ayuda a medir el grado de confianza de cada predicción.

* Es eficiente computacionalmente, ideal para proyectos de análisis y predicción como este.

### ⚙️ Aplicación en este Proyecto

En este proyecto, la Regresión Logística se utiliza como una herramienta para entender y predecir el comportamiento de los clientes durante las campañas de marketing.

El modelo aprende a partir de los datos históricos del banco, que incluyen información sobre clientes anteriores y si aceptaron o no el depósito a largo plazo.
Durante el entrenamiento, la regresión logística analiza las relaciones entre las variables: por ejemplo, puede detectar que las personas con empleo estable, buena situación económica y llamadas más largas tienden a aceptar el producto.

Una vez entrenado, el modelo puede recibir datos de un nuevo cliente y, con base en patrones aprendidos, calcular la probabilidad de que acepte la oferta.
De esta forma, el banco puede:

* Priorizar a los clientes más propensos a decir que sí.

* Evitar contactar repetidamente a quienes es muy poco probable que acepten.

* Entender qué características están más relacionadas con el éxito de una campaña.

En resumen, la regresión logística convierte un conjunto de datos complejos en una herramienta práctica de decisión, ayudando a planificar las campañas de manera más eficiente y rentable.


## 🎯 Resultados

El modelo final, ajustado con los mejores hiperparámetros, arrojó una precisión (accuracy) máxima de **0.90892**.

El rendimiento del modelo en la matriz de confusión es (valores de ejemplo del output):

| Predicción | Real: No | Real: Sí |
| :--- | :--- | :--- |
| **Predicción: No** (Verdaderos Negativos: TN) | **7129** | Falsos Negativos (FN): 169 |
| **Predicción: Sí** (Falsos Positivos: FP) | 558 | **Verdaderos Positivos** (TP): 382 |

*(Nota: Los valores de la matriz de confusión aquí son de una validación específica y pueden variar)*

---

## 🧠 Tecnologías Utilizadas

- **Python**  
- **Pandas**, **NumPy** – para manipulación y limpieza de datos  
- **Matplotlib**, **Seaborn** – para visualización de datos  
- **Scikit-learn** – para la creación y evaluación de modelos predictivos  
- **Jupyter Notebook** – entorno de desarrollo interactivo  

---

## 📊 Resultados Esperados

- Identificación de las variables con mayor influencia en la decisión del cliente.  
- Un modelo predictivo con métricas adecuadas de precisión y recall.  
- Recomendaciones para optimizar las futuras campañas de marketing.  

---

## 👤 Autor

**Bryan Jumbo Torres**  
📍 Mallorca, España  
💻 Proyecto académico / profesional de análisis de datos  