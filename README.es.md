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

## ⚙️ Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/nombre-del-repositorio.git
   ```
2. Acceder al directorio del proyecto:
   ```bash
   cd nombre-del-repositorio
   ```
3. Instalar las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```

*(Asegúrate de tener Python 3.8 o superior instalado.)*

---

## 🚀 Uso

1. Abre el archivo `explorar.ipynb` con **Jupyter Notebook** o **JupyterLab**.  
2. Ejecuta las celdas en orden para:
   - Cargar y limpiar los datos.  
   - Analizar las variables y sus relaciones.  
   - Probar modelos de clasificación.  
   - Evaluar los resultados.  

3. Revisa las conclusiones y gráficos generados al final del análisis.

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