import pickle
import numpy as np
import json
import os

# CARGAR EL MODELO =====
with open("models/modelo_entrenado.pkl", "rb") as archivo:
    datos = pickle.load(archivo)

modelo = datos["modelo"]
hiperparametros = datos["hiperparametros"]
columnas = datos.get("columnas", [
    "age", "duration", "campaign", "pdays", "previous", "emp.var.rate",
    "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
    "job_n", "marital_n", "education_n", "default_n", "housing_n",
    "loan_n", "contact_n", "month_n", "day_of_week_n", "poutcome_n"
])

print("Modelo cargado.")
print("Hiperparámetros usados:", hiperparametros)

# CARGAR DICCIONARIOS =====
ruta_data = os.path.join(os.getcwd(), "data", "processed")

dict_files = [
    "job_n.json", "marital_n.json", "education_n.json", "default_n.json",
    "housing_n.json", "loan_n.json", "contact_n.json", "month_n.json",
    "day_of_week_n.json", "poutcome_n.json", "y_n.json"
]

diccionarios = {}
for file in dict_files:
    with open(os.path.join(ruta_data, file), "r") as f:
        diccionarios[file.replace(".json","")] = json.load(f)
        
y_inverso = {v: k for k, v in diccionarios["y_n"].items()}


# EJEMPLOS DE ENTRADA LEGIBLE =====
ejemplos_legibles = [
    {
        "age": 34,
        "duration": 200,
        "campaign": 2,
        "pdays": 999,
        "previous": 0,
        "emp.var.rate": 1.1,
        "cons.price.idx": 93.994,
        "cons.conf.idx": -36.4,
        "euribor3m": 4.857,
        "nr.employed": 5195,
        "job": "technician",
        "marital": "single",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "poutcome": "success"
    },
    {
        "age": 41,
        "duration": 200,   
        "job": "blue-collar",
        "marital": "divorced",
        "education": "basic.4y",
        "default": "yes",
        "housing": "yes",
        "loan": "no",
        "contact": "telephone",
        "month": "may",
        "day_of_week": "mon",
        "poutcome": "success", 
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "emp.var.rate": 1.1,
        "cons.price.idx": 93.994,
        "cons.conf.idx": -36.4,
        "euribor3m": 4.857,
        "nr.employed": 5191.0
    }
]

# CONVERTIR A FORMATO NUMÉRICO =====
ejemplos = []
for e in ejemplos_legibles:
    ejemplo_num = [
        e["age"],
        e["duration"],
        e["campaign"],
        e["pdays"],
        e["previous"],
        e["emp.var.rate"],
        e["cons.price.idx"],
        e["cons.conf.idx"],
        e["euribor3m"],
        e["nr.employed"],
        diccionarios["job_n"][e["job"]],
        diccionarios["marital_n"][e["marital"]],
        diccionarios["education_n"][e["education"]],
        diccionarios["default_n"][e["default"]],
        diccionarios["housing_n"][e["housing"]],
        diccionarios["loan_n"][e["loan"]],
        diccionarios["contact_n"][e["contact"]],
        diccionarios["month_n"][e["month"]],
        diccionarios["day_of_week_n"][e["day_of_week"]],
        diccionarios["poutcome_n"][e["poutcome"]],
    ]
    ejemplos.append(ejemplo_num)

ejemplos = np.array(ejemplos)

print("\nEjemplos convertidos a numérico:")
print(ejemplos)

# PREDICCIÓN =====
predicciones_num = modelo.predict(ejemplos)

predicciones_legibles = [y_inverso[p] for p in predicciones_num]

print("\nPredicciones generadas:")
for i, pred in enumerate(predicciones_legibles):
    print(f"Ejemplo {i+1} - Cliente probable a contratar un depósito a largo plazo: {pred}")
