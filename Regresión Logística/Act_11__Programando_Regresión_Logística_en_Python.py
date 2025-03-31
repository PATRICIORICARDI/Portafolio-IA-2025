# Importar bibliotecas necesarias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

# Cargar datos desde GitHub
url_datos = "https://raw.githubusercontent.com/jbagnato/machine-learning/master/datasets/usuarios_win_mac_lin.csv"
dataset = pd.read_csv(url_datos)

# Exploración inicial de los datos
print("Primeras filas del dataset:")
print(dataset.head())

print("\nEstadísticas básicas:")
print(dataset.describe())

print("\nDistribución de sistemas operativos:")
print(dataset["clase"].value_counts())

# Visualizar relaciones entre variables
sns.pairplot(dataset, hue="clase")
plt.suptitle("Relación entre variables y sistemas operativos")
plt.show()

# Preparar variables predictoras (X) y objetivo (y)
X = dataset.drop(columns="clase")  # Características de los usuarios
y = dataset["clase"]                # Sistema operativo a predecir

# Dividir datos en entrenamiento (80%) y prueba (20%)
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear y entrenar modelo de regresión logística multinomial
modelo_regresion = LogisticRegression(
    multi_class="multinomial",  # Para clasificación multiclase
    solver="lbfgs",             # Algoritmo de optimización
    max_iter=1000               # Máximo de iteraciones para convergencia
)
modelo_regresion.fit(X_entrenamiento, y_entrenamiento)

# Evaluar modelo con datos de prueba
predicciones = modelo_regresion.predict(X_prueba)

# Mostrar resultados de evaluación
print("\nMatriz de Confusión:")
print(confusion_matrix(y_prueba, predicciones))

print("\nReporte de Clasificación:")
print(classification_report(y_prueba, predicciones))

# Validación cruzada (5 particiones)
precision_cruzada = cross_val_score(modelo_regresion, X, y, cv=5)
print(f"\nPrecisión promedio (validación cruzada): {precision_cruzada.mean():.2f}")

# Ejemplo de predicción para un nuevo usuario
nuevo_usuario = np.array([[3.5, 7, 2, 35.0]])  # Datos: [duración, páginas, acciones, valor]
sistema_predicho = modelo_regresion.predict(nuevo_usuario)[0]

print(f"\nPredicción para el usuario: {sistema_predicho}")