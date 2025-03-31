# Importar librerías necesarias
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Crear variable de interacciones (suma de elementos)
interacciones = (
    filtered_data["# of Links"] +
    filtered_data["# of comments"].fillna(0) +  # Rellenar valores nulos
    filtered_data["# Images video"]
)

# Crear matriz de características (variables predictoras)
X = pd.DataFrame({
    "palabras": filtered_data["Word count"],
    "interacciones": interacciones
})

# Variable objetivo (valor a predecir)
y = filtered_data["# Shares"].values

# Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X, y)

# Realizar predicciones
predicciones = modelo.predict(X)

# Mostrar resultados clave
print(f"Coeficientes: {modelo.coef_}")
print(f"Intercepto: {modelo.intercept_:.2f}")
print(f"Error cuadrático medio: {mean_squared_error(y, predicciones):,.2f}")
print(f"R² (Bondad de ajuste): {r2_score(y, predicciones):.2f}")

# Ejemplo de predicción para un artículo
ejemplo = [[2000, 20]]  # 2000 palabras y 20 interacciones
print(f"\nPredicción para el ejemplo: {modelo.predict(ejemplo)[0]:,.0f} compartidos")