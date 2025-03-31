# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Cargar el conjunto de datos desde la URL
url_dataset = "https://raw.githubusercontent.com/jbagnato/machine-learning/master/datasets/articles.csv"
dataset = pd.read_csv(url_dataset)

# Filtrar los datos: solo artículos con menos de 3000 palabras y menos de 80,000 compartidos
datos_filtrados = dataset[(dataset["Word count"] < 3000) & (dataset["# Shares"] < 80000)]

# Crear una nueva variable llamada "interacciones" (suma de enlaces, comentarios e imágenes)
interacciones = (
    datos_filtrados["# of Links"] +
    datos_filtrados["# of comments"].fillna(0) +
    datos_filtrados["# Images video"]
)

# Construir un DataFrame con las variables predictoras
variables_predictoras = pd.DataFrame()
variables_predictoras["palabras"] = datos_filtrados["Word count"]
variables_predictoras["interacciones"] = interacciones

# Variable objetivo (número de veces que se comparte el artículo)
objetivo = datos_filtrados["# Shares"].values

# Matriz de entrada para el modelo
matriz_entrada = np.array(variables_predictoras)

# Crear y entrenar el modelo de regresión lineal múltiple
modelo_regresion = LinearRegression()
modelo_regresion.fit(matriz_entrada, objetivo)

# Realizar predicciones sobre los datos de entrenamiento
predicciones = modelo_regresion.predict(matriz_entrada)

# Mostrar los resultados del modelo
print("Coeficientes del modelo:", modelo_regresion.coef_)
print("Intercepto del modelo:", modelo_regresion.intercept_)
print("Error cuadrático medio (MSE):", mean_squared_error(objetivo, predicciones))
print("Coeficiente de determinación (R²):", r2_score(objetivo, predicciones))

# Predecir el número de compartidos para un artículo hipotético:
# 2000 palabras, 10 enlaces, 4 comentarios, 6 imágenes
elementos_interaccion = 10 + 4 + 6
compartidos_predichos = modelo_regresion.predict([[2000, elementos_interaccion]])
print(f"Predicción para un artículo con 2000 palabras y {elementos_interaccion} interacciones:", int(compartidos_predichos[0]))