import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

# Cargar el conjunto de datos
file_path = './data/BD_SINDES_clean.csv'  # Ajusta la ruta según sea necesario
data = pd.read_csv(file_path)

# Definir las características (X) y las variables objetivo (y)
X = data[['YEAR', 'MO', 'DY', 'HR']]
y = data[['WS10M', 'WD10M', 'WS50M', 'WD50M']]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Construir el modelo de red neuronal
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(y_train_scaled.shape[1])  # Capa de salida con tantas neuronas como variables objetivo
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar el modelo
history = model.fit(X_train_scaled, y_train_scaled, validation_split=0.2, epochs=200, batch_size=32, verbose=1)

# Evaluar el modelo
test_loss, test_mae = model.evaluate(X_test_scaled, y_test_scaled, verbose=1)

# Visualizar la pérdida de entrenamiento y validación
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.title('Pérdida de entrenamiento y validación')
plt.legend()
plt.savefig('./static/plots/perdida_entrenamiento_validacion.png')  # Guardar el gráfico
plt.close()

# Predecir los valores en el conjunto de prueba
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test_scaled)

n_muestras = 20  # Número de muestras a graficar
indices = np.sort(np.random.choice(len(y_true), n_muestras, replace=False))  # Ordenar índices para continuidad

# Crear el gráfico con estilo similar al proporcionado
plt.figure(figsize=(10, 6))
plt.plot(indices, y_true[indices, 0], label='Real', alpha=0.8, linewidth=2)
plt.plot(indices, y_pred[indices, 0], label='Predicción', alpha=0.8, linewidth=2)

# Configuración del gráfico
plt.xlabel('Índice de muestra')
plt.ylabel('Velocidad del viento a 10m (WS10M)')
plt.title('Valores reales vs predicción de WS10M')
plt.legend()

# Guardar el gráfico en un archivo
plt.savefig('./static/plots/valores_reales_vs_predichos.png')  # Guardar el gráfico
plt.close()

# Visualizar los residuos
residuals = y_true - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(range(len(residuals)), residuals[:, 0], alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Índice de muestra')
plt.ylabel('Residuos (WS10M)')
plt.title('Gráfico de residuos para WS10M')
plt.savefig('./static/plots/grafico_residuos.png')  # Guardar el gráfico
plt.close()
