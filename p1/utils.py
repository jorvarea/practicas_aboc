import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    """
    Carga y preprocesa el dataset Boston Housing

    Parámetros:
    file_path: ruta al archivo CSV

    Retorna:
    X_scaled: características normalizadas
    y: variable objetivo
    scaler: objeto StandardScaler para futura normalización
    """
    # Cargar datos
    data = pd.read_csv(file_path)

    # Eliminar la última columna (bias añadido por el autor en Kaggle)
    data = data.iloc[:, :-1]

    print(f"Forma del dataset después de eliminar bias: {data.shape}")
    print(f"Columnas: {list(data.columns)}")

    # Separar características y variable objetivo
    # Asumiendo que la última columna es MEDV (precio)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Normalizar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Características: {X_scaled.shape}")
    print(f"Variable objetivo: {y.shape}")

    return X_scaled, y, scaler