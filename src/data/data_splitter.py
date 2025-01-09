from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(data: pd.DataFrame, target_column: str, test_size=0.25, random_state=42, stratify: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide los datos en conjuntos de entrenamiento y prueba.

    Args:
        data (pd.DataFrame): DataFrame que contiene los datos a dividir.
        target_column (str): Nombre de la columna objetivo.
        test_size (float, opcional): Proporción de los datos que se destinan al conjunto de prueba. Por defecto es 0.25.
        random_state (int, opcional): Semilla para la aleatoriedad. Por defecto es 42.
        stratify (bool, opcional): Si es True, realiza una división estratificada según la variable objetivo. Por defecto es True.

    Returns:
        tuple: Una tupla que contiene los conjuntos de entrenamiento y prueba:
            - X_train (pd.DataFrame): Conjunto de entrenamiento de las variables independientes.
            - X_test (pd.DataFrame): Conjunto de prueba de las variables independientes.
            - y_train (pd.Series): Conjunto de entrenamiento de la variable objetivo.
            - y_test (pd.Series): Conjunto de prueba de la variable objetivo.
    """
    X = data.drop(columns=target_column)
    y = data[target_column]
    
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test
