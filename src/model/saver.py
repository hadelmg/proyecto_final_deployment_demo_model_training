import joblib
from datetime import datetime

def guardar_modelo_sklearn(modelo, ruta_modelo):
    """
    Función para guardar un modelo de scikit-learn.

    Args:
        modelo (MLPClassifier): Modelo entrenado.
        ruta_modelo (str): Ruta donde guardar el modelo (sin extensión).

    Returns:
        None
    """
    fecha_actual = datetime.now().strftime("%Y-%m-%d")
    ruta_completa = f"{ruta_modelo}_{fecha_actual}.joblib"
    joblib.dump(modelo, ruta_completa)
    print(f"Modelo guardado en {ruta_completa}")

'''def guardar_modelo_keras(modelo, ruta_modelo):
    """
    Función para guardar un modelo de Keras.

    Args:
        modelo (keras.Model): Modelo entrenado.
        ruta_modelo (str): Ruta donde guardar el modelo (sin extensión).

    Returns:
        None
    """
    fecha_actual = datetime.now().strftime("%Y-%m-%d")
    ruta_completa = f"{ruta_modelo}_{fecha_actual}.h5"
    modelo.save(ruta_completa)
    print(f"Modelo guardado en {ruta_completa}")'''
