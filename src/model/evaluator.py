from sklearn.metrics import classification_report 
import numpy as np

def evaluar_modelo_sklearn(modelo, X_test, y_test):
    """
    Función para evaluar un modelo MLP de scikit-learn.

    Args:
        modelo (MLPClassifier): Modelo entrenado.
        X_test (array-like): Características de prueba.
        y_test (array-like): Etiquetas de prueba.

    Returns:
        None
    """
    y_pred = modelo.predict(X_test)
    print("Reporte de clasificación (scikit-learn):")
    print(classification_report(y_test, y_pred))

'''def evaluar_modelo_keras(modelo, X_test, y_test):
    """
    Función para evaluar un modelo MLP de Keras.

    Args:
        modelo (keras.Model): Modelo entrenado.
        X_test (array-like): Características de prueba.
        y_test (array-like): Etiquetas de prueba.

    Returns:
        None
    """
    y_pred = modelo.predict(X_test).argmax(axis=-1)
    print("Reporte de clasificación (Keras):")
    print(classification_report(y_test, y_pred))'''

