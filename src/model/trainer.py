from sklearn.neural_network import MLPClassifier
#from keras.models import Sequential
#from keras.layers import Dense, Dropout
#from keras.optimizers import Adam

def entrenar_modelo_sklearn(X_train, y_train, parametros=None):
    """
    Función para entrenar un modelo MLP utilizando scikit-learn.

    Args:
        X_train (array-like): Características de entrenamiento.
        y_train (array-like): Etiquetas de entrenamiento.
        parametros (dict, opcional): Parámetros del modelo. Por defecto es None.

    Returns:
        MLPClassifier: Modelo entrenado.
    """
    if parametros is None:
        parametros = {'hidden_layer_sizes': (128, 64), 'max_iter': 500, 'alpha': 0.01, 'random_state': 42}
    modelo = MLPClassifier(**parametros)
    modelo.fit(X_train, y_train)
    return modelo

'''def entrenar_modelo_keras(X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """
    Función para entrenar un modelo MLP utilizando Keras.

    Args:
        X_train (array-like): Características de entrenamiento.
        y_train (array-like): Etiquetas de entrenamiento.
        X_test (array-like): Características de prueba.
        y_test (array-like): Etiquetas de prueba.
        epochs (int, opcional): Número de épocas. Por defecto es 50.
        batch_size (int, opcional): Tamaño del lote. Por defecto es 32.

    Returns:
        keras.Model: Modelo entrenado.
    """
    modelo = Sequential()
    modelo.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(64, activation='relu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(len(np.unique(y_train)), activation='softmax'))

    modelo.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    modelo.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
    return modelo
'''