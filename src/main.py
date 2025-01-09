import sys
import os
# Agregar la raíz del proyecto al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.data_loader import load_data
from src.data.data_processor import preprocess_data, apply_kmeans
from src.data.data_splitter import split_data
from src.model.trainer import entrenar_modelo_sklearn #,entrenar_modelo_keras
from src.model.evaluator import evaluar_modelo_sklearn #, evaluar_modelo_keras
from src.model.saver import guardar_modelo_sklearn #, guardar_modelo_keras

def main():
    # Verificar si el archivo existe
    file_path = "data/raw/marketing_campaign.csv"
    if not os.path.exists(file_path):
        print(f"Error: El archivo '{file_path}' no se encuentra.")
        return  # Salir de la función si el archivo no existe

    # Cargar los datos
    data = load_data(file_path)
    if data.empty:
        print("Error: El archivo está vacío o no se pudo cargar correctamente.")
        return

    print("Datos cargados correctamente. Procesando los datos...")

    # Preprocesar los datos
    processed_data = preprocess_data(data)
    print("Preprocesamiento completado.")

    # Definir columnas para escalado
    columns_to_scale = [
        'income', 'kidhome', 'teenhome', 'recency', 'Wines', 'Fruits', 'Meat', 
        'Fish', 'Sweets', 'Gold', 'numdealspurchases', 'numwebpurchases', 
        'numcatalogpurchases', 'numstorepurchases', 'numwebvisitsmonth', 
        'age', 'years_customer', 'total_expenses', 'total_acc_cmp'
    ]
    
    # Aplicar K-Means
    print("Aplicando K-Means...")
    clustered_data = apply_kmeans(processed_data, columns_to_scale, n_clusters=4)
    print("Clustering completado.")

    # Eliminar las columnas no necesarias
    columns_to_remove = [
        'year_birth', 'dt_customer', 'acceptedcmp1', 'acceptedcmp2', 'acceptedcmp3', 'acceptedcmp4', 
        'acceptedcmp5', 'response', 'complain', 'z_costcontact', 'z_revenue', 'children', 'living_with', 
        'family_size', 'marital_status', 'education'
    ]
    clustered_data = clustered_data.drop(columns=columns_to_remove, errors='ignore')

    # Exportar los resultados
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "processed_data_with_clusters.csv")
    clustered_data.to_csv(output_path, index=False)
    print(f"Datos procesados y clusters guardados en: {output_path}")

    # Dividir los datos en entrenamiento y prueba
    print("Dividiendo los datos en conjunto de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = split_data(clustered_data, target_column="cluster")
    print("División completada.")

    # Entrenar el modelo utilizando scikit-learn
    print("Entrenando modelo con scikit-learn...")
    modelo_sklearn = entrenar_modelo_sklearn(X_train, y_train)
    print("Entrenamiento con scikit-learn completado.")

    # Evaluar el modelo de scikit-learn
    print("Evaluando modelo de scikit-learn...")
    evaluar_modelo_sklearn(modelo_sklearn, X_test, y_test)

    # Guardar el modelo de scikit-learn
    print("Guardando el modelo de scikit-learn...")
    guardar_modelo_sklearn(modelo_sklearn, "models/sklearn_model")

    # Entrenar el modelo utilizando Keras
    #print("Entrenando modelo con Keras...")
    #modelo_keras = entrenar_modelo_keras(X_train, y_train, X_test, y_test)
    #print("Entrenamiento con Keras completado.")

    # Evaluar el modelo de Keras
    #print("Evaluando modelo de Keras...")
    #evaluar_modelo_keras(modelo_keras, X_test, y_test)

    # Guardar el modelo de Keras
    #print("Guardando el modelo de Keras...")
    #guardar_modelo_keras(modelo_keras, "models/keras_model")

    print("Proceso completado con éxito.")

if __name__ == "__main__":
    main()
