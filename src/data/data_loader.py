import pandas as pd
import os

def load_data(file_path: str, file_type: str = 'csv') -> pd.DataFrame:
    """
    Carga los datos desde un archivo. Soporta formatos CSV, Excel, JSON y TSV.

    Args:
        file_path (str): Ruta del archivo a cargar.
        file_type (str): Tipo de archivo (opcional). 'csv', 'excel' o 'json'. Por defecto es 'csv'.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados.

    Raises:
        FileNotFoundError: Si el archivo no se encuentra en la ruta especificada.
        ValueError: Si el tipo de archivo no es soportado o el archivo está mal formado.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe.")
    
    try:
        if file_type == 'csv':
            # Aquí se usa el delimitador '\t' si el archivo CSV tiene tabulaciones
            return pd.read_csv(file_path, delimiter='\t')  # Especificar delimitador como tabulador
        elif file_type == 'excel':
            return pd.read_excel(file_path)
        elif file_type == 'json':
            return pd.read_json(file_path)
        else:
            raise ValueError("Tipo de archivo no soportado. Use 'csv', 'excel' o 'json'.")
    
    except Exception as e:
        raise ValueError(f"Error al cargar el archivo {file_path}: {e}")
