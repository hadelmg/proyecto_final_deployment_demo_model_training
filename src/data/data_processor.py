import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.cluster import KMeans

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y transforma un DataFrame para análisis.
    
    Args:
        df (pd.DataFrame): DataFrame original.
        
    Returns:
        pd.DataFrame: DataFrame procesado.
    """
    # Normalizar nombres de columnas
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("'", "", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.lower()
    )

    # Eliminar columnas irrelevantes y normalizar datos categóricos
    df = df.drop(columns=['id'], errors='ignore')
    df['marital_status'] = df['marital_status'].replace({
        'Together': 'Partner',
        'Married': 'Partner',
        'Divorced': 'Single',
        'Widow': 'Single',
        'Alone': 'Single',
        'Absurd': 'Single',
        'YOLO': 'Single'
    })
    df["education"] = df["education"].replace({
        "Basic": "Undergraduate",
        "2n Cycle": "Undergraduate", 
        "Graduation": "Graduate", 
        "Master": "Postgraduate", 
        "PhD": "Postgraduate"
    })

    # Crear nuevas características
    df['age'] = 2024 - df['year_birth']
    df['dt_customer'] = pd.to_datetime(df['dt_customer'], format='%d-%m-%Y', errors='coerce')
    df['years_customer'] = 2024 - df['dt_customer'].dt.year
    df['total_expenses'] = df.filter(like='mnt').sum(axis=1)
    df['total_acc_cmp'] = df.filter(like='acceptedcmp').sum(axis=1) + df['response']
    df["children"] = df["kidhome"] + df["teenhome"]
    df["living_with"] = df["marital_status"].map({'Single': 1, 'Partner': 2}).fillna(0).astype(int)
    df["family_size"] = df["living_with"] + df["children"]

    # Renombrar columnas
    df = df.rename(columns={
        "mntwines": "Wines", "mntfruits": "Fruits", "mntmeatproducts": "Meat", 
        "mntfishproducts": "Fish", "mntsweetproducts": "Sweets", "mntgoldprods": "Gold"
    })

    # Eliminar duplicados, outliers y manejar valores faltantes
    df = df.drop_duplicates()
    df = df[(df['age'] < 100) & (df['income'] < 150000)]
    df['income'] = df['income'].fillna(df['income'].mean())

    # Codificar columnas categóricas
    education_order = ['Undergraduate', 'Graduate', 'Postgraduate']
    ore = OrdinalEncoder(categories=[education_order])
    df['education'] = ore.fit_transform(df[['education']])
    lenc = LabelEncoder()
    df['marital_status'] = lenc.fit_transform(df['marital_status'])

    return df

def apply_kmeans(df: pd.DataFrame, columns_to_scale: list, n_clusters: int = 4) -> pd.DataFrame:
    """
    Aplica el modelo K-Means al DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame procesado.
        columns_to_scale (list): Columnas a escalar y usar en el modelo.
        n_clusters (int): Número de clusters.
        
    Returns:
        pd.DataFrame: DataFrame con etiquetas de clusters.
    """
    scaler = StandardScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[columns_to_scale])

    return df
