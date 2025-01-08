# Proyecto de Machine Learning - Pipeline de Entrenamiento e Inferencia

## Descripción

Este proyecto implementa un pipeline completo para entrenar e inferir con un modelo de Machine Learning. El enfoque principal es estructurar el código de manera modular y reproducible, con soporte para entrenamiento local y capacidad de realizar inferencias rápidas utilizando el modelo entrenado. 

Además, el proyecto está diseñado para ser fácilmente extensible, documentado y compatible con herramientas modernas como Docker y pruebas unitarias.

---

## Estructura del Proyecto

```plaintext
project_name/
│
├── data/                   # Directorio de datos
│   ├── raw/                # Datos originales sin procesar
│   └── processed/          # Datos preprocesados (opcional)
│
├── models/                 # Modelos entrenados
│   └── trained_model_2025-01-02.joblib
│
├── src/                    # Código fuente
│   ├── data/               # Módulos de manejo de datos
│   │   ├── data_loader.py      # Carga de datos
│   │   ├── data_processor.py   # Preprocesamiento de datos
│   │   └── data_splitter.py    # División en conjuntos de entrenamiento/validación
│   ├── model/              # Código relacionado con el modelo
│   │   └── main.py             # Entrenamiento e inferencia del modelo
│   └── utils/              # Funciones auxiliares (opcional)
│
├── .gitignore              # Archivos y carpetas ignorados por Git
├── LICENSE                 # Licencia del proyecto
├── README.md               # Documentación del proyecto
├── poetry.lock             # Bloqueo de dependencias (si usas Poetry)
└── pyproject.toml          # Configuración del entorno y dependencias
