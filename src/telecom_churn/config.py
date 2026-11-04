
"""
Configuración central del proyecto Telecom Churn.

Define rutas, variables objetivo y parámetros generales
utilizados por los módulos de Ciencia de Datos.
"""

from pathlib import Path

# --------------------------------------------------
# 1. Directorio raíz del proyecto
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

# --------------------------------------------------
# 2. Directorios de trabajo
# --------------------------------------------------

DATA_DIR = BASE_DIR / "data"

MODELS_DIR = BASE_DIR / "models"

REPORTS_DIR = BASE_DIR / "reports"

# Archivo original del dataset
DATA_FILE = DATA_DIR / "telecom_churn_10000.csv"

# Archivo donde se almacenará el modelo ganador
MODEL_PATH = MODELS_DIR / "best_model.pkl"

# Archivo donde se almacenarán las métricas
METRICS_PATH = REPORTS_DIR / "metrics.json"

# --------------------------------------------------
# 3. Variables del problema
# --------------------------------------------------

# Variable objetivo
TARGET_COLUMN = "Churn"

# Identificador del cliente
ID_COLUMN = "CustomerID"

# Variables que no deben utilizarse como predictoras
EXCLUDED_COLUMNS = [
    ID_COLUMN,
    "ChurnReason"
]

# --------------------------------------------------
# 4. Parámetros de experimentación
# --------------------------------------------------

RANDOM_STATE = 42

TEST_SIZE = 0.20

# Umbral inicial de clasificación
CLASSIFICATION_THRESHOLD = 0.35