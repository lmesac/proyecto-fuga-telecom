from pathlib import Path

# Directorios base del proyecto
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "telecom_churn_10000.csv"

# Variable objetivo a predecir
TARGET_COLUMN = "Churn"

# Identificador (no entra al entrenamiento, pero se conserva)
ID_COLUMN = "CustomerID"