import os
from pathlib import Path

# Chemins des dossiers
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "output"
REPORTS_DIR = BASE_DIR / "reports"

# Création des dossiers si ils n'existent pas
for directory in [DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configuration de l'application
APP_CONFIG = {
    "app_name": "Analyse Marketing",
    "description": "Application d'analyse et d'optimisation marketing basée sur la segmentation client",
    "author": "Votre Nom",
    "version": "1.0.0",
    "allowed_file_types": {
        "documents": [".docx", ".pptx", ".pdf", ".txt"],
        "data": [".csv", ".xlsx", ".xls"]
    },
    "max_upload_size": 50  # Taille maximale en Mo
}

# Paramètres par défaut pour l'analyse
ANALYSIS_CONFIG = {
    "segmentation": {
        "n_clusters": 5,
        "features": ["recency", "frequency", "monetary_value"],
        "random_state": 42
    },
    "churn": {
        "test_size": 0.2,
        "random_state": 42,
        "threshold": 0.5
    }
}

# Style CSS personnalisé
CUSTOM_CSS = """
<style>
    /* Style général */
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    
    /* En-tête */
    .stApp > header {
        background-color: #f0f2f6;
    }
    
    /* Sidebar */
    .css-1d391kg {
        padding: 1rem;
    }
    
    /* Boutons */
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        font-weight: bold;
    }
    
    /* Cartes */
    .st-bb {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main {
            padding: 1rem;
        }
        .st-bb {
            padding: 1rem;
        }
    }
</style>
"""
