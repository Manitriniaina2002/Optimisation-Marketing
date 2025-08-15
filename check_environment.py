import os
import sys
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
from sklearn import __version__ as sk_version

def main():
    print("=== Vérification de l'environnement Python ===")
    print(f"Python version: {sys.version}")
    print(f"Répertoire de travail: {os.getcwd()}")
    print("\n=== Fichiers dans le répertoire ===")
    for f in os.listdir('.'):
        print(f"- {f} (Répertoire)" if os.path.isdir(f) else f"- {f}")
    
    print("\n=== Versions des packages ===")
    print(f"pandas: {pd.__version__}")
    print(f"numpy: {np.__version__}")
    print(f"matplotlib: {matplotlib.__version__}")
    print(f"seaborn: {sns.__version__}")
    print(f"scikit-learn: {sk_version}")
    
    # Vérifier les fichiers de données
    print("\n=== Vérification des fichiers de données ===")
    data_files = ['customers_data.csv', 'products_data.csv', 'sales_data.csv']
    for file in data_files:
        exists = os.path.isfile(file)
        print(f"{file}: {'Trouvé' if exists else 'Non trouvé'}")
        if exists:
            try:
                df = pd.read_csv(file, nrows=1)
                print(f"  Lignes: {sum(1 for _ in open(file, 'r')) - 1} (estimation)")
                print(f"  Colonnes: {', '.join(df.columns)}")
            except Exception as e:
                print(f"  Erreur lors de la lecture: {str(e)}")

if __name__ == "__main__":
    main()
