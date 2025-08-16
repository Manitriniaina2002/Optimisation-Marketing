import os
import sys
import subprocess

# Configuration des variables d'environnement pour Streamlit
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_SERVER_PORT'] = '8501'

# Chemin vers l'exécutable Python
python_executable = sys.executable

# Commande pour lancer Streamlit avec les paramètres
cmd = [
    python_executable, 
    "-m", 
    "streamlit", 
    "run", 
    "app.py",
    "--server.port=8501",
    "--server.headless=true",
    "--browser.gatherUsageStats=false"
]

# Exécuter la commande
subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
