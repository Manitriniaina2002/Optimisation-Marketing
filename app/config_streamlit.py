import os
import streamlit as st
from streamlit import config as _config

# Désactiver la demande d'email
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

# Démarrer l'application Streamlit
if __name__ == '__main__':
    st.set_page_config(
        page_title="Analyse Marketing - Tableau de Bord",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("📊 Analyse Marketing - Tableau de Bord")
    st.write("L'application est en cours de chargement...")
