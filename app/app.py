"""
Point d'entrée principal de l'application d'analyse marketing.
Ce fichier sert de point d'entrée unique pour l'application Streamlit.
"""

import streamlit as st
import sys
from pathlib import Path

# Ajouter le répertoire parent au chemin pour les imports
sys.path.append(str(Path(__file__).parent))

# Configuration de la page
st.set_page_config(
    page_title="Analyse Marketing - Tableau de Bord",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre de la page
st.title("🏠 Accueil - Tableau de Bord")
st.markdown("---")

# Section principale
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Bienvenue dans l'application d'analyse marketing")
    
    st.markdown("""
    Cette application complète vous guide à travers un processus structuré d'analyse marketing,
    depuis l'importation des données jusqu'à la génération de recommandations stratégiques.
    
    ### Comment utiliser cette application :
    1. **📥 Importation des données** : Chargez facilement vos fichiers de données (CSV, Excel)
    2. **🔍 Exploration** : Visualisez et explorez vos données avec des graphiques interactifs
    3. **📊 Segmentation** : Découpez votre clientèle en segments homogènes avec l'analyse RFM
    4. **📈 Analyse des performances** : Évaluez l'efficacité de vos campagnes marketing
    5. **🔮 Analyse prédictive** : Prévoyez les tendances futures et comportements des clients
    6. **🎯 Stratégie** : Générez des recommandations personnalisées basées sur les données
    
    Pour commencer, utilisez le menu de navigation à gauche ou cliquez sur le bouton ci-dessous.
    """)
    
    # Bouton pour commencer
    if st.button("🚀 Commencer l'analyse", type="primary", use_container_width=True):
        st.switch_page("pages/1_📥_Importation_des_données.py")
    
    # Section des fonctionnalités
    st.markdown("---")
    st.subheader("🚀 Fonctionnalités clés")
    
    # Cartes de fonctionnalités
    features = [
        {
            "title": "📊 Tableaux de bord interactifs",
            "description": "Visualisez vos données avec des graphiques et tableaux interactifs pour une meilleure compréhension."
        },
        {
            "title": "🔍 Analyse RFM avancée",
            "description": "Segmentez vos clients en fonction de la récence, fréquence et montant de leurs achats."
        },
        {
            "title": "📈 Analyse prédictive",
            "description": "Prévoyez les tendances futures et comportements des clients avec des modèles avancés."
        },
        {
            "title": "📋 Rapports personnalisés",
            "description": "Générez et exportez des rapports détaillés au format PDF ou Excel."
        }
    ]
    
    for feature in features:
        with st.expander(feature["title"], expanded=True):
            st.markdown(f"{feature['description']}")

with col2:
    # Barre de progression
    st.markdown("### Progression du projet")
    progress_value = st.session_state.get('progress', 0)
    progress = st.progress(progress_value)
    st.caption(f"Progression globale : {int(progress_value*100)}%")
    
    # Section d'aide rapide
    with st.expander("❓ Aide rapide", expanded=False):
        st.markdown("""
        - Utilisez le menu de navigation à gauche pour accéder aux différentes sections
        - Consultez les infobulles (ℹ️) pour plus d'informations sur chaque fonctionnalité
        - En cas de problème, essayez de recharger la page ou de supprimer le cache du navigateur
        """)

# Pied de page
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>Application d'analyse marketing &copy; 2025 | Version 1.0.0</p>
    <p>Développé avec ❤️ par l'équipe d'analyse de données</p>
</div>
""", unsafe_allow_html=True)
