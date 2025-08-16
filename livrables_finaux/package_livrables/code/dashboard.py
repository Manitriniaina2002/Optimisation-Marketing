"""
Module 8: Tableau de Bord Interactif (M8)

Ce module déploie un tableau de bord interactif Streamlit pour visualiser les segments clients,
les prédictions de churn et les stratégies marketing.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# Configuration des chemins
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data_processed'
REPORTS_DIR = BASE_DIR / 'reports'

# Titre de l'application
st.set_page_config(
    page_title="Tableau de Bord Marketing",
    page_icon="📊",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .segment-card {
        border-left: 5px solid #3498db;
        padding: 15px;
        margin: 10px 0;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

def load_data():
    """Charge les données nécessaires pour le tableau de bord."""
    try:
        # Charger les segments clients
        segments_path = DATA_DIR / 'customer_segments.csv'
        segments_df = pd.read_csv(segments_path)
        
        # Charger les prédictions de churn si disponibles
        churn_path = DATA_DIR / 'churn_predictions.csv'
        if os.path.exists(churn_path):
            churn_df = pd.read_csv(churn_path)
            segments_df = segments_df.merge(churn_df, on='customer_id', how='left')
        
        return segments_df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return None

def display_segment_overview(df):
    """Affiche un aperçu des segments clients."""
    st.header("Aperçu des Segments Clients")
    
    # Calculer les métriques globales
    total_customers = len(df)
    segments = df['Cluster'].nunique()
    
    # Afficher les métriques clés
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nombre total de clients", total_customers)
    with col2:
        st.metric("Nombre de segments", segments)
    with col3:
        avg_clv = df['monetary'].mean()
        st.metric("Valeur moyenne du client (CLV)", f"{avg_clv:,.2f} €")
    
    # Graphique de répartition des segments
    st.subheader("Répartition des clients par segment")
    segment_counts = df['Cluster'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Nombre de clients']
    
    fig = px.pie(
        segment_counts, 
        values='Nombre de clients', 
        names='Segment',
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig, use_container_width=True)

def display_segment_details(df):
    """Affiche les détails par segment client."""
    st.header("Analyse Détail des Segments")
    
    # Sélection du segment à analyser
    selected_segment = st.selectbox(
        "Sélectionnez un segment à analyser",
        sorted(df['Cluster'].unique())
    )
    
    # Filtrer les données pour le segment sélectionné
    segment_data = df[df['Cluster'] == selected_segment]
    
    # Afficher les métriques du segment
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nombre de clients", len(segment_data))
    with col2:
        st.metric("CLV moyenne", f"{segment_data['monetary'].mean():,.2f} €")
    with col3:
        st.metric("Fréquence d'achat", f"{segment_data['frequency'].mean():.2f}/mois")
    with col4:
        st.metric("Récence moyenne", f"{segment_data['recency'].mean():.1f} jours")
    
    # Graphique de distribution de la CLV
    st.subheader("Distribution de la valeur client (CLV)")
    fig = px.histogram(
        segment_data, 
        x='monetary',
        nbins=20,
        labels={'monetary': 'Valeur du client (CLV en €)'},
        color_discrete_sequence=['#3498db']
    )
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)
    
    # Graphique de distribution par âge et genre
    if 'age_first' in segment_data.columns and 'gender_first' in segment_data.columns:
        st.subheader("Répartition par âge et genre")
        fig = px.histogram(
            segment_data, 
            x='age_first',
            color='gender_first',
            barmode='overlay',
            labels={'age_first': 'Âge', 'gender_first': 'Genre'},
            color_discrete_map={'M': '#3498db', 'F': '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)

def display_churn_analysis(df):
    """Affiche l'analyse du risque de churn."""
    if 'churn_probability' not in df.columns:
        st.warning("Les données de prédiction de churn ne sont pas disponibles.")
        return
    
    st.header("Analyse du Risque de Churn")
    
    # Seuil de risque de churn
    threshold = st.slider(
        "Définir le seuil de risque de churn",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Ajustez ce curseur pour définir le seuil à partir duquel un client est considéré comme à risque."
    )
    
    # Calculer les clients à risque
    df['at_risk'] = df['churn_probability'] >= threshold
    at_risk_count = df['at_risk'].sum()
    at_risk_pct = (at_risk_count / len(df)) * 100
    
    # Afficher les indicateurs clés
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Clients à risque", f"{at_risk_count:,}")
    with col2:
        st.metric("Pourcentage de clients à risque", f"{at_risk_pct:.1f}%")
    with col3:
        avg_risk = df['churn_probability'].mean() * 100
        st.metric("Risque moyen de churn", f"{avg_risk:.1f}%")
    
    # Graphique de distribution du risque par segment
    st.subheader("Risque de churn par segment")
    fig = px.box(
        df, 
        x='Cluster', 
        y='churn_probability',
        color='Cluster',
        labels={'churn_probability': 'Probabilité de churn', 'Cluster': 'Segment'}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def display_marketing_recommendations():
    """Affiche les recommandations marketing basées sur les segments."""
    st.header("Recommandations Marketing par Segment")
    
    # Charger les recommandations depuis le rapport généré
    recommendations = {
        1: {
            "objectif": "Fidélisation",
            "actions": [
                "Programme de fidélité avec récompenses personnalisées",
                "Offres exclusives pour les clients fidèles",
                "Contenu personnalisé basé sur l'historique d'achat"
            ],
            "canaux": ["Email", "Application mobile", "Réseaux sociaux"]
        },
        2: {
            "objectif": "Récupération",
            "actions": [
                "Campagnes de remarketing ciblées",
                "Offres spéciales pour les clients inactifs",
                "Enquêtes de satisfaction"
            ],
            "canaux": ["Email", "SMS", "Publicité ciblée"]
        },
        3: {
            "objectif": "Développement",
            "actions": [
                "Programmes d'upselling et cross-selling",
                "Contenu éducatif sur les produits",
                "Essais gratuits ou démos"
            ],
            "canaux": ["Email", "Site web", "Réseaux sociaux"]
        },
        4: {
            "objectif": "Rétention",
            "actions": [
                "Service client premium",
                "Accès anticipé aux nouvelles collections",
                "Programme de parrainage rémunéré"
            ],
            "canaux": ["Téléphone", "Email VIP", "Événements exclusifs"]
        }
    }
    
    # Afficher les recommandations pour chaque segment
    for segment, rec in recommendations.items():
        with st.expander(f"Segment {segment} - {rec['objectif']}"):
            st.subheader(f"Objectif principal: {rec['objectif']}")
            
            st.markdown("#### Actions recommandées:")
            for action in rec['actions']:
                st.markdown(f"- {action}")
            
            st.markdown("#### Canaux de communication:")
            st.markdown(", ".join(rec['canaux']))
            
            st.markdown("#### Indicateurs de performance clés (KPI):")
            st.markdown("""
                - Taux de rétention des clients
                - Fréquence d'achat moyenne
                - Valeur à vie du client (CLV)
                - Taux de conversion des campagnes
            """)

def main():
    """Fonction principale du tableau de bord."""
    # Titre du tableau de bord
    st.markdown("<h1 class='main-title'>Tableau de Bord Marketing Interactif</h1>", unsafe_allow_html=True)
    
    # Charger les données
    with st.spinner('Chargement des données...'):
        df = load_data()
    
    if df is None:
        st.error("Impossible de charger les données. Veuillez vérifier les fichiers de données.")
        return
    
    # Créer les onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "Vue d'ensemble", 
        "Analyse par segment", 
        "Risque de churn",
        "Recommandations"
    ])
    
    with tab1:
        display_segment_overview(df)
    
    with tab2:
        display_segment_details(df)
    
    with tab3:
        display_churn_analysis(df)
    
    with tab4:
        display_marketing_recommendations()
    
    # Pied de page
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #7f8c8d; font-size: 0.9rem;'>
            Tableau de bord généré le 16/08/2024 | Projet d'Analyse Marketing
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
