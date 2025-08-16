"""
Page d'exploration des données pour l'application d'analyse marketing.
Permet d'explorer et de visualiser les données importées.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import sys

# Ajouter le répertoire parent au chemin pour les imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.visualization import (
    plot_distribution,
    plot_correlation_heatmap,
    plot_time_series,
    display_metrics
)
from config import APP_CONFIG

# Configuration de la page
st.set_page_config(
    page_title=f"{APP_CONFIG['app_name']} - Exploration des données",
    page_icon="🔍",
    layout="wide"
)

# Titre de la page
st.title("🔍 Exploration des données")
st.markdown("---")

# Vérifier que les données sont chargées
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Veuillez d'abord importer des données dans l'onglet 'Importation des données'.")
    st.stop()

# Récupérer les données de la session
customers_df = st.session_state.get('customers_df')
orders_df = st.session_state.get('orders_df')
marketing_df = st.session_state.get('marketing_df')
products_df = st.session_state.get('products_df')

# Section de résumé des données
with st.expander("📊 Aperçu général des données", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Clients", f"{len(customers_df):,}" if customers_df is not None else "N/A")
    with col2:
        st.metric("🛒 Commandes", f"{len(orders_df):,}" if orders_df is not None else "N/A")
    with col3:
        st.metric("📢 Campagnes", f"{len(marketing_df):,}" if marketing_df is not None else "N/A")
    with col4:
        st.metric("👕 Produits", f"{len(products_df):,}" if products_df is not None else "N/A")
    
    # Afficher un résumé des données manquantes
    st.subheader("Données manquantes")
    
    if customers_df is not None:
        missing_data = {
            'Tableau': ['Clients', 'Commandes', 'Marketing', 'Produits'],
            'Lignes': [
                len(customers_df),
                len(orders_df) if orders_df is not None else 0,
                len(marketing_df) if marketing_df is not None else 0,
                len(products_df) if products_df is not None else 0
            ],
            'Valeurs manquantes': [
                customers_df.isnull().sum().sum(),
                orders_df.isnull().sum().sum() if orders_df is not None else 0,
                marketing_df.isnull().sum().sum() if marketing_df is not None else 0,
                products_df.isnull().sum().sum() if products_df is not None else 0
            ]
        }
        
        # Calculer le pourcentage de valeurs manquantes avec gestion de la division par zéro
        missing_data['% manquant'] = []
        for i in range(len(missing_data['Tableau'])):
            total_cells = missing_data['Lignes'][i] * len(customers_df.columns)
            if total_cells > 0:
                percentage = (missing_data['Valeurs manquantes'][i] / total_cells) * 100
                missing_data['% manquant'].append(f"{percentage:.1f}%")
            else:
                missing_data['% manquant'].append("0.0%")
        
        # Créer un DataFrame avec des types explicites pour éviter les problèmes de sérialisation
        missing_df = pd.DataFrame(missing_data)
        # Convertir les colonnes en types simples
        missing_df['Lignes'] = missing_df['Lignes'].astype(int)
        missing_df['Valeurs manquantes'] = missing_df['Valeurs manquantes'].astype(int)
        missing_df['% manquant'] = missing_df['% manquant'].str.rstrip('%').astype(float)
        
        # Afficher le DataFrame avec des formats personnalisés
        st.dataframe(
            missing_df,
            column_config={
                'Lignes': st.column_config.NumberColumn(
                    'Lignes',
                    format='%d',
                    help='Nombre total de lignes dans le jeu de données'
                ),
                'Valeurs manquantes': st.column_config.NumberColumn(
                    'Valeurs manquantes',
                    format='%d',
                    help='Nombre total de valeurs manquantes'
                ),
                '% manquant': st.column_config.NumberColumn(
                    '% manquant',
                    format='%.1f%%',
                    help='Pourcentage de valeurs manquantes par rapport au nombre total de cellules'
                )
            },
            use_container_width=True
        )

# Onglets pour explorer chaque jeu de données
tab1, tab2, tab3, tab4 = st.tabs([
    "👥 Clients", 
    "🛒 Commandes", 
    "📢 Marketing", 
    "👕 Produits"
])

with tab1:
    st.header("Données clients")
    
    if customers_df is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Sélection de la variable à analyser
            numeric_cols = customers_df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = customers_df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Afficher les statistiques descriptives
            st.subheader("Statistiques descriptives")
            st.dataframe(customers_df.describe(include='all').T, use_container_width=True)
            
            # Afficher les corrélations pour les variables numériques
            if len(numeric_cols) > 1:
                st.subheader("Matrice de corrélation")
                corr = customers_df[numeric_cols].corr()
                fig = px.imshow(
                    corr,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Visualisation des données
            st.subheader("Visualisation")
            
            # Sélection du type de graphique
            chart_type = st.selectbox(
                "Type de graphique",
                ["Distribution", "Histogramme", "Boîte à moustaches"],
                key="cust_chart_type"
            )
            
            # Variables pour l'axe X
            x_axis = st.selectbox(
                "Variable X",
                customers_df.columns,
                key="cust_x_axis"
            )
            
            # Variables pour la couleur (facultatif)
            color_by = st.selectbox(
                "Segmenter par",
                ["Aucun"] + cat_cols,
                key="cust_color_by"
            )
            
            # Générer le graphique sélectionné
            if chart_type == "Distribution":
                fig = px.histogram(
                    customers_df,
                    x=x_axis,
                    color=None if color_by == "Aucun" else color_by,
                    marginal="box",
                    title=f"Distribution de {x_axis}",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == "Histogramme":
                fig = px.histogram(
                    customers_df,
                    x=x_axis,
                    color=None if color_by == "Aucun" else color_by,
                    barmode="group",
                    title=f"Histogramme de {x_axis}",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == "Boîte à moustaches":
                if x_axis in numeric_cols and color_by in cat_cols:
                    fig = px.box(
                        customers_df,
                        x=color_by,
                        y=x_axis,
                        color=color_by,
                        title=f"Boîte à moustaches de {x_axis} par {color_by}",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("La boîte à moustaches nécessite une variable numérique et une variable catégorielle.")
                    
        # Afficher les données brutes
        with st.expander("Afficher les données brutes", expanded=False):
            st.dataframe(customers_df, use_container_width=True)
    else:
        st.warning("Aucune donnée client disponible.")

with tab2:
    st.header("Données de commandes")
    
    if orders_df is not None:
        # Convertir la date si nécessaire
        if 'order_date' in orders_df.columns and not pd.api.types.is_datetime64_any_dtype(orders_df['order_date']):
            orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Statistiques de base
            st.subheader("Statistiques des commandes")
            
            if 'amount' in orders_df.columns:
                total_sales = orders_df['amount'].sum()
                avg_order_value = orders_df['amount'].mean()
                orders_count = len(orders_df)
                
                metrics = {
                    "Chiffre d'affaires total": f"{total_sales:,.2f} €",
                    "Valeur moyenne des commandes": f"{avg_order_value:,.2f} €",
                    "Nombre total de commandes": f"{orders_count:,}",
                    "Clients uniques": f"{orders_df['customer_id'].nunique():,}" if 'customer_id' in orders_df.columns else "N/A"
                }
                
                for label, value in metrics.items():
                    st.metric(label, value)
            
            # Produits les plus vendus
            if 'product_id' in orders_df.columns and 'quantity' in orders_df.columns:
                st.subheader("Produits les plus vendus")
                top_products = orders_df.groupby('product_id')['quantity'].sum().nlargest(5)
                st.bar_chart(top_products)
        
        with col2:
            # Évolution temporelle des ventes
            st.subheader("Évolution des ventes")
            
            if 'order_date' in orders_df.columns and 'amount' in orders_df.columns:
                # Agrégation par jour
                daily_sales = orders_df.set_index('order_date')['amount'].resample('D').sum()
                
                # Sélection de la période
                time_period = st.selectbox(
                    "Période",
                    ["7 derniers jours", "30 derniers jours", "3 derniers mois", "Toutes les données"],
                    key="sales_time_period"
                )
                
                # Filtrer les données en fonction de la période sélectionnée
                end_date = daily_sales.index.max()
                if time_period == "7 derniers jours":
                    start_date = end_date - pd.Timedelta(days=7)
                elif time_period == "30 derniers jours":
                    start_date = end_date - pd.Timedelta(days=30)
                elif time_period == "3 derniers mois":
                    start_date = end_date - pd.Timedelta(days=90)
                else:
                    start_date = daily_sales.index.min()
                
                filtered_sales = daily_sales.loc[start_date:end_date]
                
                # Créer un DataFrame pour le graphique
                sales_df = pd.DataFrame({
                    'Date': filtered_sales.index,
                    "Chiffre d'affaires": filtered_sales.values
                })
                
                # Créer le graphique
                fig = px.line(
                    sales_df,
                    x='Date',
                    y="Chiffre d'affaires",
                    title="Évolution du chiffre d'affaires",
                    template="plotly_white"
                )
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Chiffre d'affaires (€)",
                    hovermode="x unified"
                )
                
                # Ajouter une ligne de tendance
                if len(filtered_sales) > 1:
                    z = np.polyfit(range(len(filtered_sales)), filtered_sales.values, 1)
                    p = np.poly1d(z)
                    fig.add_scatter(
                        x=filtered_sales.index,
                        y=p(range(len(filtered_sales))),
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name='Tendance'
                    )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Analyse des paniers moyens
        with st.expander("Analyse des paniers moyens", expanded=False):
            if 'amount' in orders_df.columns and 'customer_id' in orders_df.columns:
                # Calculer le panier moyen par client
                avg_basket = orders_df.groupby('customer_id')['amount'].agg(['count', 'sum']).reset_index()
                avg_basket.columns = ['customer_id', 'orders_count', 'total_spent']
                avg_basket['avg_basket'] = avg_basket['total_spent'] / avg_basket['orders_count']
                
                # Afficher les métriques
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Panier moyen global", f"{avg_basket['avg_basket'].mean():.2f} €")
                with col2:
                    st.metric("Client le plus fidèle", f"{avg_basket['orders_count'].max()} commandes")
                with col3:
                    st.metric("Panier maximum", f"{avg_basket['avg_basket'].max():.2f} €")
                
                # Distribution des paniers moyens
                fig = px.histogram(
                    avg_basket,
                    x='avg_basket',
                    nbins=30,
                    title="Distribution des paniers moyens par client",
                    labels={'avg_basket': 'Panier moyen (€)'},
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Données insuffisantes pour l'analyse des paniers moyens.")
        
        # Afficher les données brutes
        with st.expander("Afficher les données brutes", expanded=False):
            st.dataframe(orders_df, use_container_width=True)
    else:
        st.warning("Aucune donnée de commande disponible.")

# Sections pour les onglets Marketing et Produits (similaires à celles des clients et commandes)
# ... (le code pour ces sections serait similaire en structure mais adapté aux données marketing et produits)

# Navigation entre les pages
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("← Précédent", use_container_width=True):
        st.switch_page("pages/1_📥_Importation_des_données.py")

with col3:
    if st.button("Suivant →", type="primary", use_container_width=True):
        st.switch_page("pages/3_📊_Segmentation_client.py")

# Style CSS personnalisé
st.markdown("""
    <style>
    /* Style des onglets */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding: 10px 15px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
    }
    
    /* Style des expanders */
    .streamlit-expanderHeader {
        font-size: 1.1em;
        font-weight: 600;
    }
    
    /* Style des métriques */
    .stMetric {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    
    .stMetric [data-testid="stMetricLabel"] {
        font-size: 0.9em;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.5em;
        font-weight: 600;
    }
    
    /* Style des boutons de navigation */
    .stButton>button {
        border-radius: 20px;
        padding: 0.5rem 1rem;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)
