"""
Page d'importation des données pour l'application d'analyse marketing.
Permet d'importer des documents explicatifs et des fichiers de données.
"""

import streamlit as st
import pandas as pd
import os
from io import StringIO
from datetime import datetime
import base64
import sys
from pathlib import Path

# Ajouter le répertoire parent au chemin pour les imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.csv_importer import load_csv

# Configuration de la page
st.set_page_config(
    page_title="📊 Importation des données - Tableau de bord",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #4CAF50;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .stAlert {
        border-radius: 10px;
    }
    .stExpander {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Titre de la page avec icône
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 1rem;">
    <h1 style="margin: 0;">📥 Importation des données</h1>
</div>
""", unsafe_allow_html=True)

# Vérifier si les données sont déjà chargées
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.customers_df = None
    st.session_state.orders_df = None
    st.session_state.marketing_df = None
    st.session_state.products_df = None
    st.session_state.documents = []

# Ajouter le répertoire parent au chemin pour les imports
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import DataLoader, display_data_preview
from utils.visualization import display_metrics
from config import APP_CONFIG, UPLOAD_DIR

# Initialisation du chargeur de données
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader(UPLOAD_DIR)

# Section d'importation des documents explicatifs
with st.expander("📄 Documents explicatifs", expanded=True):
    st.markdown("""
    Téléchargez vos documents explicatifs (Word, PowerPoint, PDF) pour fournir un contexte supplémentaire 
    à votre analyse. Ces documents seront utilisés pour enrichir les rapports générés.
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_docs = st.file_uploader(
            "Sélectionnez un ou plusieurs documents",
            type=APP_CONFIG['allowed_file_types']['documents'],
            accept_multiple_files=True,
            key="doc_uploader"
        )
    
    with col2:
        st.markdown("#### Types de fichiers acceptés")
        st.markdown("""
        - 📝 Word (.docx)
        - 📊 PowerPoint (.pptx)
        - 📑 PDF (.pdf)
        - 📄 Texte (.txt)
        """)
    
    # Afficher la liste des documents téléversés
    if uploaded_docs:
        st.markdown("#### Documents téléversés")
        for doc in uploaded_docs:
            st.markdown(f"- {doc.name} ({doc.size / 1024:.1f} KB)")
    
    # Bouton pour traiter les documents
    if st.button("Traiter les documents", key="process_docs"):
        if uploaded_docs:
            with st.spinner("Traitement des documents en cours..."):
                # Ici, vous pourriez ajouter la logique pour extraire le texte des documents
                st.session_state.processed_docs = True
                st.success("Documents traités avec succès !")
        else:
            st.warning("Veuillez d'abord téléverser des documents.")

# Section d'importation des fichiers de données
with st.expander("📊 Données à analyser", expanded=True):
    st.markdown("""
    Téléchargez les fichiers de données nécessaires à l'analyse. Assurez-vous que les fichiers sont au format CSV 
    et respectent la structure attendue pour chaque type de données.
    """)
    
    # Créer des onglets pour chaque type de données
    tab1, tab2, tab3, tab4 = st.tabs([
        "👥 Clients", 
        "🛒 Commandes", 
        "📢 Marketing", 
        "👕 Produits"
    ])
    
    with tab1:
        st.markdown("### Données clients")
        st.markdown("""
        Fichier CSV contenant les informations sur les clients.
        **Colonnes attendues** :
        - `customer_id` : Identifiant unique du client
        - `age` : Âge du client
        - `gender` : Genre du client
        - `city` : Ville de résidence
        - `registration_date` : Date d'inscription
        """)
        
        customers_file = st.file_uploader(
            "Téléverser le fichier clients (customers.csv)",
            type=["csv"],
            key="customers_uploader"
        )
        
        if customers_file is not None:
            try:
                # Utiliser la fonction load_csv pour un chargement robuste
                expected_columns = ['customer_id', 'age', 'gender', 'city', 'registration_date']
                date_columns = ['registration_date']
                numeric_columns = ['age']
                
                # Réinitialiser le contenu du fichier au début
                customers_file.seek(0)
                
                # Charger avec la fonction utilitaire
                customers_df = load_csv(
                    customers_file,
                    expected_columns=expected_columns,
                    date_columns=date_columns,
                    numeric_columns=numeric_columns,
                    required_columns=['customer_id', 'age']
                )
                
                if customers_df is not None:
                    # Nettoyer les noms de colonnes
                    customers_df.columns = [col.strip().lower() for col in customers_df.columns]
                    
                    # S'assurer que customer_id est une chaîne de caractères
                    if 'customer_id' in customers_df.columns:
                        customers_df['customer_id'] = customers_df['customer_id'].astype(str).str.strip()
                    
                    # Vérifier et convertir les colonnes numériques
                    for col in numeric_columns:
                        if col in customers_df.columns:
                            # Remplacer les virgules par des points pour les nombres décimaux
                            if customers_df[col].dtype == 'object':
                                customers_df[col] = customers_df[col].astype(str).str.replace(',', '.')
                            # Convertir en numérique
                            customers_df[col] = pd.to_numeric(customers_df[col], errors='coerce')
                
                st.session_state.customers_df = customers_df
                
                # Aperçu des données
                with st.expander("Aperçu des données clients", expanded=True):
                    st.dataframe(customers_df.head())
                    
                    # Afficher des métriques de base
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Nombre de clients", len(customers_df))
                    with col2:
                        st.metric("Colonnes", ", ".join(customers_df.columns[:3]) + "...")
                    with col3:
                        missing = customers_df.isnull().sum().sum()
                        st.metric("Valeurs manquantes", missing)
                        
            except Exception as e:
                st.error(f"Erreur lors de la lecture du fichier : {e}")
                import traceback
                st.error(f"Détails de l'erreur : {traceback.format_exc()}")
    
    with tab2:
        st.markdown("### 📦 Données de commandes")
        
        # Section d'information sur les colonnes attendues
        with st.expander("ℹ️ Structure attendue du fichier", expanded=False):
            st.markdown("""
            Le fichier doit contenir les colonnes suivantes :
            - `order_id` : Identifiant unique de la commande
            - `customer_id` : Identifiant du client
            - `tshirt_id` : Identifiant du produit
            - `quantity` : Quantité commandée
            - `price` ou `amount` : Montant total de la commande
            - `order_date` : Date de la commande (format JJ/MM/AAAA)
            """)
        
        # Téléchargement du fichier
        orders_file = st.file_uploader(
            "Téléverser le fichier de commandes (orders.csv)",
            type=["csv"],
            key="orders_uploader"
        )
        
        if orders_file is not None:
            # Définir les colonnes attendues et les types
            expected_columns = ['order_id', 'customer_id', 'tshirt_id', 'quantity', 'price', 'amount', 'order_date']
            date_columns = ['order_date']
            numeric_columns = ['quantity', 'price', 'amount']
            required_columns = ['order_id', 'customer_id', 'order_date']
            
            # Charger le fichier avec notre utilitaire
            orders_df = load_csv(
                orders_file,
                expected_columns=expected_columns,
                date_columns=date_columns,
                numeric_columns=numeric_columns,
                required_columns=required_columns
            )
            
            if orders_df is not None:
                # Vérifier si nous avons besoin de récupérer les prix des produits
                if ('price' not in orders_df.columns or orders_df['price'].isna().all()) and 'tshirt_id' in orders_df.columns:
                    # Essayer de charger les produits pour récupérer les prix
                    products_file = Path("Data/tshirts.csv")
                    if products_file.exists():
                        try:
                            products_df = pd.read_csv(products_file, sep=';')
                            if 'tshirt_id' in products_df.columns and 'base_price' in products_df.columns:
                                # Créer un dictionnaire de correspondance tshirt_id -> base_price
                                price_map = dict(zip(products_df['tshirt_id'], products_df['base_price']))
                                
                                # Remplir les prix manquants avec ceux des produits
                                orders_df['price'] = orders_df['tshirt_id'].map(price_map)
                                st.success("✅ Les prix manquants ont été complétés à partir du catalogue produits.")
                                
                                # Afficher un avertissement si des prix n'ont pas pu être trouvés
                                if orders_df['price'].isna().any():
                                    missing_prices = orders_df[orders_df['price'].isna()]['tshirt_id'].unique()
                                    st.warning(f"⚠️ Prix non trouvés pour les produits suivants : {', '.join(missing_prices)}")
                        except Exception as e:
                            st.error(f"❌ Erreur lors de la lecture du fichier des produits : {e}")
                
                # Calculer le montant si nécessaire
                if 'amount' not in orders_df.columns and 'price' in orders_df.columns and 'quantity' in orders_df.columns:
                    orders_df['amount'] = orders_df['price'] * orders_df['quantity']
                    st.success("✅ La colonne 'amount' a été calculée comme produit de 'price' et 'quantity'.")
                
                # Vérifier que nous avons bien une colonne amount valide
                if 'amount' not in orders_df.columns or orders_df['amount'].isna().all():
                    st.error("❌ Impossible de déterminer la colonne de montant. Assurez-vous d'avoir une colonne 'amount' ou 'price' et 'quantity' valides.")
                    # Ne pas sauvegarder les données dans la session en cas d'erreur
                    st.session_state.orders_df = None
                else:
                    # Sauvegarder dans la session
                    st.session_state.orders_df = orders_df
                    
                    # Afficher les statistiques uniquement si les données sont valides
                    with st.expander("📊 Aperçu des données de commandes", expanded=True):
                        st.dataframe(orders_df.head())
                        
                        # Afficher des métriques de base
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📋 Nombre de commandes", len(orders_df))
                        with col2:
                            unique_customers = orders_df['customer_id'].nunique()
                            st.metric("👥 Clients uniques", unique_customers)
                        with col3:
                            date_range = orders_df['order_date'].dropna()
                            if not date_range.empty:
                                st.metric("📅 Période couverte", 
                                        f"{date_range.min().strftime('%d/%m/%Y')} au {date_range.max().strftime('%d/%m/%Y')}")
                        
                        # Afficher les statistiques de montant si disponible
                        if 'amount' in orders_df.columns:
                            st.subheader("📈 Statistiques financières")
                            amount_stats = orders_df['amount'].agg(['sum', 'mean', 'min', 'max'])
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("💵 Chiffre d'affaires total", f"{amount_stats['sum']:,.2f} €")
                            with col2:
                                st.metric("💰 Panier moyen", f"{amount_stats['mean']:,.2f} €")
                            with col3:
                                st.metric("🛒 Commande minimale", f"{amount_stats['min']:,.2f} €")
                            with col4:
                                st.metric("🏆 Commande maximale", f"{amount_stats['max']:,.2f} €")
    
    with tab3:
        st.markdown("### 📢 Données marketing")
        st.markdown("""
        Fichier CSV contenant les informations sur les campagnes marketing.
        **Colonnes attendues** :
        - `campaign_id` : Identifiant de la campagne
        - `date` : Date de la campagne
        - `channel` : Canal de diffusion
        - `spend` : Budget dépensé
        - `impressions` : Nombre d'impressions
        - `clicks` : Nombre de clics
        - `conversions` : Nombre de conversions
        - `revenue` : Revenu généré
        """)
        
        marketing_file = st.file_uploader(
            "Téléverser le fichier marketing (marketing.csv)",
            type=["csv"],
            key="marketing_uploader"
        )
        
        if marketing_file is not None:
            # Définir les colonnes attendues et les types
            expected_columns = ['campaign_id', 'date', 'channel', 'spend', 'impressions', 'clicks', 'conversions', 'revenue']
            date_columns = ['date']
            numeric_columns = ['spend', 'impressions', 'clicks', 'conversions', 'revenue']
            required_columns = ['campaign_id', 'date', 'channel']
            
            # Charger le fichier avec notre utilitaire
            marketing_df = load_csv(
                marketing_file,
                expected_columns=expected_columns,
                date_columns=date_columns,
                numeric_columns=numeric_columns,
                required_columns=required_columns
            )
            
            if marketing_df is not None:
                # Sauvegarder dans la session
                st.session_state.marketing_df = marketing_df
                
                # Afficher les statistiques
                with st.expander("📊 Aperçu des données marketing", expanded=True):
                    st.dataframe(marketing_df.head())
                    
                    # Afficher des métriques de base
                    st.subheader("📊 Statistiques marketing")
                    
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Nombre de campagnes", marketing_df['campaign_id'].nunique())
                    
                    if 'spend' in marketing_df.columns:
                        with cols[1]:
                            total_spend = marketing_df['spend'].sum()
                            st.metric("Budget total dépensé", f"{total_spend:,.2f} €")
                    
                    if 'revenue' in marketing_df.columns:
                        with cols[2]:
                            total_revenue = marketing_df['revenue'].sum()
                            st.metric("Revenu total généré", f"{total_revenue:,.2f} €")
                    
                    if 'channel' in marketing_df.columns:
                        st.markdown("**📈 Répartition par canal**")
                        channel_dist = marketing_df['channel'].value_counts()
                        st.bar_chart(channel_dist)
    
    with tab4:
        st.markdown("### 👕 Données produits (T-shirts)")
        st.markdown("""
        Fichier CSV contenant les informations sur les produits.
        **Colonnes attendues** :
        - `tshirt_id` : Identifiant unique du produit
        - `category` : Catégorie du produit (ex: Streetwear, Sport, etc.)
        - `style` : Style du T-shirt
        - `size` : Taille du produit
        - `color` : Couleur du produit
        - `base_price` : Prix de base du produit
        """)
        
        products_file = st.file_uploader(
            "Téléverser le fichier produits (tshirts.csv)",
            type=["csv"],
            key="products_uploader"
        )
        
        if products_file is not None:
            # Définir les colonnes attendues et les types
            expected_columns = ['tshirt_id', 'category', 'style', 'size', 'color', 'base_price']
            numeric_columns = ['base_price']
            required_columns = ['tshirt_id', 'category', 'base_price']
            
            # Charger le fichier avec notre utilitaire
            products_df = load_csv(
                products_file,
                expected_columns=expected_columns,
                numeric_columns=numeric_columns,
                required_columns=required_columns
            )
            
            if products_df is not None:
                # Renommer base_price en price pour la cohérence avec le reste de l'application
                if 'base_price' in products_df.columns and 'price' not in products_df.columns:
                    products_df['price'] = products_df['base_price']
                
                # Ajouter un coût par défaut si nécessaire
                if 'cost' not in products_df.columns and 'price' in products_df.columns:
                    products_df['cost'] = products_df['price'] * 0.6  # Coût à 60% du prix par défaut
                
                # Sauvegarder dans la session
                st.session_state.products_df = products_df
                
                # Aperçu des données
                with st.expander("📦 Aperçu des données produits", expanded=True):
                    st.dataframe(products_df.head())
                    
                    # Afficher des métriques de base
                    st.subheader("📊 Statistiques produits")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Nombre de produits", len(products_df))
                        if 'category' in products_df.columns:
                            categories = products_df['category'].nunique()
                            st.metric("Catégories différentes", categories)
                    
                    with col2:
                        if 'price' in products_df.columns:
                            avg_price = products_df['price'].mean()
                            st.metric("Prix moyen", f"{avg_price:,.2f} €")
                        
                        if all(col in products_df.columns for col in ['price', 'cost']):
                            products_df['margin'] = (products_df['price'] - products_df['cost']) / products_df['price']
                            avg_margin = products_df['margin'].mean()
                            st.metric("Marge moyenne", f"{avg_margin:.1%}")
                    
                    # Afficher la répartition par catégorie si disponible
                    if 'category' in products_df.columns:
                        st.markdown("**📊 Répartition par catégorie**")
                        category_dist = products_df['category'].value_counts()
                        st.bar_chart(category_dist)

# Bouton de validation
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Valider et passer à l'analyse", type="primary", use_container_width=True):
        # Vérifier que tous les fichiers requis sont chargés
        required_files = ['customers_df', 'orders_df', 'marketing_df', 'products_df']
        missing_files = [f for f in required_files if f not in st.session_state]
        
        if missing_files:
            st.error(f"Veuillez d'abord téléverser tous les fichiers requis. Manquant : {', '.join([f.replace('_df', '') for f in missing_files])}")
        else:
            # Enregistrer les données dans la session
            st.session_state.data_loaded = True
            # Rediriger vers la page d'exploration des données
            st.switch_page("pages/2_🔍_Exploration_des_données.py")

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
    
    /* Style des boutons */
    .stButton>button {
        border-radius: 20px;
        padding: 0.5rem 1rem;
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
    </style>
""", unsafe_allow_html=True)
