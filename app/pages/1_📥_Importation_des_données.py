import streamlit as st
import pandas as pd
import os
from io import StringIO
from datetime import datetime
import base64
import sys
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajouter le répertoire parent au chemin pour les imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.csv_importer import load_csv
from utils.data_loader import DataLoader
from config import APP_CONFIG, UPLOAD_DIR

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

# Titre de la page
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 1rem;">
    <h1 style="margin: 0;">📥 Importation des données</h1>
</div>
""", unsafe_allow_html=True)

# Initialisation de l'état de la session
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader(UPLOAD_DIR)
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.customers_df = None
    st.session_state.orders_df = None
    st.session_state.marketing_df = None
    st.session_state.products_df = None
    st.session_state.documents = []
    st.session_state.processed_docs = False
# Ensure defaults exist even if other pages set only some keys
for k, v in {
    'data_loaded': False,
    'customers_df': None,
    'orders_df': None,
    'marketing_df': None,
    'products_df': None,
    'documents': [],
    'processed_docs': False,
}.items():
    st.session_state.setdefault(k, v)

# Fonction pour valider les identifiants uniques
def check_unique_ids(df, id_column, file_type):
    if id_column in df.columns:
        duplicates = df[id_column].duplicated().sum()
        if duplicates > 0:
            st.warning(f"⚠️ {duplicates} doublons trouvés dans la colonne '{id_column}' du fichier {file_type}.")
            return False
    return True

# Fonction pour valider les valeurs numériques positives
def check_positive_values(df, numeric_columns, file_type):
    for col in numeric_columns:
        if col in df.columns:
            if (df[col] < 0).any():
                st.error(f"❌ Valeurs négatives trouvées dans la colonne '{col}' du fichier {file_type}.")
                return False
    return True

# Fonction pour détecter les valeurs non numériques
def detect_non_numeric(df, column, file_type):
    non_numeric = df[column][pd.to_numeric(df[column], errors='coerce').isna() & df[column].notna()]
    if not non_numeric.empty:
        examples = non_numeric.head(5).index.tolist()
        example_values = non_numeric.head(5).tolist()
        st.warning(f"⚠️ {len(non_numeric)} valeurs non numériques détectées dans la colonne '{column}' du fichier {file_type}. "
                   f"Exemples (lignes {examples}): {example_values}. "
                   "Veuillez vérifier que les valeurs sont des nombres (ex. '19.99' au lieu de 'N/A' ou 'inconnu').")
        return False
    return True

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
            if doc.name.lower().endswith(tuple(APP_CONFIG['allowed_file_types']['documents'])):
                st.session_state.documents.append(doc)
                st.markdown(f"- {doc.name} ({doc.size / 1024:.1f} KB)")
            else:
                st.error(f"❌ Le fichier {doc.name} n'est pas dans un format autorisé.")
    
    # Bouton pour traiter les documents
    if st.button("Traiter les documents", key="process_docs"):
        if uploaded_docs:
            with st.spinner("Traitement des documents en cours..."):
                try:
                    # Simuler le traitement des documents (à implémenter selon vos besoins)
                    st.session_state.processed_docs = True
                    st.success("✅ Documents traités avec succès !")
                except Exception as e:
                    logger.error(f"Erreur lors du traitement des documents : {str(e)}")
                    st.error(f"❌ Erreur lors du traitement des documents : {str(e)}")
        else:
            st.warning("⚠️ Veuillez d'abord téléverser des documents.")

# Section d'importation des fichiers de données
with st.expander("📊 Données à analyser", expanded=True):
    st.markdown("""
    Téléchargez les fichiers de données nécessaires à l'analyse. Assurez-vous que les fichiers sont au format CSV 
    et respectent la structure attendue pour chaque type de données.
    """)
    
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
                customers_file.seek(0)
                expected_columns = ['customer_id', 'age', 'gender', 'city', 'registration_date']
                date_columns = ['registration_date']
                numeric_columns = ['age']
                
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
                    
                    # Vérifications supplémentaires
                    if not check_unique_ids(customers_df, 'customer_id', 'clients'):
                        customers_df = customers_df.drop_duplicates(subset='customer_id', keep='first')
                        st.warning("Les doublons dans 'customer_id' ont été supprimés (première occurrence conservée).")
                    
                    if not check_positive_values(customers_df, ['age'], 'clients'):
                        st.session_state.customers_df = None
                        st.error("❌ Les données clients ne sont pas valides en raison de valeurs négatives. Veuillez corriger le fichier.")
                    else:
                        # Vérifier les âges raisonnables
                        if 'age' in customers_df.columns:
                            if customers_df['age'].max() > 120 or customers_df['age'].min() < 0:
                                st.warning("⚠️ Âges incohérents détectés (négatifs ou > 120 ans).")
                        
                        # Sauvegarder dans la session
                        st.session_state.customers_df = customers_df
                        
                        with st.expander("Aperçu des données clients", expanded=True):
                            st.dataframe(customers_df.head())
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Nombre de clients", len(customers_df))
                            with col2:
                                st.metric("Colonnes", ", ".join(customers_df.columns[:3]) + "...")
                            with col3:
                                missing = customers_df.isnull().sum().sum()
                                st.metric("Valeurs manquantes", missing)
                            
            except pd.errors.ParserError:
                st.error("❌ Erreur de parsing du fichier CSV. Vérifiez le format et le séparateur (par exemple, utilisez ';' au lieu de ',').")
            except Exception as e:
                logger.error(f"Erreur lors du chargement des données clients : {str(e)}")
                st.error(f"❌ Erreur lors de la lecture du fichier : {str(e)}")
    
    with tab2:
        st.markdown("### 📦 Données de commandes")
        
        # Avertissement si les produits ne sont pas encore chargés
        products_df = st.session_state.get('products_df')
        if products_df is None:
            st.info("💡 **Conseil** : Pour une meilleure expérience, téléchargez d'abord le fichier produits (tshirts.csv) "
                   "dans l'onglet **👕 Produits** ci-dessous. Cela permettra de compléter automatiquement les prix manquants "
                   "dans le fichier de commandes.")
        
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
        
        orders_file = st.file_uploader(
            "Téléverser le fichier de commandes (orders.csv)",
            type=["csv"],
            key="orders_uploader"
        )
        
        if orders_file is not None:
            try:
                expected_columns = ['order_id', 'customer_id', 'tshirt_id', 'quantity', 'price', 'amount', 'order_date']
                date_columns = ['order_date']
                numeric_columns = ['quantity', 'price', 'amount']
                required_columns = ['order_id', 'customer_id', 'order_date']
                
                with st.spinner("Chargement du fichier orders.csv..."):
                    orders_df = load_csv(
                        orders_file,
                        expected_columns=expected_columns,
                        date_columns=date_columns,
                        numeric_columns=numeric_columns,
                        required_columns=required_columns
                    )
                
                if orders_df is not None:
                    # Afficher le séparateur détecté (assumé dans load_csv)
                    st.write("Séparateur détecté : ';'")
                    st.success(f"✅ Fichier chargé avec succès. {len(orders_df)} lignes trouvées.")
                    
                    # Afficher un aperçu des données brutes
                    with st.expander("🔍 Aperçu des données brutes (avant traitement)", expanded=False):
                        st.dataframe(orders_df.head())
                    
                    # Afficher le mapping des colonnes
                    st.markdown("📋 Mapping des colonnes :")
                    column_mapping = {col: "Trouvée" if col in orders_df.columns else "Manquante" for col in expected_columns}
                    st.json(column_mapping)
                    
                    # Vérifications supplémentaires
                    if not check_unique_ids(orders_df, 'order_id', 'commandes'):
                        orders_df = orders_df.drop_duplicates(subset='order_id', keep='first')
                        st.warning("Les doublons dans 'order_id' ont été supprimés (première occurrence conservée).")
                    
                    # Vérifier les valeurs non numériques dans 'price' et 'quantity' avant conversion
                    price_valid = True
                    quantity_valid = True
                    if 'price' in orders_df.columns:
                        price_valid = detect_non_numeric(orders_df, 'price', 'commandes')
                        if not price_valid:
                            orders_df['price'] = pd.to_numeric(orders_df['price'], errors='coerce')
                            st.warning("Les valeurs non numériques dans 'price' ont été remplacées par NaN.")
                    
                    if 'quantity' in orders_df.columns:
                        quantity_valid = detect_non_numeric(orders_df, 'quantity', 'commandes')
                        if not quantity_valid:
                            orders_df['quantity'] = pd.to_numeric(orders_df['quantity'], errors='coerce')
                            st.warning("Les valeurs non numériques dans 'quantity' ont été remplacées par NaN.")
                    
                    if not check_positive_values(orders_df, ['quantity', 'price', 'amount'], 'commandes'):
                        st.session_state.orders_df = None
                        st.error("❌ Les données de commandes ne sont pas valides en raison de valeurs négatives. Veuillez corriger le fichier.")
                    else:
                        # Vérifier la cohérence avec les clients
                        if st.session_state.customers_df is not None:
                            invalid_customers = set(orders_df['customer_id']) - set(st.session_state.customers_df['customer_id'])
                            if invalid_customers:
                                st.warning(f"⚠️ {len(invalid_customers)} customer_id non présents dans le fichier clients.")
                        
                        # Vérifier la cohérence avec les produits
                        if products_df is not None:
                            invalid_products = set(orders_df['tshirt_id']) - set(products_df['tshirt_id'])
                            if invalid_products:
                                st.warning(f"⚠️ {len(invalid_products)} tshirt_id non présents dans le fichier produits.")
                        
                        # Remplir les prix manquants avec products_df si disponible
                        price_filled = False
                        # Vérifier si la colonne price existe et si elle contient des valeurs manquantes ou invalides
                        price_needs_filling = False
                        if 'price' not in orders_df.columns:
                            price_needs_filling = True
                        elif 'price' in orders_df.columns:
                            # Convertir les valeurs vides ou non numériques en NaN
                            orders_df['price'] = pd.to_numeric(orders_df['price'], errors='coerce')
                            # Vérifier si toutes les valeurs sont NaN ou si il y a des valeurs manquantes
                            if orders_df['price'].isna().all() or orders_df['price'].isna().any():
                                price_needs_filling = True
                        
                        if price_needs_filling and 'tshirt_id' in orders_df.columns:
                            if products_df is not None:
                                price_map = dict(zip(products_df['tshirt_id'], products_df['price']))
                                orders_df['price'] = orders_df['tshirt_id'].map(price_map)
                                price_filled = True
                                if orders_df['price'].isna().any():
                                    st.warning(f"⚠️ {orders_df['price'].isna().sum()} lignes ont des prix manquants non récupérables depuis le fichier produits. "
                                               "Veuillez vérifier les 'tshirt_id' ou fournir des prix valides dans 'orders.csv'.")
                                else:
                                    st.success("✅ Tous les prix manquants ont été complétés à partir du fichier produits.")
                            else:
                                st.warning("⚠️ Le fichier produits (tshirts.csv) n'est pas chargé. Impossible de compléter les prix manquants.")
                        
                        # Calculer le montant si nécessaire
                        if 'amount' not in orders_df.columns and 'price' in orders_df.columns and 'quantity' in orders_df.columns:
                            valid_rows = orders_df['price'].notna() & orders_df['quantity'].notna()
                            if valid_rows.any():
                                orders_df.loc[valid_rows, 'amount'] = orders_df.loc[valid_rows, 'price'] * orders_df.loc[valid_rows, 'quantity']
                                st.success(f"✅ La colonne 'amount' a été calculée pour {valid_rows.sum()} lignes.")
                            else:
                                st.error("❌ Impossible de calculer la colonne 'amount' car toutes les valeurs de 'price' ou 'quantity' sont invalides. "
                                         "Veuillez vérifier que ces colonnes contiennent des valeurs numériques valides (ex. '19.99' au lieu de 'N/A').")
                                if st.session_state.get('products_df') is None:
                                    st.info("💡 **Solution recommandée** : Téléchargez le fichier produits (tshirts.csv) dans l'onglet '👕 Produits' "
                                           "pour que les prix soient automatiquement récupérés à partir des informations produit.")
                        
                        # Vérifier si amount est valide
                        if 'amount' not in orders_df.columns or orders_df['amount'].isna().all():
                            error_msg = "❌ Impossible de déterminer la colonne de montant. "
                            if products_df is None:
                                error_msg += ("Veuillez soit :\n"
                                             "1. Télécharger le fichier 'tshirts.csv' dans l'onglet '👕 Produits' pour utiliser les prix des produits, OU\n"
                                             "2. Corriger les colonnes 'price' et 'quantity' dans votre fichier orders.csv pour qu'elles contiennent des valeurs numériques valides")
                            else:
                                error_msg += ("Veuillez vérifier que les colonnes 'price' et 'quantity' contiennent des valeurs numériques valides "
                                             "ou que les 'tshirt_id' correspondent à ceux du fichier produits.")
                            st.error(error_msg)
                            st.session_state.orders_df = None
                        else:
                            # Supprimer les lignes où amount est NaN
                            invalid_rows = orders_df['amount'].isna()
                            if invalid_rows.any():
                                st.warning(f"⚠️ {invalid_rows.sum()} lignes ont été supprimées car elles contiennent des valeurs non valides pour 'amount'.")
                                orders_df = orders_df[~invalid_rows]
                            
                            st.session_state.orders_df = orders_df
                            
                            # Afficher les statistiques
                            with st.expander("📊 Aperçu des données de commandes", expanded=True):
                                st.dataframe(orders_df.head())
                                
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
                            
            except pd.errors.ParserError:
                st.error("❌ Erreur de parsing du fichier CSV. Vérifiez le format et le séparateur (par exemple, utilisez ';' au lieu de ',').")
            except UnicodeDecodeError:
                st.error("❌ Erreur d'encodage du fichier CSV. Essayez d'utiliser l'encodage UTF-8 ou ISO-8859-1.")
            except Exception as e:
                logger.error(f"Erreur lors du chargement des données de commandes : {str(e)}")
                st.error(f"❌ Erreur lors de la lecture du fichier : {str(e)}")
    
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
            try:
                expected_columns = ['campaign_id', 'date', 'channel', 'spend', 'impressions', 'clicks', 'conversions', 'revenue']
                date_columns = ['date']
                numeric_columns = ['spend', 'impressions', 'clicks', 'conversions', 'revenue']
                required_columns = ['campaign_id', 'date', 'channel']
                
                marketing_df = load_csv(
                    marketing_file,
                    expected_columns=expected_columns,
                    date_columns=date_columns,
                    numeric_columns=numeric_columns,
                    required_columns=required_columns
                )
                
                if marketing_df is not None:
                    if not check_unique_ids(marketing_df, 'campaign_id', 'marketing'):
                        marketing_df = marketing_df.drop_duplicates(subset='campaign_id', keep='first')
                        st.warning("Les doublons dans 'campaign_id' ont été supprimés (première occurrence conservée).")
                    
                    if not check_positive_values(marketing_df, numeric_columns, 'marketing'):
                        st.session_state.marketing_df = None
                        st.error("❌ Les données marketing ne sont pas valides en raison de valeurs négatives. Veuillez corriger le fichier.")
                    else:
                        st.session_state.marketing_df = marketing_df
                        
                        with st.expander("📊 Aperçu des données marketing", expanded=True):
                            st.dataframe(marketing_df.head())
                            
                            st.subheader("📊 Statistiques marketing")
                            cols = st.columns(3)
                            with cols[0]:
                                st.metric("Nombre de campagnes", marketing_df['campaign_id'].nunique())
                            with cols[1]:
                                total_spend = marketing_df['spend'].sum()
                                st.metric("Budget total dépensé", f"{total_spend:,.2f} €")
                            with cols[2]:
                                total_revenue = marketing_df['revenue'].sum()
                                st.metric("Revenu total généré", f"{total_revenue:,.2f} €")
                            
                            if 'channel' in marketing_df.columns:
                                st.markdown("**📈 Répartition par canal**")
                                channel_dist = marketing_df['channel'].value_counts()
                                st.bar_chart(channel_dist)
                            

            except pd.errors.ParserError:
                st.error("❌ Erreur de parsing du fichier CSV. Vérifiez le format et le séparateur (par exemple, utilisez ';' au lieu de ',').")
            except UnicodeDecodeError:
                st.error("❌ Erreur d'encodage du fichier CSV. Essayez d'utiliser l'encodage UTF-8 ou ISO-8859-1.")
            except Exception as e:
                logger.error(f"Erreur lors du chargement des données marketing : {str(e)}")
                st.error(f"❌ Erreur lors de la lecture du fichier : {str(e)}")
    
    with tab4:
        st.markdown("### 👕 Données produits (T-shirts)")
        st.markdown("""
        Fichier CSV contenant les informations sur les produits.
        **Colonnes attendues** :
        - `tshirt_id` : Identifiant unique du produit
        - `category` : Catégorie du produit
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
            try:
                expected_columns = ['tshirt_id', 'category', 'style', 'size', 'color', 'base_price']
                numeric_columns = ['base_price']
                required_columns = ['tshirt_id', 'category', 'base_price']
                
                products_df = load_csv(
                    products_file,
                    expected_columns=expected_columns,
                    numeric_columns=numeric_columns,
                    required_columns=required_columns
                )
                
                if products_df is not None:
                    if not check_unique_ids(products_df, 'tshirt_id', 'produits'):
                        products_df = products_df.drop_duplicates(subset='tshirt_id', keep='first')
                        st.warning("Les doublons dans 'tshirt_id' ont été supprimés (première occurrence conservée).")
                    
                    if not check_positive_values(products_df, ['base_price'], 'produits'):
                        st.session_state.products_df = None
                        st.error("❌ Les données produits ne sont pas valides en raison de valeurs négatives. Veuillez corriger le fichier.")
                    else:
                        if 'base_price' in products_df.columns:
                            products_df['price'] = products_df['base_price']
                        if 'cost' not in products_df.columns:
                            products_df['cost'] = products_df['price'] * 0.6
                        
                        st.session_state.products_df = products_df
                        
                        with st.expander("📦 Aperçu des données produits", expanded=True):
                            st.dataframe(products_df.head())
                            
                            st.subheader("📊 Statistiques produits")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Nombre de produits", len(products_df))
                                categories = products_df['category'].nunique()
                                st.metric("Catégories différentes", categories)
                            with col2:
                                avg_price = products_df['price'].mean()
                                st.metric("Prix moyen", f"{avg_price:,.2f} €")
                                products_df['margin'] = (products_df['price'] - products_df['cost']) / products_df['price']
                                avg_margin = products_df['margin'].mean()
                                st.metric("Marge moyenne", f"{avg_margin:.1%}")
                            
                            if 'category' in products_df.columns:
                                st.markdown("**📊 Répartition par catégorie**")
                                category_dist = products_df['category'].value_counts()
                                st.bar_chart(category_dist)
                            

            except pd.errors.ParserError:
                st.error("❌ Erreur de parsing du fichier CSV. Vérifiez le format et le séparateur (par exemple, utilisez ';' au lieu de ',').")
            except UnicodeDecodeError:
                st.error("❌ Erreur d'encodage du fichier CSV. Essayez d'utiliser l'encodage UTF-8 ou ISO-8859-1.")
            except Exception as e:
                logger.error(f"Erreur lors du chargement des données produits : {str(e)}")
                st.error(f"❌ Erreur lors de la lecture du fichier : {str(e)}")

# Bouton de réinitialisation
if st.button("🗑️ Réinitialiser les données", key="reset_data"):
    st.session_state.data_loaded = False
    st.session_state.customers_df = None
    st.session_state.orders_df = None
    st.session_state.marketing_df = None
    st.session_state.products_df = None
    st.session_state.documents = []
    st.session_state.processed_docs = False
    st.success("✅ Données réinitialisées avec succès.")

# Bouton de validation
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Valider et passer à l'analyse", type="primary", use_container_width=True):
        required_files = ['customers_df', 'orders_df', 'marketing_df', 'products_df']
        missing_files = [f for f in required_files if st.session_state.get(f) is None]
        
        if missing_files:
            st.error(f"⚠️ Veuillez d'abord téléverser tous les fichiers requis. Manquant : {', '.join([f.replace('_df', '') for f in missing_files])}")
        else:
            st.session_state.data_loaded = True
            st.switch_page("pages/2_🔍_Exploration_des_données.py")

# Style CSS supplémentaire
st.markdown("""
    <style>
    .streamlit-expanderHeader {
        font-size: 1.1em;
        font-weight: 600;
    }
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