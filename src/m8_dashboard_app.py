import os
import sys
import glob
import logging
import traceback
import zipfile
import io
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.m6_predictive_models import render_sales_objective_prediction_section
try:
    # Import facultatif pour entraînement du modèle d'objectif commercial depuis l'UI
    from src.m6_train_objective_model import train_objective_model, EXPECTED_FEATURES, TARGET_NAME
except Exception:
    train_objective_model = None
    EXPECTED_FEATURES = [
        "CA_mensuel",
        "nb_ventes",
        "panier_moyen",
        "taux_conversion",
        "prospects_qualifies",
        "taux_transformation",
    ]
    TARGET_NAME = "objectif_atteint"

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration Streamlit
st.set_page_config(
    page_title="TeeTech Analytics Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

class TeeTechAnalytics:
    def __init__(self):
        self.ensure_directories()
        self.targets = {
            'followers_target': 5000,
            'engagement_rate_min': 0.05,
            'ctr_min': 0.02,
            'cpc_max': 50.0,
            'cpa_max': 5000.0,
            'roi_min': 3.0,
            'monthly_revenue_target': 864000.0,
            'conversion_rate_target': 0.04,
            'satisfaction_min': 0.90,
            'nps_min': 50,
            'repeat_rate_min': 0.40
        }

    def ensure_directories(self):
        """Créer les répertoires nécessaires"""
        for directory in ['data', 'output', 'reports', 'visualizations']:
            os.makedirs(directory, exist_ok=True)

    def read_csv_robust(self, file_path, sample_size=4096):
        """Lecture robuste de fichiers CSV avec détection automatique"""
        if os.path.getsize(file_path) == 0:
            st.warning(f"Fichier vide: {file_path}")
            return pd.DataFrame()

        encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
        separators = [',', ';', '\t', '|']
        
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                    if len(df.columns) > 1:  # Validation basique
                        return df
                except:
                    continue
        
        # Dernier recours
        try:
            return pd.read_csv(file_path, sep=None, engine='python')
        except Exception as e:
            st.error(f"Impossible de lire le fichier {file_path}: {str(e)}")
            return pd.DataFrame()

    def process_uploaded_files(self, uploaded_files):
        """Traiter les fichiers uploadés"""
        if not uploaded_files:
            return False
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join('data', uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.read())
            st.success(f"Fichier {uploaded_file.name} uploadé avec succès!")
        
        return True

    def detect_file_type(self, filename):
        """Détecter le type de fichier basé sur le nom"""
        filename_lower = filename.lower()
        
        if any(word in filename_lower for word in ['customer', 'client']):
            return 'customers'
        elif any(word in filename_lower for word in ['product', 'tshirt', 'tee']):
            return 'products'
        elif any(word in filename_lower for word in ['sale', 'order', 'transaction']):
            return 'sales'
        elif any(word in filename_lower for word in ['marketing', 'campaign', 'ads']):
            return 'marketing'
        else:
            return 'unknown'

    def load_data(self):
        """Charger et détecter automatiquement les fichiers de données"""
        data_files = glob.glob(os.path.join('data', '*.csv'))
        datasets = {}
        
        for file_path in data_files:
            filename = os.path.basename(file_path)
            file_type = self.detect_file_type(filename)
            
            try:
                df = self.read_csv_robust(file_path)
                if not df.empty:
                    datasets[file_type] = df
                    st.info(f"📁 {filename} → Détecté comme '{file_type}' ({len(df)} lignes)")
            except Exception as e:
                st.error(f"Erreur lors du chargement de {filename}: {str(e)}")
        
        return datasets

    def clean_and_standardize_data(self, datasets):
        """Nettoyer et standardiser les données"""
        cleaned_data = {}
        
        # Nettoyage des données clients
        if 'customers' in datasets:
            df = datasets['customers'].copy()
            df.columns = df.columns.str.lower().str.strip()
            
            # Mapping des colonnes
            column_mapping = {
                'customer_id': ['customer_id', 'id', 'client_id', 'cust_id'],
                'age': ['age', 'âge'],
                'city': ['city', 'ville', 'location'],
                'client_type': ['client_type', 'type', 'category', 'segment'],
                'gender': ['gender', 'sex', 'genre'],
                'signup_date': ['signup_date', 'registration_date', 'created_at', 'date_inscription']
            }
            
            standardized = {}
            for target_col, possible_cols in column_mapping.items():
                for col in possible_cols:
                    if col in df.columns:
                        standardized[target_col] = df[col]
                        break
            
            if standardized:
                customers_clean = pd.DataFrame(standardized)
                # Nettoyage des types de données
                if 'age' in customers_clean.columns:
                    customers_clean['age'] = pd.to_numeric(customers_clean['age'], errors='coerce')
                if 'signup_date' in customers_clean.columns:
                    customers_clean['signup_date'] = pd.to_datetime(customers_clean['signup_date'], errors='coerce')
                # Normalisation du genre
                if 'gender' in customers_clean.columns:
                    def norm_gender(x):
                        if pd.isna(x):
                            return 'U'
                        s = str(x).strip().lower()
                        if s in ['m', 'male', 'homme', 'h']:
                            return 'H'
                        if s in ['f', 'female', 'femme']:
                            return 'F'
                        return 'U'
                    customers_clean['gender'] = customers_clean['gender'].apply(norm_gender)
                
                cleaned_data['customers'] = customers_clean
                customers_clean.to_csv('output/customers_clean.csv', index=False)

        # Nettoyage des données produits
        if 'products' in datasets:
            df = datasets['products'].copy()
            df.columns = df.columns.str.lower().str.strip()
            
            column_mapping = {
                'product_id': ['product_id', 'id', 'sku', 'code'],
                'name': ['name', 'product_name', 'designation', 'nom'],
                'category': ['category', 'categorie', 'type', 'famille'],
                'price': ['price', 'unit_price', 'prix', 'tarif']
            }
            
            standardized = {}
            for target_col, possible_cols in column_mapping.items():
                for col in possible_cols:
                    if col in df.columns:
                        standardized[target_col] = df[col]
                        break
            
            if standardized:
                products_clean = pd.DataFrame(standardized)
                if 'price' in products_clean.columns:
                    products_clean['price'] = pd.to_numeric(products_clean['price'], errors='coerce')
                
                cleaned_data['products'] = products_clean
                products_clean.to_csv('output/products_clean.csv', index=False)

        # Nettoyage des données de ventes
        if 'sales' in datasets:
            df = datasets['sales'].copy()
            df.columns = df.columns.str.lower().str.strip()
            
            column_mapping = {
                'order_id': ['order_id', 'id', 'sale_id', 'transaction_id'],
                'customer_id': ['customer_id', 'client_id', 'cust_id'],
                'product_id': ['product_id', 'sku', 'item_id'],
                'order_date': ['order_date', 'date', 'created_at', 'purchase_date'],
                'quantity': ['quantity', 'qty', 'qte'],
                'unit_price': ['unit_price', 'price', 'prix'],
                'total_amount': ['total_amount', 'amount', 'total', 'montant']
            }
            
            standardized = {}
            for target_col, possible_cols in column_mapping.items():
                for col in possible_cols:
                    if col in df.columns:
                        standardized[target_col] = df[col]
                        break
            
            if standardized:
                sales_clean = pd.DataFrame(standardized)
                
                # Nettoyage des types
                if 'order_date' in sales_clean.columns:
                    sales_clean['order_date'] = pd.to_datetime(sales_clean['order_date'], errors='coerce')
                for col in ['quantity', 'unit_price', 'total_amount']:
                    if col in sales_clean.columns:
                        sales_clean[col] = pd.to_numeric(sales_clean[col], errors='coerce')
                
                # Calcul du montant total si manquant
                if 'total_amount' not in sales_clean.columns or sales_clean['total_amount'].isna().all():
                    if 'quantity' in sales_clean.columns and 'unit_price' in sales_clean.columns:
                        sales_clean['total_amount'] = sales_clean['quantity'] * sales_clean['unit_price']
                
                cleaned_data['sales'] = sales_clean
                sales_clean.to_csv('output/sales_clean.csv', index=False)

        # Nettoyage des données marketing
        if 'marketing' in datasets:
            df = datasets['marketing'].copy()
            df.columns = df.columns.str.lower().str.strip()
            
            column_mapping = {
                'date': ['date', 'day', 'jour'],
                'channel': ['channel', 'canal', 'platform'],
                'campaign': ['campaign', 'campagne', 'campaign_name'],
                'impressions': ['impressions', 'impr', 'vues'],
                'clicks': ['clicks', 'clics'],
                'conversions': ['conversions', 'purchases', 'achats'],
                'cost': ['cost', 'spend', 'cout', 'budget'],
                'revenue': ['revenue', 'revenu', 'ca', 'chiffre_affaires']
            }
            
            standardized = {}
            for target_col, possible_cols in column_mapping.items():
                for col in possible_cols:
                    if col in df.columns:
                        standardized[target_col] = df[col]
                        break
            
            if standardized:
                marketing_clean = pd.DataFrame(standardized)
                
                if 'date' in marketing_clean.columns:
                    marketing_clean['date'] = pd.to_datetime(marketing_clean['date'], errors='coerce')
                
                for col in ['impressions', 'clicks', 'conversions', 'cost', 'revenue']:
                    if col in marketing_clean.columns:
                        marketing_clean[col] = pd.to_numeric(marketing_clean[col], errors='coerce')
                
                # Canal par défaut
                if 'channel' not in marketing_clean.columns:
                    marketing_clean['channel'] = 'Facebook'
                
                cleaned_data['marketing'] = marketing_clean
                marketing_clean.to_csv('output/marketing_clean.csv', index=False)

        return cleaned_data

    def perform_customer_segmentation(self, customers_data, sales_data):
        """Segmentation des clients avec RFM"""
        try:
            # Nettoyage et conversion des dates
            if 'order_date' in sales_data.columns:
                sales_data = sales_data.copy()
                sales_data['order_date'] = pd.to_datetime(sales_data['order_date'], errors='coerce')
                # Supprimer les lignes avec des dates invalides
                sales_data = sales_data.dropna(subset=['order_date'])
                
                has_sales = sales_data is not None and not sales_data.empty and 'order_date' in sales_data.columns
                if has_sales:
                    current_date = sales_data['order_date'].max()
                else:
                    current_date = datetime.now()
            else:
                current_date = datetime.now()
                st.warning("Colonne 'order_date' manquante. Utilisation de valeurs par défaut pour la récence.")
            
            # Calcul RFM avec gestion des erreurs
            if 'order_date' in sales_data.columns and not sales_data.empty:
                rfm = sales_data.groupby('customer_id').agg({
                    'order_date': lambda x: (current_date - x.max()).days if len(x) > 0 and pd.notna(x.max()) else 365,  # Recency
                    'order_id': 'nunique',  # Frequency  
                    'total_amount': 'sum'   # Monetary
                }).reset_index()
            else:
                # Fallback si pas de dates valides
                rfm = sales_data.groupby('customer_id').agg({
                    'order_id': 'nunique',  # Frequency
                    'total_amount': 'sum'   # Monetary
                }).reset_index()
                rfm['recency'] = 365  # Valeur par défaut
            
            rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
            
            # Validation des données RFM
            rfm['recency'] = pd.to_numeric(rfm['recency'], errors='coerce').fillna(365)
            rfm['frequency'] = pd.to_numeric(rfm['frequency'], errors='coerce').fillna(1)
            rfm['monetary'] = pd.to_numeric(rfm['monetary'], errors='coerce').fillna(0)
            
            # Ajout des données clients si disponibles
            if 'age' in customers_data.columns:
                customer_features = customers_data[['customer_id', 'age']].copy()
                customer_features['age'] = pd.to_numeric(customer_features['age'], errors='coerce').fillna(30)
                rfm = rfm.merge(customer_features, on='customer_id', how='left')
                rfm['age'] = rfm['age'].fillna(30)  # Âge par défaut
            
            # Préparation pour clustering
            feature_cols = ['recency', 'frequency', 'monetary']
            if 'age' in rfm.columns:
                feature_cols.append('age')
            
            X = rfm[feature_cols].copy()
            
            # Vérification qu'on a des données valides
            if X.empty or len(X) < 2:
                st.error("Pas assez de données pour effectuer la segmentation")
                return None, None, None
            
            # Nettoyage final des valeurs aberrantes
            for col in feature_cols:
                # Remplacer les valeurs infinies par NaN puis par la médiane
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)
                median_val = X[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X[col] = X[col].fillna(median_val)
            
            # Standardisation
            scaler = StandardScaler()
            try:
                X_scaled = scaler.fit_transform(X)
            except Exception as e:
                st.error(f"Erreur lors de la standardisation: {str(e)}")
                return None, None, None
            
            # Clustering K-means
            optimal_k = self.find_optimal_clusters(X_scaled, max_k=min(8, len(X)))
            
            try:
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                rfm['cluster'] = kmeans.fit_predict(X_scaled)
                
                # Score de silhouette
                if len(np.unique(rfm['cluster'])) > 1:
                    silhouette_avg = silhouette_score(X_scaled, rfm['cluster'])
                else:
                    silhouette_avg = 0
                    
            except Exception as e:
                st.error(f"Erreur lors du clustering: {str(e)}")
                # Fallback: segmentation simple basée sur la valeur monétaire
                rfm['cluster'] = pd.cut(rfm['monetary'], bins=3, labels=[0,1,2]).astype(int)
                silhouette_avg = 0
                optimal_k = 3
                st.warning("Utilisation d'une segmentation simplifiée basée sur la valeur monétaire.")
            
            # Sauvegarde
            rfm.to_csv('output/customer_segments.csv', index=False)
            
            return rfm, silhouette_avg, optimal_k
            
        except Exception as e:
            st.error(f"Erreur lors de la segmentation: {str(e)}")
            # Informations de debug
            if 'sales_data' in locals():
                st.write("**Colonnes dans sales_data:**", list(sales_data.columns))
                if 'order_date' in sales_data.columns:
                    st.write("**Types de données dans order_date:**", sales_data['order_date'].dtype)
                    st.write("**Exemples de order_date:**", sales_data['order_date'].head().tolist())
            return None, None, None

    def find_optimal_clusters(self, X, max_k=8):
        """Trouver le nombre optimal de clusters"""
        try:
            n_samples = len(X)
            if n_samples < 4:
                return 2
            
            # Limiter max_k selon le nombre d'échantillons
            max_k = min(max_k, n_samples - 1, 8)
            
            if max_k < 2:
                return 2
            
            silhouette_scores = []
            K_range = range(2, max_k + 1)
            
            for k in K_range:
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X)
                    
                    # Vérifier qu'il y a au moins 2 clusters différents
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(X, labels)
                        silhouette_scores.append(score)
                    else:
                        silhouette_scores.append(-1)  # Score faible si un seul cluster
                        
                except Exception as e:
                    st.warning(f"Erreur pour k={k}: {str(e)}")
                    silhouette_scores.append(-1)
            
            if not silhouette_scores or all(score <= 0 for score in silhouette_scores):
                return 3  # Valeur par défaut
            
            best_k_index = silhouette_scores.index(max(silhouette_scores))
            optimal_k = K_range[best_k_index]
            return optimal_k
            
        except Exception as e:
            st.warning(f"Erreur dans find_optimal_clusters: {str(e)}")
            return 3  # Valeur par défaut sûre

    def create_customer_personas(self, segments_data, customers_data, sales_data, products_data):
        """Créer des personas détaillés"""
        try:
            # Fonction helper pour obtenir le mode en sécurité
            def safe_mode(series, fallback='Unknown'):
                try:
                    if series.empty or series.isna().all():
                        return fallback
                    mode_values = series.mode(dropna=True)
                    return mode_values.iloc[0] if len(mode_values) > 0 else fallback
                except Exception:
                    return fallback
            
            # Préparation des données de ventes enrichies
            sales_enriched = sales_data.copy()
            
            # Enrichissement avec données produits si disponibles et compatibles
            if (not products_data.empty and 
                'product_id' in products_data.columns and 
                'product_id' in sales_data.columns):
                
                try:
                    # Assurer la compatibilité des types pour le merge
                    products_clean = products_data.copy()
                    products_clean['product_id'] = products_clean['product_id'].astype(str)
                    sales_enriched['product_id'] = sales_enriched['product_id'].astype(str)
                    
                    # Sélectionner seulement les colonnes nécessaires
                    merge_cols = ['product_id']
                    if 'category' in products_clean.columns:
                        merge_cols.append('category')
                    elif 'name' in products_clean.columns:
                        # Utiliser le nom comme proxy de catégorie
                        products_clean['category'] = products_clean['name']
                        merge_cols.append('category')
                    
                    if len(merge_cols) > 1:
                        sales_enriched = sales_enriched.merge(
                            products_clean[merge_cols], 
                            on='product_id', 
                            how='left'
                        )
                        st.info(f"✅ Données produits intégrées: {len(products_clean)} produits")
                    else:
                        sales_enriched['category'] = 'Unknown'
                        st.warning("⚠️ Pas de catégorie trouvée dans les données produits")
                        
                except Exception as e:
                    sales_enriched['category'] = 'Unknown'
                    st.warning(f"⚠️ Impossible d'intégrer les données produits: {str(e)}")
            else:
                sales_enriched['category'] = 'Unknown'
                if products_data.empty:
                    st.info("ℹ️ Pas de données produits disponibles")
            
            personas = {}
            
            # Vérification des colonnes essentielles
            if 'cluster' not in segments_data.columns:
                st.error("❌ Colonne 'cluster' manquante dans les données de segmentation")
                return {}
            
            if 'customer_id' not in segments_data.columns:
                st.error("❌ Colonne 'customer_id' manquante dans les données de segmentation")
                return {}
                
            # Création des personas pour chaque cluster
            for cluster_id in sorted(segments_data['cluster'].unique()):
                try:
                    cluster_customers = segments_data[segments_data['cluster'] == cluster_id]['customer_id']
                    
                    if len(cluster_customers) == 0:
                        continue
                    
                    # Statistiques RFM du cluster
                    cluster_rfm = segments_data[segments_data['cluster'] == cluster_id]
                    
                    # Statistiques démographiques
                    cluster_demo = customers_data[customers_data['customer_id'].isin(cluster_customers)]
                    
                    # Comportement d'achat
                    cluster_sales = sales_enriched[sales_enriched['customer_id'].isin(cluster_customers)]
                    
                    # Catégories préférées
                    top_categories = {}
                    if not cluster_sales.empty and 'category' in cluster_sales.columns:
                        category_counts = cluster_sales['category'].value_counts().head(5)
                        top_categories = category_counts.to_dict()
                    
                    # Calcul des métriques avec gestion des erreurs
                    def safe_mean(series, default=0):
                        try:
                            return series.mean() if not series.empty and not series.isna().all() else default
                        except:
                            return default
                    
                    def safe_sum(series, default=0):
                        try:
                            return series.sum() if not series.empty and not series.isna().all() else default
                        except:
                            return default
                    
                    # Construction du persona
                    persona = {
                        'size': len(cluster_customers),
                        'avg_recency': safe_mean(cluster_rfm['recency']) if 'recency' in cluster_rfm.columns else 0,
                        'avg_frequency': safe_mean(cluster_rfm['frequency']) if 'frequency' in cluster_rfm.columns else 0,
                        'avg_monetary': safe_mean(cluster_rfm['monetary']) if 'monetary' in cluster_rfm.columns else 0,
                        'avg_age': safe_mean(cluster_demo['age']) if 'age' in cluster_demo.columns else None,
                        'top_city': safe_mode(cluster_demo['city']) if 'city' in cluster_demo.columns else 'Unknown',
                        'top_categories': top_categories,
                        'total_revenue': safe_sum(cluster_sales['total_amount']) if 'total_amount' in cluster_sales.columns else 0,
                        'avg_order_value': safe_mean(cluster_sales['total_amount']) if 'total_amount' in cluster_sales.columns else 0,
                        'order_count': len(cluster_sales) if not cluster_sales.empty else 0
                    }
                    
                    personas[f"Segment_{cluster_id}"] = persona
                    
                except Exception as e:
                    st.warning(f"⚠️ Erreur lors de la création du persona pour le cluster {cluster_id}: {str(e)}")
                    continue
            
            if not personas:
                st.error("❌ Aucun persona n'a pu être créé")
                return {}
                
            st.success(f"✅ {len(personas)} personas créés avec succès!")
            return personas
            
        except Exception as e:
            st.error(f"❌ Erreur générale lors de la création des personas: {str(e)}")
            # Debug info
            if 'segments_data' in locals():
                st.write("**Colonnes dans segments_data:**", list(segments_data.columns))
            if 'sales_data' in locals():
                st.write("**Colonnes dans sales_data:**", list(sales_data.columns))
            if 'products_data' in locals() and not products_data.empty:
                st.write("**Colonnes dans products_data:**", list(products_data.columns))
            return {}

    def calculate_marketing_kpis(self, marketing_data):
        """Calculer les KPIs marketing"""
        try:
            df = marketing_data.copy()
            # Assurer des types numériques pour les colonnes clés
            num_cols = ['impressions', 'clicks', 'conversions', 'cost', 'revenue']
            for c in num_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            # Remplacer NaN par 0 pour éviter les erreurs dans les agrégations
            for c in num_cols:
                if c in df.columns:
                    df[c] = df[c].fillna(0)
            
            # KPIs de base
            df['ctr'] = df['clicks'] / df['impressions'].replace(0, np.nan)
            df['cvr'] = df['conversions'] / df['clicks'].replace(0, np.nan)
            df['cpc'] = df['cost'] / df['clicks'].replace(0, np.nan)
            df['cpa'] = df['cost'] / df['conversions'].replace(0, np.nan)
            
            if 'revenue' in df.columns:
                df['roas'] = df['revenue'] / df['cost'].replace(0, np.nan)
                df['roi'] = (df['revenue'] - df['cost']) / df['cost'].replace(0, np.nan)
            
            # Agrégation globale
            summary = {
                'total_impressions': float(df['impressions'].sum()) if 'impressions' in df.columns else 0.0,
                'total_clicks': float(df['clicks'].sum()) if 'clicks' in df.columns else 0.0,
                'total_conversions': float(df['conversions'].sum()) if 'conversions' in df.columns else 0.0,
                'total_cost': float(df['cost'].sum()) if 'cost' in df.columns else 0.0,
                'total_revenue': float(df['revenue'].sum()) if 'revenue' in df.columns else 0.0,
            }
            
            # KPIs globaux
            summary['overall_ctr'] = summary['total_clicks'] / summary['total_impressions'] if summary['total_impressions'] > 0 else 0
            summary['overall_cvr'] = summary['total_conversions'] / summary['total_clicks'] if summary['total_clicks'] > 0 else 0
            summary['overall_cpc'] = summary['total_cost'] / summary['total_clicks'] if summary['total_clicks'] > 0 else 0
            summary['overall_cpa'] = summary['total_cost'] / summary['total_conversions'] if summary['total_conversions'] > 0 else 0
            summary['overall_roas'] = summary['total_revenue'] / summary['total_cost'] if summary['total_cost'] > 0 else 0
            
            # Benchmark vs objectifs
            benchmarks = []
            for kpi, value in [
                ('CTR', summary['overall_ctr']),
                ('CPC', summary['overall_cpc']),
                ('CPA', summary['overall_cpa']),
                ('ROAS', summary['overall_roas'])
            ]:
                target_key = f"{kpi.lower()}_{'min' if kpi in ['CTR', 'ROAS'] else 'max'}"
                target = self.targets.get(target_key, 0)
                
                # Forcer la comparaison sur des floats robustes
                try:
                    v = float(value) if value is not None else 0.0
                except Exception:
                    v = 0.0
                try:
                    t = float(target) if target is not None else 0.0
                except Exception:
                    t = 0.0

                if kpi in ['CTR', 'ROAS']:
                    status = 'Atteint' if v >= t else 'À améliorer'
                else:
                    status = 'Atteint' if v <= t else 'À améliorer'
                
                benchmarks.append({
                    'KPI': kpi,
                    'Valeur': v,
                    'Objectif': t,
                    'Statut': status
                })
            
            df.to_csv('output/campaign_kpis.csv', index=False)
            pd.DataFrame(benchmarks).to_csv('output/kpi_benchmark.csv', index=False)
            
            return df, summary, benchmarks
            
        except Exception as e:
            st.error(f"Erreur lors du calcul des KPIs: {str(e)}")
            return None, None, None

    def build_predictive_model(self, customers_data, sales_data):
        """Construire un modèle prédictif de churn"""
        try:
            # Préparation des données pour prédiction churn
            if 'order_date' in sales_data.columns:
                sales_data_clean = sales_data.copy()
                sales_data_clean['order_date'] = pd.to_datetime(sales_data_clean['order_date'], errors='coerce')
                sales_data_clean = sales_data_clean.dropna(subset=['order_date'])
                
                if not sales_data_clean.empty:
                    current_date = sales_data_clean['order_date'].max()
                else:
                    current_date = datetime.now()
                    sales_data_clean = sales_data.copy()
            else:
                current_date = datetime.now()
                sales_data_clean = sales_data.copy()
                st.warning("⚠️ Pas de colonne de dates. Utilisation de valeurs par défaut.")
            
            # Vérification des colonnes essentielles
            required_cols = ['customer_id', 'total_amount']
            missing_cols = [col for col in required_cols if col not in sales_data_clean.columns]
            
            if missing_cols:
                st.error(f"❌ Colonnes manquantes pour le modèle prédictif: {missing_cols}")
                return None, None, None
            
            # Calcul des features avec gestion robuste
            try:
                # Création de la colonne order_id si manquante
                if 'order_id' not in sales_data_clean.columns:
                    sales_data_clean['order_id'] = range(len(sales_data_clean))

                # Named aggregation pour éviter les colonnes multi-index
                agg_kwargs = {
                    'total_spent': ('total_amount', 'sum'),
                    'avg_order_value': ('total_amount', 'mean'),
                    'orders_count': ('total_amount', 'count'),
                }
                if 'order_id' in sales_data_clean.columns:
                    agg_kwargs['order_id_nunique'] = ('order_id', 'nunique')
                if 'order_date' in sales_data_clean.columns:
                    agg_kwargs['recency'] = (
                        'order_date',
                        lambda x: (current_date - x.max()).days if len(x) > 0 and pd.notna(x.max()) else 365,
                    )

                customer_features = (
                    sales_data_clean
                    .groupby('customer_id')
                    .agg(**agg_kwargs)
                    .reset_index()
                )

                # Déduire frequency depuis orders_count ou order_id_nunique
                if 'order_id_nunique' in customer_features.columns:
                    customer_features['frequency'] = customer_features['order_id_nunique']
                else:
                    customer_features['frequency'] = customer_features['orders_count']

                # Valeurs par défaut si certains champs manquent
                if 'recency' not in customer_features.columns:
                    customer_features['recency'] = 365
                for c in ['total_spent', 'avg_order_value', 'frequency']:
                    if c not in customer_features.columns:
                        customer_features[c] = 0

            except Exception as e:
                st.error(f"❌ Erreur lors de l'agrégation des données: {str(e)}")
                st.write({'agg_input_cols': list(sales_data_clean.columns)})
                return None, None, None
            
            # Merge avec données clients si disponibles
            if any(col in customers_data.columns for col in ['age', 'gender', 'city']):
                try:
                    demo_cols = ['customer_id']
                    if 'age' in customers_data.columns:
                        demo_cols.append('age')
                    if 'gender' in customers_data.columns:
                        demo_cols.append('gender')
                    if 'city' in customers_data.columns:
                        demo_cols.append('city')
                    customer_demo = customers_data[demo_cols].copy()
                    if 'age' in customer_demo.columns:
                        customer_demo['age'] = pd.to_numeric(customer_demo['age'], errors='coerce').fillna(30)
                    # Fusion
                    customer_features = customer_features.merge(customer_demo, on='customer_id', how='left')
                    if 'age' in customer_features.columns:
                        customer_features['age'] = customer_features['age'].fillna(30)
                    # Encodage one-hot du genre (H/F/U)
                    if 'gender' in customer_features.columns:
                        customer_features['gender'] = customer_features['gender'].fillna('U')
                        gender_dummies = pd.get_dummies(customer_features['gender'], prefix='gender')
                        # Conserver H et F, U optionnel
                        for col in ['gender_H', 'gender_F']:
                            if col not in gender_dummies.columns:
                                gender_dummies[col] = 0
                        customer_features = pd.concat([customer_features, gender_dummies[['gender_H', 'gender_F']]], axis=1)
                    # Encodage one-hot des villes (top 10)
                    if 'city' in customer_features.columns:
                        # Déterminer top villes sur l'ensemble clients dispo
                        try:
                            top_cities = (
                                customers_data['city'].dropna().astype(str).str.strip().value_counts().head(10).index.tolist()
                            )
                        except Exception:
                            top_cities = []
                        customer_features['city'] = customer_features['city'].astype(str).str.strip().fillna('Autres')
                        customer_features['city_encoded'] = customer_features['city'].where(customer_features['city'].isin(top_cities), 'Autres')
                        city_dummies = pd.get_dummies(customer_features['city_encoded'], prefix='city')
                        # S'assurer que colonnes existent même si absentes
                        ensure_cols = [f'city_{c}' for c in top_cities] + ['city_Autres']
                        for col in ensure_cols:
                            if col not in city_dummies.columns:
                                city_dummies[col] = 0
                        customer_features = pd.concat([customer_features, city_dummies[ensure_cols]], axis=1)
                except Exception as e:
                    st.warning(f"⚠️ Impossible d'intégrer les données démographiques: {str(e)}")
            
            # Définition du churn (pas d'achat depuis X jours)
            churn_threshold = 90  # jours
            customer_features['is_churned'] = (customer_features['recency'] > churn_threshold).astype(int)
            
            # Préparation des features pour le modèle
            feature_columns = ['recency', 'frequency', 'total_spent', 'avg_order_value']
            if 'age' in customer_features.columns:
                feature_columns.append('age')
            if 'gender_H' in customer_features.columns:
                feature_columns.append('gender_H')
            if 'gender_F' in customer_features.columns:
                feature_columns.append('gender_F')
            # Ajouter villes encodées (limiter à 11 colonnes max: top10 + autres)
            city_cols = [c for c in customer_features.columns if c.startswith('city_')]
            # Garder au plus 11 colonnes pour éviter haute dimension
            if city_cols:
                feature_columns.extend(city_cols[:11])

            # Validation et nettoyage des features
            X = customer_features[feature_columns].copy()

            # S'assurer que les colonnes existent et sont bien formées
            missing_feats = [c for c in feature_columns if c not in X.columns]
            if missing_feats:
                st.error(f"❌ Colonnes de features manquantes: {missing_feats}")
                return None, None, None

            for col in feature_columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                # Remplacer les valeurs infinies
                median_val = X[col].replace([np.inf, -np.inf], np.nan).median()
                if pd.isna(median_val):
                    median_val = 0
                X[col] = X[col].replace([np.inf, -np.inf], median_val).fillna(median_val)

            # y en Series 1-D
            y = pd.Series(customer_features['is_churned']).astype(int)
            
            # Vérification qu'on a assez de données
            if len(X) < 10:
                st.warning("⚠️ Pas assez de données pour un modèle robuste (minimum 10 clients)")
                return None, None, None
            
            # Vérification qu'on a des cas de churn et de non-churn
            if len(y.unique()) < 2:
                st.warning("⚠️ Tous les clients ont le même statut de churn. Ajustement du seuil.")
                # Ajuster le seuil de churn
                churn_threshold = customer_features['recency'].median()
                customer_features['is_churned'] = (customer_features['recency'] > churn_threshold).astype(int)
                # Assurer que y reste une Series 1-D après réajustement
                y = pd.Series(customer_features['is_churned']).astype(int)
                
                if len(y.unique()) < 2:
                    st.error("❌ Impossible de créer un modèle de churn avec ces données")
                    return None, None, None
            
            # Division train/test
            # Coercition finale des types
            X = X.astype(float)
            y = pd.Series(y).astype(int)

            try:
                test_size = 0.2 if len(X) >= 20 else 0.3
                # Stratification conditionnelle selon la disponibilité des classes
                class_counts = y.value_counts()
                can_stratify = (class_counts.min() >= 2) and (len(class_counts) >= 2)
                stratify_arg = y if can_stratify else None
                if not can_stratify:
                    st.info("ℹ️ Stratification désactivée (classes insuffisantes). Split aléatoire simple utilisé.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=stratify_arg
                )
                # Coercition après split
                X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
                X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
                y_train = pd.Series(y_train).astype(int)
                y_test = pd.Series(y_test).astype(int)
                # Aplatir pour sklearn si nécessaire
                y_train = y_train.to_numpy().ravel()
                y_test = y_test.to_numpy().ravel()
            except Exception as e:
                st.error(f"❌ Erreur lors du split train/test: {str(e)}")
                st.write({
                    'X_shape': X.shape,
                    'y_len': len(y),
                    'y_unique': y.unique().tolist() if hasattr(y, 'unique') else None
                })
                return None, None, None
            
            # Entraînement du modèle
            model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                st.error(f"❌ Erreur lors de l'entraînement du modèle: {str(e)}")
                st.write({
                    'X_train_shape': X_train.shape,
                    'y_train_type': type(y_train).__name__,
                    'y_train_shape': getattr(y_train, 'shape', None),
                    'y_train_preview': y_train[:10].tolist() if hasattr(y_train, 'tolist') else None
                })
                return None, None, None
            
            # Prédictions et métriques
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, y_pred_proba)
            except Exception as e:
                st.warning(f"⚠️ Impossible de calculer l'AUC: {str(e)}")
                st.write({'y_test_type': type(y_test).__name__, 'y_test_head': getattr(y_test, 'head', lambda: y_test)()})
                auc_score = 0
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Prédictions sur tous les clients
            try:
                all_predictions = model.predict_proba(X)[:, 1]
                customer_features['churn_probability'] = all_predictions
                
                # Définition des niveaux de risque
                customer_features['churn_risk'] = pd.cut(
                    customer_features['churn_probability'],
                    bins=[0, 0.3, 0.7, 1.0],
                    labels=['Faible', 'Moyen', 'Élevé'],
                    include_lowest=True
                )
                
            except Exception as e:
                st.error(f"❌ Erreur lors des prédictions: {str(e)}")
                return None, None, None
            
            # Sauvegarde
            try:
                customer_features.to_csv('output/churn_predictions.csv', index=False)
                feature_importance.to_csv('output/feature_importance.csv', index=False)
            except Exception as e:
                st.warning(f"⚠️ Erreur lors de la sauvegarde: {str(e)}")
            
            return customer_features, feature_importance, auc_score
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la construction du modèle prédictif: {str(e)}")
            # Debug info
            if 'sales_data' in locals():
                st.write("**Colonnes dans sales_data:**", list(sales_data.columns))
            if 'customers_data' in locals():
                st.write("**Colonnes dans customers_data:**", list(customers_data.columns))
            return None, None, None

    def create_visualizations(self, customers_data, sales_data, marketing_data, segments_data, personas):
        """Créer toutes les visualisations"""
        viz_results = {}
        
        # 1. Distribution des segments
        if segments_data is not None and not segments_data.empty:
            fig_segments = px.pie(
                values=segments_data['cluster'].value_counts().values,
                names=[f'Segment {i}' for i in segments_data['cluster'].value_counts().index],
                title='Distribution des Segments Clients'
            )
            viz_results['segments_distribution'] = fig_segments
        
        # 2. RFM Analysis
        if segments_data is not None and not segments_data.empty:
            fig_rfm = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Récence', 'Fréquence', 'Valeur Monétaire', 'RFM par Segment')
            )
            
            for i, col in enumerate(['recency', 'frequency', 'monetary']):
                if col in segments_data.columns:
                    fig_rfm.add_trace(
                        go.Histogram(x=segments_data[col], name=col.title()),
                        row=(i//2)+1, col=(i%2)+1
                    )
            
            # Scatter RFM par segment
            if all(col in segments_data.columns for col in ['frequency', 'monetary', 'cluster']):
                fig_rfm.add_trace(
                    go.Scatter(
                        x=segments_data['frequency'],
                        y=segments_data['monetary'],
                        mode='markers',
                        color=segments_data['cluster'],
                        name='Segments'
                    ),
                    row=2, col=2
                )
            
            fig_rfm.update_layout(height=600, title_text="Analyse RFM")
            viz_results['rfm_analysis'] = fig_rfm
        
        # 3. Évolution des ventes
        has_sales = sales_data is not None and not sales_data.empty and 'order_date' in sales_data.columns
        if has_sales:
            sales_timeline = sales_data.groupby(sales_data['order_date'].dt.date)['total_amount'].sum().reset_index()
            
            fig_timeline = px.line(
                sales_timeline,
                x='order_date',
                y='total_amount',
                title='Évolution du Chiffre d\'Affaires'
            )
            viz_results['sales_timeline'] = fig_timeline
        
        # 4. Performance Marketing
        if not marketing_data.empty:
            # KPIs par canal
            if 'channel' in marketing_data.columns:
                channel_performance = marketing_data.groupby('channel').agg({
                    'impressions': 'sum',
                    'clicks': 'sum',
                    'cost': 'sum',
                    'conversions': 'sum' if 'conversions' in marketing_data.columns else lambda x: 0
                }).reset_index()
                
                channel_performance['ctr'] = channel_performance['clicks'] / channel_performance['impressions']
                channel_performance['cpc'] = channel_performance['cost'] / channel_performance['clicks']
                
                fig_channels = px.bar(
                    channel_performance,
                    x='channel',
                    y=['impressions', 'clicks', 'conversions'],
                    title='Performance par Canal Marketing'
                )
                viz_results['channel_performance'] = fig_channels
        
        # 5. Personas Comparison
        if personas:
            personas_df = pd.DataFrame.from_dict(personas, orient='index').reset_index()
            personas_df.columns = ['Segment'] + list(personas_df.columns[1:])
            
            fig_personas = px.bar(
                personas_df,
                x='Segment',
                y='avg_monetary',
                title='Valeur Moyenne par Segment',
                color='avg_frequency'
            )
            viz_results['personas_comparison'] = fig_personas
        
        return viz_results

    def generate_comprehensive_report(self, customers_data, sales_data, marketing_data, 
                                    segments_data, personas, kpi_summary, churn_data):
        """Générer un rapport complet"""
        # Utiliser les ventes traitées si disponibles, sinon reconstruire
        sd = (st.session_state.get('sales_processed') if 'sales_processed' in st.session_state else sales_data).copy()
        default_unit_price = float(st.session_state.get('default_unit_price', 0.0))
        # Reconstruire total_amount si manquant/NaN/0
        if not sd.empty:
            unit_cols = [c for c in sd.columns if c in ['unit_price', 'price', 'prix']]
            qty_cols = [c for c in sd.columns if c in ['quantity', 'qty', 'quantite', 'quantité']]
            qt = pd.Series(0, index=sd.index)
            if qty_cols:
                qt = pd.to_numeric(sd[qty_cols[0]].astype(str).str.replace(r"[^0-9,.-]", "", regex=True).str.replace(',', '.'), errors='coerce').fillna(0)
            if unit_cols:
                up = pd.to_numeric(sd[unit_cols[0]].astype(str).str.replace(r"[^0-9,.-]", "", regex=True).str.replace(',', '.'), errors='coerce').fillna(default_unit_price)
            else:
                up = pd.Series(default_unit_price, index=sd.index)
            if 'total_amount' not in sd.columns:
                sd['total_amount'] = (up * qt)
            else:
                ta_clean = pd.to_numeric(sd['total_amount'].astype(str).str.replace(r"[^0-9,.-]", "", regex=True).str.replace(',', '.'), errors='coerce')
                sd['total_amount'] = ta_clean.where(ta_clean.notna() & (ta_clean > 0), (up * qt))
        # Sécurisation des types
        if 'order_date' in sd.columns:
            sd['order_date'] = pd.to_datetime(sd['order_date'], errors='coerce')
        if 'total_amount' in sd.columns:
            sd['total_amount'] = pd.to_numeric(sd['total_amount'], errors='coerce')
        # Valeurs sûres
        total_ca = float(sd['total_amount'].sum()) if 'total_amount' in sd.columns else 0.0
        avg_basket = float(sd['total_amount'].mean()) if 'total_amount' in sd.columns else 0.0
        order_count = int(sd['order_id'].nunique()) if 'order_id' in sd.columns else len(sd)
        active_customers = int(sd['customer_id'].nunique()) if 'customer_id' in sd.columns else 0
        period_min = sd['order_date'].min() if 'order_date' in sd.columns else None
        period_max = sd['order_date'].max() if 'order_date' in sd.columns else None
        # kpi_summary sécurisé
        ks = kpi_summary or {}
        def fget(d, k, default=0.0):
            try:
                return float(d.get(k, default))
            except Exception:
                return default
        ks_total_impr = fget(ks, 'total_impressions')
        ks_total_clicks = fget(ks, 'total_clicks')
        ks_total_conv = fget(ks, 'total_conversions')
        ks_total_cost = fget(ks, 'total_cost')
        ks_total_rev = fget(ks, 'total_revenue')
        ks_ctr = fget(ks, 'overall_ctr')
        ks_cvr = fget(ks, 'overall_cvr')
        ks_cpc = fget(ks, 'overall_cpc')
        ks_cpa = fget(ks, 'overall_cpa')
        ks_roas = fget(ks, 'overall_roas')
        report_content = f"""
# RAPPORT D'ANALYSE MARKETING - TEETECH DESIGN
**Généré le : {datetime.now().strftime('%d/%m/%Y à %H:%M')}**

## RÉSUMÉ EXÉCUTIF

### Données Analysées
- **Clients** : {len(customers_data)} enregistrements
- **Ventes** : {len(sd)} transactions
- **Campagnes Marketing** : {len(marketing_data)} entrées
- **Période d'analyse** : {period_min.strftime('%d/%m/%Y') if period_min is not None and pd.notna(period_min) else 'Non disponible'} - {period_max.strftime('%d/%m/%Y') if period_max is not None and pd.notna(period_max) else 'Non disponible'}

### Indicateurs Clés
- **Chiffre d'Affaires Total** : {total_ca:,.0f} Ar
- **Panier Moyen** : {avg_basket:,.0f} Ar
- **Nombre de Commandes** : {order_count}
- **Clients Actifs** : {active_customers if active_customers>0 else 'N/A'}

## ANALYSE DE SEGMENTATION

### Vue d'ensemble
{f"Nombre de segments identifiés : {segments_data['cluster'].nunique()}" if segments_data is not None else "Segmentation non disponible"}

### Profils des Segments"""

        # Ajout des personas
        if personas:
            for segment_name, persona_data in personas.items():
                report_content += f"""

#### {segment_name.replace('_', ' ').title()}
- **Taille** : {persona_data['size']} clients ({persona_data['size']/len(customers_data)*100:.1f}% de la base)
- **Récence moyenne** : {persona_data['avg_recency']:.0f} jours
- **Fréquence d'achat** : {persona_data['avg_frequency']:.1f} commandes
- **Valeur moyenne** : {persona_data['avg_monetary']:,.0f} Ar
- **Âge moyen** : {persona_data['avg_age']:.0f} ans (si disponible)
- **Ville principale** : {persona_data['top_city']}
- **Catégories préférées** : {', '.join(list(persona_data['top_categories'].keys())[:3])}
- **Revenus générés** : {persona_data['total_revenue']:,.0f} Ar"""

        # Performance Marketing
        if kpi_summary:
            report_content += f"""

## PERFORMANCE MARKETING

### KPIs Globaux
- **Impressions totales** : {ks_total_impr:,.0f}
- **Clics totaux** : {ks_total_clicks:,.0f}
- **Conversions** : {ks_total_conv:,.0f}
- **Coût total** : {ks_total_cost:,.0f} Ar
- **Revenus** : {ks_total_rev:,.0f} Ar

### Ratios de Performance
- **CTR** : {ks_ctr*100:.2f}%
- **CVR** : {ks_cvr*100:.2f}%
- **CPC** : {ks_cpc:,.0f} Ar
- **CPA** : {ks_cpa:,.0f} Ar
- **ROAS** : {ks_roas:.2f}x"""

        # Analyse Prédictive
        if churn_data is not None:
            churn_stats = churn_data['churn_risk'].value_counts()
            report_content += f"""

## ANALYSE PRÉDICTIVE - RISQUE DE CHURN

### Distribution du Risque
- **Risque Faible** : {churn_stats.get('Faible', 0)} clients
- **Risque Moyen** : {churn_stats.get('Moyen', 0)} clients  
- **Risque Élevé** : {churn_stats.get('Élevé', 0)} clients

### Recommandations par Niveau de Risque
#### Clients à Risque Élevé ({churn_stats.get('Élevé', 0)} clients)
- Campagnes de rétention immédiates
- Offres personnalisées exclusives
- Contact direct par l'équipe commerciale

#### Clients à Risque Moyen ({churn_stats.get('Moyen', 0)} clients)
- Programmes de fidélisation
- Communications ciblées
- Offres incitatives

#### Clients à Faible Risque ({churn_stats.get('Faible', 0)} clients)
- Maintenir l'engagement
- Programmes d'ambassadeurs
- Cross-selling/Up-selling"""

        report_content += """

## RECOMMANDATIONS STRATÉGIQUES

### Court Terme (1-3 mois)
1. **📈 Rétention Clients** : Lancer des campagnes ciblées pour les clients à risque élevé
2. **📈 Optimisation Marketing** : Réallouer le budget vers les canaux les plus performants
3. **📈 Personnalisation** : Adapter les offres par segment client

### Moyen Terme (3-6 mois)
1. **📈 Développement Produits** : Créer des produits adaptés aux segments les plus rentables
2. **📈 Acquisition** : Cibler des prospects similaires aux clients à forte valeur
3. **📈 Automatisation** : Mettre en place des workflows marketing automatisés

### Long Terme (6-12 mois)
1. **📈 Expansion** : Explorer de nouveaux marchés basés sur l'analyse des segments
2. **📈 Innovation** : Développer de nouvelles catégories de produits
3. **📈 Fidélisation** : Créer un programme de fidélité complet

## CONCLUSION

Cette analyse révèle des opportunités significatives d'optimisation de la stratégie marketing de TeeTech Design. 
La segmentation clientèle permet une approche plus ciblée, tandis que l'analyse prédictive aide à prévenir le churn.

Les recommandations proposées, si mises en œuvre, devraient permettre d'atteindre les objectifs fixés :
- Augmentation du chiffre d'affaires de 20%
- Amélioration du taux de conversion à 4%
- Réduction du coût d'acquisition client

---
*Rapport généré automatiquement par le système d'analyse TeeTech*
"""
        
        # Sauvegarde du rapport
        with open('reports/rapport_complet.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_content

def main():
    # En-tête
    st.markdown('<h1 class="main-header">🎯 TeeTech Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialisation
    analytics = TeeTechAnalytics()
    
    # Sidebar pour navigation
    with st.sidebar:
        st.header("📋 Menu Principal")
        
        selected_tab = st.selectbox(
            "Sélectionner un module",
            ["🏠 Accueil", "📤 Import de Données", "🧹 M2: Nettoyage", 
             "👥 M3: Segmentation", "🎭 M4: Personas", "📊 M5: Marketing KPIs", 
             "🔮 M6: Analyse Prédictive", "📈 Dashboard Complet", "📑 Rapport Final"]
        )
        
        st.markdown("---")
        st.info("💡 **Tip**: Commencez par importer vos données, puis suivez les modules dans l'ordre.")

    # Module Accueil
    if selected_tab == "🏠 Accueil":
        st.header("Bienvenue dans TeeTech Analytics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Modules", "8", "Complets")
        with col2:
            st.metric("🎯 KPIs", "15+", "Suivis")
        with col3:
            st.metric("📈 Visualisations", "20+", "Graphiques")
        
        st.markdown("""
        ### 🚀 Guide de Démarrage Rapide
        
        1. **📤 Import de Données** : Uploadez vos fichiers CSV
        2. **🧹 M2: Nettoyage** : Standardisation automatique des données
        3. **👥 M3: Segmentation** : Classification RFM des clients
        4. **🎭 M4: Personas** : Profils détaillés par segment
        5. **📊 M5: Marketing KPIs** : Performance des campagnes
        6. **🔮 M6: Analyse Prédictive** : Modèle de prédiction de churn
        7. **📈 Dashboard Complet** : Vue d'ensemble interactive
        8. **📑 Rapport Final** : Synthèse complète exportable
        """)
        
        # État des fichiers
        st.subheader("📂 État des Fichiers")
        data_files = glob.glob('data/*.csv')
        output_files = glob.glob('output/*.csv')
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Données brutes (data/)**")
            if data_files:
                for f in data_files:
                    st.success(f"✅ {os.path.basename(f)}")
            else:
                st.warning("Aucun fichier CSV trouvé")
        
        with col2:
            st.write("**Données traitées (output/)**")
            if output_files:
                for f in output_files:
                    st.success(f"✅ {os.path.basename(f)}")
            else:
                st.info("Aucun fichier traité")

    # Module Import
    elif selected_tab == "📤 Import de Données":
        st.header("📤 Import et Gestion des Données")
        
        # Upload multiple files
        uploaded_files = st.file_uploader(
            "Sélectionnez vos fichiers CSV",
            type=['csv'],
            accept_multiple_files=True,
            help="Vous pouvez uploader plusieurs fichiers CSV. Le système détectera automatiquement le type de données."
        )
        
        if uploaded_files:
            if st.button("📥 Traiter les Fichiers", type="primary"):
                with st.spinner("Traitement des fichiers en cours..."):
                    success = analytics.process_uploaded_files(uploaded_files)
                    if success:
                        st.balloons()
                        st.success("🎉 Tous les fichiers ont été uploadés avec succès!")
                        
                        # Aperçu des données
                        st.subheader("👀 Aperçu des Données")
                        datasets = analytics.load_data()
                        
                        for data_type, df in datasets.items():
                            with st.expander(f"📋 {data_type.title()} ({len(df)} lignes)"):
                                st.dataframe(df.head())
                                st.info(f"Colonnes détectées: {', '.join(df.columns)}")
        
        # Exemples de format attendu
        st.subheader("📋 Formats de Données Attendus")
        
        format_examples = {
            "👥 Clients": ["customer_id", "age", "city", "client_type", "signup_date"],
            "🛍️ Produits": ["product_id", "name", "category", "price"],
            "💰 Ventes": ["order_id", "customer_id", "product_id", "order_date", "quantity", "total_amount"],
            "📢 Marketing": ["date", "channel", "campaign", "impressions", "clicks", "conversions", "cost", "revenue"]
        }
        
        cols = st.columns(2)
        for i, (data_type, columns) in enumerate(format_examples.items()):
            with cols[i % 2]:
                st.write(f"**{data_type}**")
                for col in columns:
                    st.write(f"• {col}")

    # Module M2: Nettoyage
    elif selected_tab == "🧹 M2: Nettoyage":
        st.header("🧹 M2: Nettoyage et Standardisation des Données")
        
        if st.button("🚀 Lancer le Nettoyage", type="primary"):
            with st.spinner("Nettoyage des données en cours..."):
                # Charger les données brutes
                raw_datasets = analytics.load_data()
                
                if not raw_datasets:
                    st.error("❌ Aucune donnée trouvée. Veuillez d'abord importer vos fichiers.")
                else:
                    # Nettoyage et standardisation
                    cleaned_datasets = analytics.clean_and_standardize_data(raw_datasets)
                    
                    if cleaned_datasets:
                        st.success("✅ Nettoyage terminé avec succès!")
                        
                        # Affichage des résultats
                        st.subheader("📊 Résultats du Nettoyage")
                        
                        for data_type, df in cleaned_datasets.items():
                            with st.expander(f"📋 {data_type.title()} - Données Nettoyées"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Lignes", len(df))
                                with col2:
                                    st.metric("Colonnes", len(df.columns))
                                
                                st.dataframe(df.head())
                                
                                # Statistiques de qualité
                                st.write("**Qualité des Données**")
                                quality_info = []
                                for col in df.columns:
                                    null_pct = (df[col].isnull().sum() / len(df)) * 100
                                    quality_info.append({
                                        "Colonne": col,
                                        "Valeurs Nulles (%)": f"{null_pct:.1f}%",
                                        "Type": str(df[col].dtype)
                                    })
                                
                                st.dataframe(pd.DataFrame(quality_info), use_container_width=True)
                    else:
                        st.warning("⚠️ Aucune donnée n'a pu être nettoyée.")
        
        # Affichage des fichiers nettoyés existants
        cleaned_files = glob.glob('output/*_clean.csv')
        if cleaned_files:
            st.subheader("📂 Fichiers Nettoyés Disponibles")
            for file_path in cleaned_files:
                filename = os.path.basename(file_path)
                with st.expander(f"📄 {filename}"):
                    try:
                        df = pd.read_csv(file_path)
                        st.dataframe(df.head())
                        st.download_button(
                            f"⬇️ Télécharger {filename}",
                            df.to_csv(index=False),
                            filename,
                            "text/csv"
                        )
                    except Exception as e:
                        st.error(f"Erreur lors de la lecture de {filename}: {str(e)}")

    # Module M3: Segmentation
    elif selected_tab == "👥 M3: Segmentation":
        st.header("👥 M3: Segmentation Clients (RFM Analysis)")
        
        if st.button("🎯 Lancer la Segmentation", type="primary"):
            with st.spinner("Analyse RFM en cours..."):
                try:
                    customers_data = pd.read_csv('output/customers_clean.csv')
                    sales_data = pd.read_csv('output/sales_clean.csv')
                    
                    segments_data, silhouette_score, optimal_k = analytics.perform_customer_segmentation(customers_data, sales_data)
                    
                    if segments_data is not None:
                        st.success(f"✅ Segmentation terminée! {optimal_k} segments identifiés (Score Silhouette: {silhouette_score:.3f})")
                        
                        # Métriques de segmentation
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Segments", optimal_k)
                        with col2:
                            st.metric("Score Silhouette", f"{silhouette_score:.3f}")
                        with col3:
                            st.metric("Clients Segmentés", len(segments_data))
                        with col4:
                            st.metric("Features Utilisées", "RFM + Âge")
                        
                        # Visualisations
                        st.subheader("📊 Visualisations des Segments")
                        
                        # Distribution des segments
                        fig_dist = px.pie(
                            values=segments_data['cluster'].value_counts().values,
                            names=[f'Segment {i}' for i in segments_data['cluster'].value_counts().index],
                            title="Distribution des Segments Clients"
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                        
                        # Analyse RFM par segment
                        st.subheader("🔍 Analyse RFM par Segment")
                        segment_summary = segments_data.groupby('cluster').agg({
                            'recency': 'mean',
                            'frequency': 'mean',
                            'monetary': 'mean'
                        }).round(2)
                        
                        st.dataframe(segment_summary, use_container_width=True)
                        
                        # Scatter plot RFM
                        fig_scatter = px.scatter_3d(
                            segments_data,
                            x='frequency',
                            y='monetary',
                            z='recency',
                            color='cluster',
                            title="Analyse RFM 3D par Segment",
                            labels={
                                'frequency': 'Fréquence',
                                'monetary': 'Valeur Monétaire (Ar)',
                                'recency': 'Récence (jours)'
                            }
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        # Matrice de corrélation
                        st.subheader("🔗 Matrice de Corrélation des Variables RFM")
                        corr_matrix = segments_data[['recency', 'frequency', 'monetary']].corr()
                        fig_corr = px.imshow(
                            corr_matrix,
                            title="Matrice de Corrélation RFM",
                            color_continuous_scale="RdBu"
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Tableau détaillé
                        st.subheader("📋 Données de Segmentation")
                        st.dataframe(segments_data, use_container_width=True)
                        
                        # Téléchargement
                        csv_data = segments_data.to_csv(index=False)
                        st.download_button(
                            "⬇️ Télécharger les Segments",
                            csv_data,
                            "customer_segments.csv",
                            "text/csv"
                        )
                    
                except FileNotFoundError:
                    st.error("❌ Fichiers de données manquants. Veuillez d'abord exécuter M2 (Nettoyage).")
                except Exception as e:
                    st.error(f"❌ Erreur lors de la segmentation: {str(e)}")

    # Module M4: Personas
    elif selected_tab == "🎭 M4: Personas":
        st.header("🎭 M4: Création de Personas Clients")
        
        if st.button("👤 Générer les Personas", type="primary"):
            with st.spinner("Création des personas en cours..."):
                try:
                    customers_data = pd.read_csv('output/customers_clean.csv')
                    sales_data = pd.read_csv('output/sales_clean.csv')
                    segments_data = pd.read_csv('output/customer_segments.csv')
                    
                    # Chargement optionnel des produits
                    try:
                        products_data = pd.read_csv('output/products_clean.csv')
                    except:
                        products_data = pd.DataFrame()
                    
                    personas = analytics.create_customer_personas(segments_data, customers_data, sales_data, products_data)
                    
                    if personas:
                        st.success(f"✅ {len(personas)} personas créés avec succès!")
                        
                        # Vue d'ensemble des personas
                        st.subheader("👥 Vue d'Ensemble des Personas")
                        
                        personas_summary = []
                        for segment_name, persona_data in personas.items():
                            personas_summary.append({
                                "Segment": segment_name.replace('_', ' ').title(),
                                "Taille": persona_data['size'],
                                "% Base": f"{persona_data['size']/len(customers_data)*100:.1f}%",
                                "Valeur Moy.": f"{persona_data['avg_monetary']:,.0f} Ar",
                                "Fréquence": f"{persona_data['avg_frequency']:.1f}",
                                "CA Total": f"{persona_data['total_revenue']:,.0f} Ar"
                            })
                        
                        summary_df = pd.DataFrame(personas_summary)
                        st.dataframe(summary_df, use_container_width=True)
                        
                        # Visualisations comparatives
                        st.subheader("📊 Comparaison des Personas")
                        
                        # Graphique en barres - Valeur par segment
                        fig_value = px.bar(
                            summary_df,
                            x="Segment",
                            y=[col for col in summary_df.columns if "Valeur" in col or "CA" in col],
                            title="Valeur Monétaire par Segment"
                        )
                        st.plotly_chart(fig_value, use_container_width=True)
                        
                        # Graphique radar pour comparaison multidimensionnelle
                        if len(personas) > 1:
                            radar_data = []
                            for segment_name, persona_data in personas.items():
                                radar_data.append({
                                    'Segment': segment_name.replace('_', ' ').title(),
                                    'Taille_Normalisée': persona_data['size'] / max(p['size'] for p in personas.values()),
                                    'Fréquence_Normalisée': persona_data['avg_frequency'] / max(p['avg_frequency'] for p in personas.values()),
                                    'Valeur_Normalisée': persona_data['avg_monetary'] / max(p['avg_monetary'] for p in personas.values()),
                                    'Récence_Normalisée': 1 - (persona_data['avg_recency'] / max(p['avg_recency'] for p in personas.values()))
                                })
                            
                            radar_df = pd.DataFrame(radar_data)
                            
                            fig_radar = go.Figure()
                            
                            for _, row in radar_df.iterrows():
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=[row['Taille_Normalisée'], row['Fréquence_Normalisée'], 
                                       row['Valeur_Normalisée'], row['Récence_Normalisée']],
                                    theta=['Taille', 'Fréquence', 'Valeur', 'Récence'],
                                    fill='toself',
                                    name=row['Segment']
                                ))
                            
                            fig_radar.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 1]
                                    )),
                                title="Comparaison Multidimensionnelle des Personas"
                            )
                            
                            st.plotly_chart(fig_radar, use_container_width=True)
                        
                        # Profils détaillés
                        st.subheader("🔍 Profils Détaillés des Personas")
                        
                        for segment_name, persona_data in personas.items():
                            with st.expander(f"👤 {segment_name.replace('_', ' ').title()}", expanded=True):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Taille du Segment", f"{persona_data['size']} clients")
                                    st.metric("Âge Moyen", f"{persona_data.get('avg_age', 0):.0f} ans" if persona_data.get('avg_age') else "N/A")
                                
                                with col2:
                                    st.metric("Valeur Moyenne", f"{persona_data['avg_monetary']:,.0f} Ar")
                                    st.metric("Fréquence", f"{persona_data['avg_frequency']:.1f} commandes")
                                
                                with col3:
                                    st.metric("CA Total", f"{persona_data['total_revenue']:,.0f} Ar")
                                    st.metric("Ville Principale", persona_data.get('top_city', 'N/A'))
                                
                                # Catégories préférées
                                if persona_data.get('top_categories'):
                                    st.write("**🛍️ Catégories Préférées:**")
                                    categories_df = pd.DataFrame(
                                        list(persona_data['top_categories'].items()),
                                        columns=['Catégorie', 'Commandes']
                                    )
                                    st.dataframe(categories_df, use_container_width=True)
                                
                                # Recommandations marketing
                                st.write("**💡 Recommandations Marketing:**")
                                if "0" in segment_name or "High" in segment_name:
                                    st.info("🎯 Clients Premium: Offres exclusives, service VIP, produits haut de gamme")
                                elif "1" in segment_name or "Medium" in segment_name:
                                    st.info("📈 Clients Réguliers: Programmes fidélité, cross-selling, offres groupées")
                                else:
                                    st.info("🌱 Clients à Développer: Promotions attractives, onboarding, content marketing")
                        
                        # Sauvegarde des personas
                        personas_df = pd.DataFrame.from_dict(personas, orient='index')
                        personas_csv = personas_df.to_csv()
                        st.download_button(
                            "⬇️ Télécharger les Personas",
                            personas_csv,
                            "customer_personas.csv",
                            "text/csv"
                        )
                    
                except FileNotFoundError as e:
                    st.error(f"❌ Fichiers manquants: {str(e)}. Veuillez d'abord exécuter les modules précédents.")
                except Exception as e:
                    st.error(f"❌ Erreur lors de la création des personas: {str(e)}")

    # Module M5: Marketing KPIs  
    elif selected_tab == "📊 M5: Marketing KPIs":
        st.header("📊 M5: Analyse des KPIs Marketing")
        
        if st.button("📈 Calculer les KPIs", type="primary"):
            with st.spinner("Calcul des KPIs marketing..."):
                try:
                    marketing_data = pd.read_csv('output/marketing_clean.csv')
                    
                    kpis_df, kpi_summary, benchmarks = analytics.calculate_marketing_kpis(marketing_data)
                    
                    if kpis_df is not None:
                        st.success("✅ KPIs calculés avec succès!")
                        
                        # Métriques principales
                        st.subheader("🎯 KPIs Globaux")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Impressions", 
                                f"{kpi_summary['total_impressions']:,.0f}",
                                help="Nombre total d'impressions"
                            )
                        with col2:
                            st.metric(
                                "CTR", 
                                f"{kpi_summary['overall_ctr']*100:.2f}%",
                                delta=f"Obj: {analytics.targets['ctr_min']*100:.1f}%"
                            )
                        with col3:
                            st.metric(
                                "CPC", 
                                f"{kpi_summary['overall_cpc']:,.0f} Ar",
                                delta=f"Obj: ≤{analytics.targets['cpc_max']:.0f} Ar"
                            )
                        with col4:
                            st.metric(
                                "ROAS", 
                                f"{kpi_summary['overall_roas']:.2f}x",
                                delta=f"Obj: ≥{analytics.targets['roi_min']:.1f}x"
                            )
                        
                        # Tableau de bord des KPIs détaillés
                        st.subheader("📋 Tableau de Bord Détaillé")
                        
                        detailed_metrics = [
                            {"KPI": "Impressions", "Valeur": f"{kpi_summary['total_impressions']:,.0f}", "Description": "Nombre total de vues"},
                            {"KPI": "Clics", "Valeur": f"{kpi_summary['total_clicks']:,.0f}", "Description": "Nombre total de clics"},
                            {"KPI": "Conversions", "Valeur": f"{kpi_summary['total_conversions']:,.0f}", "Description": "Nombre total de conversions"},
                            {"KPI": "CTR", "Valeur": f"{kpi_summary['overall_ctr']*100:.2f}%", "Description": "Taux de clic"},
                            {"KPI": "CVR", "Valeur": f"{kpi_summary['overall_cvr']*100:.2f}%", "Description": "Taux de conversion"},
                            {"KPI": "CPC", "Valeur": f"{kpi_summary['overall_cpc']:,.0f} Ar", "Description": "Coût par clic"},
                            {"KPI": "CPA", "Valeur": f"{kpi_summary['overall_cpa']:,.0f} Ar", "Description": "Coût par acquisition"},
                            {"KPI": "ROAS", "Valeur": f"{kpi_summary['overall_roas']:.2f}x", "Description": "Retour sur investissement publicitaire"}
                        ]
                        
                        metrics_df = pd.DataFrame(detailed_metrics)
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        # Benchmarks vs Objectifs
                        st.subheader("🎯 Performance vs Objectifs")
                        
                        benchmarks_df = pd.DataFrame(benchmarks)
                        
                        # Coloration conditionnelle
                        def color_status(val):
                            if val == 'Atteint':
                                return 'background-color: #d4edda'
                            else:
                                return 'background-color: #f8d7da'
                        
                        styled_benchmarks = benchmarks_df.style.applymap(color_status, subset=['Statut'])
                        st.dataframe(styled_benchmarks, use_container_width=True)
                        
                        # Visualisations des KPIs
                        st.subheader("📊 Visualisations des Performances")
                        
                        # Évolution temporelle des KPIs
                        if 'date' in kpis_df.columns:
                            kpis_df['date'] = pd.to_datetime(kpis_df['date'])
                            
                            # Graphique temporel multi-KPIs
                            fig_timeline = make_subplots(
                                rows=2, cols=2,
                                subplot_titles=('CTR (%)', 'CPC (Ar)', 'Conversions', 'ROAS'),
                                vertical_spacing=0.1
                            )
                            
                            # CTR
                            fig_timeline.add_trace(
                                go.Scatter(x=kpis_df['date'], y=kpis_df['ctr']*100, name='CTR'),
                                row=1, col=1
                            )
                            
                            # CPC
                            fig_timeline.add_trace(
                                go.Scatter(x=kpis_df['date'], y=kpis_df['cpc'], name='CPC'),
                                row=1, col=2
                            )
                            
                            # Conversions
                            fig_timeline.add_trace(
                                go.Scatter(x=kpis_df['date'], y=kpis_df['conversions'], name='Conversions'),
                                row=2, col=1
                            )
                            
                            # ROAS
                            if 'roas' in kpis_df.columns:
                                fig_timeline.add_trace(
                                    go.Scatter(x=kpis_df['date'], y=kpis_df['roas'], name='ROAS'),
                                    row=2, col=2
                                )
                            
                            fig_timeline.update_layout(height=600, title_text="Évolution des KPIs Marketing")
                            st.plotly_chart(fig_timeline, use_container_width=True)
                        
                        # Performance par canal
                        if 'channel' in kpis_df.columns:
                            channel_perf = kpis_df.groupby('channel').agg({
                                'impressions': 'sum',
                                'clicks': 'sum',
                                'conversions': 'sum',
                                'cost': 'sum',
                                'revenue': 'sum' if 'revenue' in kpis_df.columns else lambda x: 0
                            }).reset_index()
                            
                            channel_perf['ctr'] = channel_perf['clicks'] / channel_perf['impressions']
                            channel_perf['cpc'] = channel_perf['cost'] / channel_perf['clicks']
                            channel_perf['roas'] = channel_perf['revenue'] / channel_perf['cost']
                            
                            fig_channels = px.bar(
                                channel_perf,
                                x='channel',
                                y=['impressions', 'clicks', 'conversions'],
                                title='Performance par Canal Marketing',
                                barmode='group'
                            )
                            st.plotly_chart(fig_channels, use_container_width=True)
                        
                        # Matrice de corrélation des KPIs
                        st.subheader("🔗 Corrélations entre KPIs")
                        numeric_kpis = kpis_df.select_dtypes(include=[np.number])
                        if len(numeric_kpis.columns) > 1:
                            corr_matrix = numeric_kpis.corr()
                            
                            fig_corr = px.imshow(
                                corr_matrix,
                                title="Matrice de Corrélation des KPIs",
                                color_continuous_scale="RdBu",
                                aspect="auto"
                            )
                            st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Recommandations automatiques
                        st.subheader("💡 Recommandations Automatiques")
                        
                        recommendations = []
                        
                        if kpi_summary['overall_ctr'] < analytics.targets['ctr_min']:
                            recommendations.append("🎯 **CTR faible**: Optimiser les créatifs publicitaires et le ciblage")
                        
                        if kpi_summary['overall_cpc'] > analytics.targets['cpc_max']:
                            recommendations.append("💰 **CPC élevé**: Revoir la stratégie d'enchères et affiner le ciblage")
                        
                        if kpi_summary['overall_roas'] < analytics.targets['roi_min']:
                            recommendations.append("📈 **ROAS insuffisant**: Analyser le funnel de conversion et optimiser les landing pages")
                        
                        if kpi_summary['overall_cvr'] < analytics.targets['conversion_rate_target']:
                            recommendations.append("🔄 **Taux de conversion faible**: Améliorer l'expérience utilisateur et les call-to-action")
                        
                        if not recommendations:
                            st.success("🎉 Excellente performance! Tous les objectifs sont atteints.")
                        else:
                            for rec in recommendations:
                                st.warning(rec)
                        
                        # Téléchargements
                        st.subheader("⬇️ Téléchargements")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            kpis_csv = kpis_df.to_csv(index=False)
                            st.download_button(
                                "📊 Télécharger KPIs Détaillés",
                                kpis_csv,
                                "campaign_kpis.csv",
                                "text/csv"
                            )
                        
                        with col2:
                            benchmarks_csv = benchmarks_df.to_csv(index=False)
                            st.download_button(
                                "🎯 Télécharger Benchmarks",
                                benchmarks_csv,
                                "kpi_benchmarks.csv",
                                "text/csv"
                            )
                    
                except FileNotFoundError:
                    st.error("❌ Fichier marketing_clean.csv manquant. Veuillez d'abord exécuter M2 (Nettoyage).")
                except Exception as e:
                    st.error(f"❌ Erreur lors du calcul des KPIs: {str(e)}")

    # Module M6: Analyse Prédictive
    elif selected_tab == "🔮 M6: Analyse Prédictive":
        st.header("🔮 M6: Modèle Prédictif de Churn")
        
        if st.button("🤖 Entraîner le Modèle", type="primary"):
            with st.spinner("Entraînement du modèle prédictif..."):
                try:
                    customers_data = pd.read_csv('output/customers_clean.csv')
                    sales_data = pd.read_csv('output/sales_clean.csv')
                    
                    churn_data, feature_importance, auc_score = analytics.build_predictive_model(customers_data, sales_data)
                    
                    if churn_data is not None:
                        st.success(f"✅ Modèle entraîné avec succès! Score AUC: {auc_score:.3f}")
                        
                        # Métriques du modèle
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Score AUC", f"{auc_score:.3f}")
                        with col2:
                            st.metric("Clients Analysés", len(churn_data))
                        with col3:
                            churn_rate = (churn_data['is_churned'].sum() / len(churn_data)) * 100
                            st.metric("Taux de Churn", f"{churn_rate:.1f}%")
                        with col4:
                            high_risk = len(churn_data[churn_data['churn_risk'] == 'Élevé'])
                            st.metric("Clients à Risque", high_risk)
                        
                        # Distribution du risque de churn
                        st.subheader("📊 Distribution du Risque de Churn")
                        
                        risk_counts = churn_data['churn_risk'].value_counts()
                        fig_risk = px.pie(
                            values=risk_counts.values,
                            names=risk_counts.index,
                            title="Répartition des Clients par Niveau de Risque",
                            color_discrete_map={
                                'Faible': '#28a745',
                                'Moyen': '#ffc107', 
                                'Élevé': '#dc3545'
                            }
                        )
                        st.plotly_chart(fig_risk, use_container_width=True)
                        
                        # Histogramme des probabilités de churn
                        fig_hist = px.histogram(
                            churn_data,
                            x='churn_probability',
                            nbins=20,
                            title="Distribution des Probabilités de Churn",
                            labels={'churn_probability': 'Probabilité de Churn', 'count': 'Nombre de Clients'}
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Importance des variables
                        st.subheader("🎯 Importance des Variables")
                        
                        fig_importance = px.bar(
                            feature_importance,
                            x='importance',
                            y='feature',
                            orientation='h',
                            title="Importance des Variables dans la Prédiction de Churn"
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                        
                        # Analyse par segment de risque
                        st.subheader("🔍 Analyse par Segment de Risque")
                        
                        for risk_level in ['Élevé', 'Moyen', 'Faible']:
                            risk_data = churn_data[churn_data['churn_risk'] == risk_level]
                            
                            if len(risk_data) > 0:
                                with st.expander(f"🎯 Clients à Risque {risk_level} ({len(risk_data)} clients)", expanded=(risk_level == 'Élevé')):
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Récence Moyenne", f"{risk_data['recency'].mean():.0f} jours")
                                        st.metric("Probabilité Moyenne", f"{risk_data['churn_probability'].mean():.1%}")
                                    
                                    with col2:
                                        st.metric("Fréquence Moyenne", f"{risk_data['frequency'].mean():.1f}")
                                        st.metric("Dépense Totale Moy.", f"{risk_data['total_spent'].mean():,.0f} Ar")
                                    
                                    with col3:
                                        st.metric("Panier Moyen", f"{risk_data['avg_order_value'].mean():,.0f} Ar")
                                        if 'age' in risk_data.columns:
                                            st.metric("Âge Moyen", f"{risk_data['age'].mean():.0f} ans")
                                    
                                    # Recommandations par niveau de risque
                                    st.write("**💡 Recommandations:**")
                                    if risk_level == 'Élevé':
                                        st.error("🚨 **Actions Urgentes**: Contact immédiat, offres exclusives, support personnalisé")
                                    elif risk_level == 'Moyen':
                                        st.warning("⚠️ **Surveillance Active**: Campagnes de rétention, programmes fidélité, enquêtes satisfaction")
                                    else:
                                        st.success("✅ **Maintien de l'Engagement**: Communication régulière, programmes d'ambassadeurs, cross-selling")
                                    
                                    # Top clients à risque pour le niveau élevé
                                    if risk_level == 'Élevé' and len(risk_data) > 0:
                                        st.write("**👥 Clients Prioritaires:**")
                                        priority_clients = risk_data.nlargest(10, 'total_spent')[['customer_id', 'churn_probability', 'total_spent', 'recency']]
                                        priority_clients['churn_probability'] = priority_clients['churn_probability'].apply(lambda x: f"{x:.1%}")
                                        priority_clients['total_spent'] = priority_clients['total_spent'].apply(lambda x: f"{x:,.0f} Ar")
                                        st.dataframe(priority_clients, use_container_width=True)
                        
                        # Matrice de confusion visuelle (si applicable)
                        if auc_score > 0:
                            st.subheader("📈 Performance du Modèle")
                            
                            # Seuils de probabilité pour analyse
                            thresholds = [0.3, 0.5, 0.7]
                            threshold_analysis = []
                            
                            for threshold in thresholds:
                                predicted_churn = (churn_data['churn_probability'] >= threshold).astype(int)
                                actual_churn = churn_data['is_churned']
                                
                                tp = ((predicted_churn == 1) & (actual_churn == 1)).sum()
                                fp = ((predicted_churn == 1) & (actual_churn == 0)).sum()
                                tn = ((predicted_churn == 0) & (actual_churn == 0)).sum()
                                fn = ((predicted_churn == 0) & (actual_churn == 1)).sum()
                                
                                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                                accuracy = (tp + tn) / len(churn_data)
                                
                                threshold_analysis.append({
                                    'Seuil': threshold,
                                    'Précision': f"{precision:.1%}",
                                    'Rappel': f"{recall:.1%}",
                                    'Exactitude': f"{accuracy:.1%}",
                                    'Prédictions Positives': tp + fp
                                })
                            
                            threshold_df = pd.DataFrame(threshold_analysis)
                            st.dataframe(threshold_df, use_container_width=True)
                        
                        # Plans d'action par segment
                        st.subheader("📋 Plans d'Action Recommandés")
                        
                        action_plans = {
                            "🚨 Risque Élevé": [
                                "Appel téléphonique dans les 24h",
                                "Offre de réactivation exclusive (-30%)",
                                "Invitation à un événement VIP",
                                "Enquête de satisfaction personnalisée",
                                "Attribution d'un account manager dédié"
                            ],
                            "⚠️ Risque Moyen": [
                                "Email de rétention personnalisé",
                                "Recommandations de produits basées sur l'historique",
                                "Invitation au programme de fidélité",
                                "Offre de bundling produits",
                                "Newsletter avec contenu exclusif"
                            ],
                            "✅ Risque Faible": [
                                "Communication marketing standard",
                                "Sollicitation pour avis/témoignages",
                                "Invitation programme parrainage",
                                "Cross-selling intelligent",
                                "Enquête de satisfaction trimestrielle"
                            ]
                        }
                        
                        for risk_category, actions in action_plans.items():
                            with st.expander(f"{risk_category} - Plan d'Action"):
                                for i, action in enumerate(actions, 1):
                                    st.write(f"{i}. {action}")
                        
                        # Téléchargements
                        st.subheader("⬇️ Exports")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            churn_csv = churn_data.to_csv(index=False)
                            st.download_button(
                                "📊 Télécharger Prédictions",
                                churn_csv,
                                "churn_predictions.csv",
                                "text/csv"
                            )
                        
                        with col2:
                            importance_csv = feature_importance.to_csv(index=False)
                            st.download_button(
                                "🎯 Télécharger Importance Variables",
                                importance_csv,
                                "feature_importance.csv",
                                "text/csv"
                            )
                    
                except FileNotFoundError:
                    st.error("❌ Fichiers de données manquants. Veuillez d'abord exécuter les modules précédents.")
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'analyse prédictive: {str(e)}")
        
        # --- Section: Prédiction d'objectifs commerciaux (M6 UI) ---
        st.markdown("---")
        render_sales_objective_prediction_section()

        # --- Section: Entraîner le modèle Objectif Commercial (à partir d'un CSV importé) ---
        st.markdown("---")
        with st.expander("⚙️ Entraîner le Modèle d'Objectif Commercial (à partir d'un fichier importé)"):
            st.caption(
                "Le fichier CSV doit contenir les colonnes: "
                + ", ".join(EXPECTED_FEATURES)
                + f" et la cible binaire '{TARGET_NAME}'"
            )

            uploaded_csv = st.file_uploader("Importer un CSV d'entraînement", type=["csv"], key="m6_obj_train_csv")
            manual_path = st.text_input("ou Chemin d'un CSV existant", value="")
            output_model_path = st.text_input(
                "Chemin de sortie du modèle (.pkl)", value="models/modele_objectif_commercial.pkl"
            )

            if st.button("🚀 Entraîner et Sauvegarder le Modèle", type="primary"):
                try:
                    data_path = None
                    if uploaded_csv is not None:
                        os.makedirs("output", exist_ok=True)
                        tmp_path = os.path.join("output", "objective_training.csv")
                        with open(tmp_path, "wb") as f:
                            f.write(uploaded_csv.getbuffer())
                        data_path = tmp_path
                    elif manual_path:
                        data_path = manual_path

                    if data_path is None:
                        st.error("Veuillez importer un CSV ou renseigner un chemin valide.")
                    else:
                        # Validation rapide des colonnes
                        df_check = pd.read_csv(data_path)
                        missing = [c for c in EXPECTED_FEATURES + [TARGET_NAME] if c not in df_check.columns]
                        if missing:
                            st.error(
                                f"Colonnes manquantes: {missing}. Colonnes attendues: {EXPECTED_FEATURES + [TARGET_NAME]}"
                            )
                        else:
                            if train_objective_model is None:
                                st.error("La fonction d'entraînement n'est pas disponible. Redémarrez l'app et réessayez.")
                            else:
                                path = train_objective_model(data_path, output_model_path)
                                st.success(f"✅ Modèle entraîné et sauvegardé: {path}")
                                st.info("Revenez à la section de prédiction ci-dessus pour l'utiliser.")
                except Exception as e:
                    st.error(f"❌ Échec de l'entraînement: {str(e)}")

    # Module Dashboard Complet
    elif selected_tab == "📈 Dashboard Complet":
        st.header("📈 Dashboard Analytics Complet")
        
        try:
            # Chargement de tous les fichiers disponibles (par défaut depuis output/)
            customers_data = pd.read_csv('output/customers_clean.csv') if os.path.exists('output/customers_clean.csv') else pd.DataFrame()
            sales_data = pd.read_csv('output/sales_clean.csv') if os.path.exists('output/sales_clean.csv') else pd.DataFrame()
            marketing_data = pd.read_csv('output/marketing_clean.csv') if os.path.exists('output/marketing_clean.csv') else pd.DataFrame()
            segments_data = pd.read_csv('output/customer_segments.csv') if os.path.exists('output/customer_segments.csv') else pd.DataFrame()

            # Option: utiliser des fichiers importés directement pour ce dashboard
            with st.expander("📥 Importer des CSV pour ce Dashboard (remplace les données par défaut)"):
                uploaded_files = st.file_uploader(
                    "Importer des fichiers CSV", type=["csv"], accept_multiple_files=True, key="dash_manual_csvs"
                )
                if uploaded_files:
                    if analytics.process_uploaded_files(uploaded_files):
                        datasets = analytics.load_data()
                        cleaned = analytics.clean_and_standardize_data(datasets)
                        # Remplacer les DataFrames si disponibles
                        if 'customers' in cleaned:
                            customers_data = cleaned['customers']
                        if 'sales' in cleaned:
                            sales_data = cleaned['sales']
                        if 'marketing' in cleaned:
                            marketing_data = cleaned['marketing']
                        # segments_data: généré par M3, pas directement depuis uploads
                        st.success("Les données importées ont été appliquées au dashboard.")
                        st.caption(
                            f"Clients: {0 if customers_data is None or customers_data.empty else len(customers_data)} | "
                            f"Ventes: {0 if sales_data is None or sales_data.empty else len(sales_data)} | "
                            f"Marketing: {0 if marketing_data is None or marketing_data.empty else len(marketing_data)}"
                        )
            
            if customers_data.empty and sales_data.empty:
                st.warning("📤 Aucune donnée disponible. Veuillez d'abord importer et traiter vos données.")
                st.stop()
            
            # KPIs Généraux
            st.subheader("🎯 KPIs Généraux")
            # Paramètre: prix unitaire par défaut si non fourni dans ventes
            default_unit_price = st.number_input(
                "💵 Prix unitaire par défaut (Ar) si absent dans les ventes",
                min_value=0, max_value=10_000_000, value=30_000, step=1_000,
                help="Utilisé pour calculer total_amount = unit_price * quantity quand unit_price/total_amount sont manquants."
            )
            # Persister la valeur de référence
            try:
                st.session_state['default_unit_price'] = float(default_unit_price)
            except Exception:
                st.session_state['default_unit_price'] = 0.0

            # Construire et persister un sales_processed unique pour tout le Dashboard
            has_sales = sales_data is not None and not sales_data.empty
            sales_processed = sales_data.copy() if has_sales else pd.DataFrame()
            if not sales_processed.empty:
                unit_cols = [c for c in sales_processed.columns if c in ['unit_price', 'price', 'prix']]
                qty_cols = [c for c in sales_processed.columns if c in ['quantity', 'qty', 'quantite', 'quantité']]
                qt = pd.Series(0, index=sales_processed.index)
                if qty_cols:
                    qt = pd.to_numeric(sales_processed[qty_cols[0]].astype(str).str.replace(r"[^0-9,.-]", "", regex=True).str.replace(',', '.'), errors='coerce').fillna(0)
                if unit_cols:
                    up = pd.to_numeric(sales_processed[unit_cols[0]].astype(str).str.replace(r"[^0-9,.-]", "", regex=True).str.replace(',', '.'), errors='coerce').fillna(float(default_unit_price))
                else:
                    up = pd.Series(float(default_unit_price), index=sales_processed.index)
                if 'total_amount' not in sales_processed.columns:
                    sales_processed['total_amount'] = (up * qt)
                else:
                    ta_clean = pd.to_numeric(
                        sales_processed['total_amount'].astype(str).str.replace(r"[^0-9,.-]", "", regex=True).str.replace(',', '.'),
                        errors='coerce'
                    )
                    sales_processed['total_amount'] = ta_clean.where(ta_clean.notna() & (ta_clean > 0), (up * qt))
                # Dates
                if 'order_date' in sales_processed.columns:
                    sales_processed['order_date'] = pd.to_datetime(sales_processed['order_date'], errors='coerce')
                st.session_state['sales_processed'] = sales_processed.copy()
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                total_customers = len(customers_data) if not customers_data.empty else 0
                st.metric("👥 Clients Total", f"{total_customers:,}")
            
            with col2:
                total_revenue = 0.0
                has_sales = sales_data is not None and not sales_data.empty
                if has_sales:
                    sd = st.session_state.get('sales_processed', sales_data).copy()
                    unit_cols = [c for c in sd.columns if c in ['unit_price', 'price', 'prix']]
                    qty_cols = [c for c in sd.columns if c in ['quantity', 'qty', 'quantite', 'quantité']]
                    # Préparer qt et up
                    qt = pd.Series(0, index=sd.index)
                    if qty_cols:
                        qt = pd.to_numeric(sd[qty_cols[0]].astype(str).str.replace(r"[^0-9,.-]", "", regex=True).str.replace(',', '.'), errors='coerce').fillna(0)
                    if unit_cols:
                        up = pd.to_numeric(sd[unit_cols[0]].astype(str).str.replace(r"[^0-9,.-]", "", regex=True).str.replace(',', '.'), errors='coerce').fillna(float(default_unit_price))
                    else:
                        up = pd.Series(float(default_unit_price), index=sd.index)
                    # Construire/Nettoyer total_amount
                    if 'total_amount' not in sd.columns:
                        sd['total_amount'] = (up * qt)
                    else:
                        ta_clean = pd.to_numeric(
                            sd['total_amount'].astype(str).str.replace(r"[^0-9,.-]", "", regex=True).str.replace(',', '.'),
                            errors='coerce'
                        )
                        fill_series = (up * qt)
                        sd['total_amount'] = ta_clean.where(ta_clean.notna() & (ta_clean > 0), fill_series)
                    # Série finale propre
                    ta_series = pd.to_numeric(sd['total_amount'], errors='coerce').fillna(0)
                    total_revenue = float(ta_series.sum())
                st.metric("💰 CA Total", f"{total_revenue:,.0f} Ar")
            
            with col3:
                has_sales = sales_data is not None and not sales_data.empty
                if has_sales:
                    sd = st.session_state.get('sales_processed', sales_data).copy()
                    unit_cols = [c for c in sd.columns if c in ['unit_price', 'price', 'prix']]
                    qty_cols = [c for c in sd.columns if c in ['quantity', 'qty', 'quantite', 'quantité']]
                    qt = pd.Series(0, index=sd.index)
                    if qty_cols:
                        qt = pd.to_numeric(sd[qty_cols[0]].astype(str).str.replace(r"[^0-9,.-]", "", regex=True).str.replace(',', '.'), errors='coerce').fillna(0)
                    if unit_cols:
                        up = pd.to_numeric(sd[unit_cols[0]].astype(str).str.replace(r"[^0-9,.-]", "", regex=True).str.replace(',', '.'), errors='coerce').fillna(float(default_unit_price))
                    else:
                        up = pd.Series(float(default_unit_price), index=sd.index)
                    if 'total_amount' not in sd.columns:
                        sd['total_amount'] = (up * qt)
                    else:
                        ta_clean = pd.to_numeric(
                            sd['total_amount'].astype(str).str.replace(r"[^0-9,.-]", "", regex=True).str.replace(',', '.'),
                            errors='coerce'
                        )
                        fill_series = (up * qt)
                        sd['total_amount'] = ta_clean.where(ta_clean.notna() & (ta_clean > 0), fill_series)
                    total_amount_series = pd.to_numeric(sd['total_amount'], errors='coerce').fillna(0)
                    avg_order = total_amount_series.mean()
                    if pd.isna(avg_order):
                        avg_order = 0
                    st.metric("🛒 Panier Moyen", f"{avg_order:,.0f} Ar")
                else:
                    st.metric("🛒 Panier Moyen", "N/A")
                    st.caption("Ventes indisponibles ou colonne 'total_amount' manquante")
            
            with col4:
                has_sales = sales_data is not None and not sales_data.empty
                if has_sales and 'order_id' in sales_data.columns:
                    total_orders = sales_data['order_id'].nunique()
                else:
                    total_orders = len(sales_data) if has_sales else 0
                st.metric("📦 Commandes", f"{total_orders:,}")
            
            with col5:
                if not marketing_data.empty and 'cost' in marketing_data.columns:
                    total_marketing_cost = marketing_data['cost'].sum()
                    st.metric("📢 Coût Marketing", f"{total_marketing_cost:,.0f} Ar")
                else:
                    st.metric("📢 Coût Marketing", "N/A")
            
            with col6:
                if marketing_data.empty:
                    st.metric("📈 ROI Marketing", "N/A")
                    st.caption("Données marketing indisponibles")
                elif 'cost' not in marketing_data.columns:
                    st.metric("📈 ROI Marketing", "N/A")
                    st.caption("Colonne 'cost' manquante dans marketing")
                else:
                    # Nettoyage numérique du cost
                    cost_series = pd.to_numeric(
                        marketing_data['cost'].astype(str).str.replace(r"[^0-9,.-]", "", regex=True).str.replace(',', '.'),
                        errors='coerce'
                    ).fillna(0)
                    total_cost = float(cost_series.sum())
                    # Source de revenus pour ROI: priorité à marketing.revenue si disponible sinon CA ventes
                    if 'revenue' in marketing_data.columns:
                        rev_series = pd.to_numeric(
                            marketing_data['revenue'].astype(str).str.replace(r"[^0-9,.-]", "", regex=True).str.replace(',', '.'),
                            errors='coerce'
                        ).fillna(0)
                        roi_revenue = float(rev_series.sum())
                    else:
                        roi_revenue = float(total_revenue)
                    if total_cost == 0 and roi_revenue > 0:
                        st.metric("📈 ROI Marketing", "∞")
                        st.caption("Coût marketing nul, CA > 0")
                    elif total_cost == 0 and roi_revenue == 0:
                        st.metric("📈 ROI Marketing", "N/A")
                        st.caption("Coût et CA nuls")
                    elif roi_revenue == 0:
                        st.metric("📈 ROI Marketing", "0.0x")
                        st.caption("CA = 0")
                    else:
                        marketing_roi = roi_revenue / total_cost
                        st.metric("📈 ROI Marketing", f"{marketing_roi:.1f}x")
            
            st.markdown("---")
            
            # Section 1: Analyse des Ventes
            has_sales = sales_data is not None and not sales_data.empty
            if has_sales:
                st.subheader("📊 Analyse des Ventes")
                
                col1, col2 = st.columns(2)
                
                with st.expander("🧪 Diagnostics KPIs (CA & Panier)", expanded=False):
                    try:
                        dbg = sales_data.copy()
                        unit_cols = [c for c in dbg.columns if c in ['unit_price', 'price', 'prix']]
                        qty_cols = [c for c in dbg.columns if c in ['quantity', 'qty', 'quantite', 'quantité']]
                        qty_series = pd.to_numeric(dbg[qty_cols[0]], errors='coerce') if qty_cols else pd.Series([], dtype=float)
                        st.caption(f"Lignes ventes: {len(dbg)} | Colonne quantité: {qty_cols[0] if qty_cols else '—'} | Somme quantités: {float(qty_series.fillna(0).sum()) if not qty_series.empty else 0}")
                        st.caption(f"Prix unitaire par défaut: {float(default_unit_price):,.0f} Ar")
                        # Reconstitution rapide
                        if unit_cols:
                            up = pd.to_numeric(dbg[unit_cols[0]].astype(str).str.replace(r"[^0-9,.-]","", regex=True).str.replace(',','.'), errors='coerce').fillna(float(default_unit_price))
                        else:
                            up = pd.Series(float(default_unit_price), index=dbg.index)
                        qt = pd.to_numeric(dbg[qty_cols[0]].astype(str).str.replace(r"[^0-9,.-]","", regex=True).str.replace(',','.'), errors='coerce').fillna(0) if qty_cols else pd.Series(0, index=dbg.index)
                        if 'total_amount' in dbg.columns:
                            ta_clean = pd.to_numeric(dbg['total_amount'].astype(str).str.replace(r"[^0-9,.-]","", regex=True).str.replace(',','.'), errors='coerce')
                            dbg['__ta'] = ta_clean.where(ta_clean.notna() & (ta_clean > 0), up*qt)
                        else:
                            dbg['__ta'] = up*qt
                        st.caption(f"Somme total_amount (après nettoyage/reconstruction): {float(pd.to_numeric(dbg['__ta'], errors='coerce').fillna(0).sum()):,.0f} Ar")
                        st.dataframe(dbg[['customer_id', qty_cols[0] if qty_cols else None, unit_cols[0] if unit_cols else None, '__ta']].head(5))
                    except Exception as e:
                        st.caption(f"Diagnostics indisponible: {str(e)}")

                has_sales = not sales_data.empty if sales_data is not None else False
                if has_sales:
                    # Utiliser sales_processed si disponible, sinon utiliser sales_data
                    sp = st.session_state.get('sales_processed', sales_data)
                    if 'order_date' in sp.columns and not sp.empty:
                        daily_sales = sp.groupby(sp['order_date'].dt.date)['total_amount'].sum().reset_index()
                        
                        fig_sales_trend = px.line(
                            daily_sales,
                            x='order_date',
                            y='total_amount',
                            title='Évolution du Chiffre d\'Affaires'
                        )
                        st.plotly_chart(fig_sales_trend, use_container_width=True)
                
                with col2:
                    # Top produits
                    if 'product_id' in sales_data.columns:
                        top_products = sales_data.groupby('product_id')['total_amount'].sum().nlargest(10)
                        
                        fig_top_products = px.bar(
                            x=top_products.values,
                            y=top_products.index,
                            orientation='h',
                            title='Top 10 Produits par CA'
                        )
                        st.plotly_chart(fig_top_products, use_container_width=True)
            
            # Section 2: Segmentation Clients
            if not segments_data.empty:
                st.subheader("👥 Segmentation Clients")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution des segments
                    segment_counts = segments_data['cluster'].value_counts()
                    fig_segments = px.pie(
                        values=segment_counts.values,
                        names=[f'Segment {i}' for i in segment_counts.index],
                        title='Distribution des Segments'
                    )
                    st.plotly_chart(fig_segments, use_container_width=True)
                
                with col2:
                    # RFM par segment
                    rfm_by_segment = segments_data.groupby('cluster')[['recency', 'frequency', 'monetary']].mean()
                    
                    fig_rfm = px.bar(
                        rfm_by_segment.reset_index(),
                        x='cluster',
                        y=['recency', 'frequency', 'monetary'],
                        title='Profil RFM par Segment',
                        barmode='group'
                    )
                    st.plotly_chart(fig_rfm, use_container_width=True)
            
            # Section 3: Performance Marketing
            if not marketing_data.empty:
                st.subheader("📢 Performance Marketing")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # KPIs Marketing par période
                    if 'date' in marketing_data.columns:
                        marketing_data['date'] = pd.to_datetime(marketing_data['date'])
                        marketing_timeline = marketing_data.groupby(marketing_data['date'].dt.date).agg({
                            'impressions': 'sum',
                            'clicks': 'sum',
                            'cost': 'sum'
                        }).reset_index()
                        
                        marketing_timeline['ctr'] = marketing_timeline['clicks'] / marketing_timeline['impressions']
                        
                        fig_marketing = px.line(
                            marketing_timeline,
                            x='date',
                            y='ctr',
                            title='Évolution du CTR'
                        )
                        st.plotly_chart(fig_marketing, use_container_width=True)
                
                with col2:
                    # Performance par canal
                    if 'channel' in marketing_data.columns:
                        channel_perf = marketing_data.groupby('channel').agg({
                            'impressions': 'sum',
                            'clicks': 'sum',
                            'cost': 'sum'
                        }).reset_index()
                        
                        fig_channels = px.bar(
                            channel_perf,
                            x='channel',
                            y='clicks',
                            title='Clics par Canal'
                        )
                        st.plotly_chart(fig_channels, use_container_width=True)
            
            # Section 4: Analyse Géographique
            if not customers_data.empty and 'city' in customers_data.columns:
                st.subheader("🌍 Répartition Géographique")
                
                city_stats = customers_data['city'].value_counts().head(10)
                fig_geo = px.bar(
                    x=city_stats.values,
                    y=city_stats.index,
                    orientation='h',
                    title='Top 10 Villes'
                )
                st.plotly_chart(fig_geo, use_container_width=True)
            
            # Section 4b: Segmentation Démographique (Âge, Genre, Ville)
            if not customers_data.empty and not sales_data.empty:
                available_demo = [c for c in ['age', 'gender', 'city'] if c in customers_data.columns]
                if available_demo:
                    st.subheader("👨‍👩‍👧‍👦 Segmentation Démographique")
                    try:
                        has_sales = not sales_data.empty if sales_data is not None else False
                        if has_sales and customers_data is not None and not customers_data.empty:
                            # Fusion ventes + clients pour KPIs par segment
                            sp = st.session_state.get('sales_processed', sales_data)
                            sales_demo = sp.merge(
                                customers_data[['customer_id'] + available_demo], on='customer_id', how='left'
                            )
                        else:
                            raise ValueError("Données de vente ou clients manquantes")
                        # Reconstruire/Nettoyer total_amount avec le même algorithme que les KPIs
                        unit_cols = [c for c in sales_demo.columns if c in ['unit_price', 'price', 'prix']]
                        qty_cols = [c for c in sales_demo.columns if c in ['quantity', 'qty', 'quantite', 'quantité']]
                        qt = pd.Series(0, index=sales_demo.index)
                        if qty_cols:
                            qt = pd.to_numeric(sales_demo[qty_cols[0]].astype(str).str.replace(r"[^0-9,.-]", "", regex=True).str.replace(',', '.'), errors='coerce').fillna(0)
                        if unit_cols:
                            up = pd.to_numeric(sales_demo[unit_cols[0]].astype(str).str.replace(r"[^0-9,.-]", "", regex=True).str.replace(',', '.'), errors='coerce').fillna(float(default_unit_price))
                        else:
                            up = pd.Series(float(default_unit_price), index=sales_demo.index)
                        if 'total_amount' not in sales_demo.columns:
                            sales_demo['total_amount'] = (up * qt)
                        else:
                            ta_clean = pd.to_numeric(
                                sales_demo['total_amount'].astype(str).str.replace(r"[^0-9,.-]", "", regex=True).str.replace(',', '.'),
                                errors='coerce'
                            )
                            sales_demo['total_amount'] = ta_clean.where(ta_clean.notna() & (ta_clean > 0), (up * qt))
                        # Tranches d'âge
                        if 'age' in available_demo:
                            sales_demo['age'] = pd.to_numeric(sales_demo['age'], errors='coerce').fillna(30)
                            bins = [0, 17, 24, 34, 44, 54, 64, 150]
                            labels = ['0-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
                            sales_demo['age_band'] = pd.cut(sales_demo['age'], bins=bins, labels=labels, right=True)
                        # KPIs par âge
                        if 'age_band' in sales_demo.columns:
                            age_kpi = sales_demo.groupby('age_band').agg(
                                clients=('customer_id', 'nunique'),
                                ca=('total_amount', 'sum'),
                                commandes=('customer_id', 'count')
                            ).reset_index()
                            age_kpi['panier_moyen'] = (age_kpi['ca'] / age_kpi['commandes']).replace([np.inf, -np.inf], 0).fillna(0)
                            colA, colB = st.columns(2)
                            with colA:
                                fig_age_ca = px.bar(age_kpi, x='age_band', y='ca', title='CA par tranche d\'âge (Ar)')
                                st.plotly_chart(fig_age_ca, use_container_width=True)
                            with colB:
                                fig_age_pm = px.bar(age_kpi, x='age_band', y='panier_moyen', title='Panier Moyen par tranche d\'âge (Ar)')
                                st.plotly_chart(fig_age_pm, use_container_width=True)
                        # KPIs par genre
                        if 'gender' in available_demo:
                            gender_kpi = sales_demo.groupby('gender').agg(
                                clients=('customer_id', 'nunique'),
                                ca=('total_amount', 'sum')
                            ).reset_index()
                            fig_gender = px.bar(gender_kpi, x='gender', y='ca', title='CA par genre (Ar)')
                            st.plotly_chart(fig_gender, use_container_width=True)
                        # KPIs par ville (Top 10)
                        if 'city' in available_demo:
                            city_kpi = sales_demo.groupby('city')['total_amount'].sum().sort_values(ascending=False).head(10)
                            fig_city_rev = px.bar(x=city_kpi.values, y=city_kpi.index, orientation='h', title='Top 10 Villes par CA (Ar)')
                            st.plotly_chart(fig_city_rev, use_container_width=True)
                    except Exception as e:
                        st.warning(f"⚠️ Segmentation démographique indisponible: {str(e)}")

            # Section 5: Alertes et Recommandations
            st.subheader("🚨 Alertes et Recommandations")
            
            alerts = []
            recommendations = []
            
            # Analyse des tendances
            has_sales = sales_data is not None and not sales_data.empty and 'order_date' in sales_data.columns
            if has_sales:
                # S'assurer que les dates sont au format datetime
                sales_data['order_date'] = pd.to_datetime(sales_data['order_date'], errors='coerce')
                # Supprimer les lignes avec des dates invalides
                valid_dates = sales_data.dropna(subset=['order_date'])
                if not valid_dates.empty:
                    max_date = valid_dates['order_date'].max()
                    recent_sales = valid_dates[valid_dates['order_date'] >= (max_date - pd.Timedelta(days=7))]
                    previous_sales = valid_dates[(valid_dates['order_date'] >= (max_date - pd.Timedelta(days=14))) & 
                                               (valid_dates['order_date'] < (max_date - pd.Timedelta(days=7)))]
                else:
                    recent_sales = pd.DataFrame()
                    previous_sales = pd.DataFrame()
                
                if len(recent_sales) > 0 and len(previous_sales) > 0:
                    recent_avg = recent_sales['total_amount'].mean()
                    previous_avg = previous_sales['total_amount'].mean()
                    change = (recent_avg - previous_avg) / previous_avg * 100
                    
                    if change < -10:
                        alerts.append("🔴 Baisse significative du panier moyen (-{:.1f}%)".format(abs(change)))
                        recommendations.append("Analyser les causes de la baisse et lancer une campagne de stimulation")
                    elif change > 10:
                        alerts.append("🟢 Hausse significative du panier moyen (+{:.1f}%)".format(change))
                        recommendations.append("Capitaliser sur cette tendance positive")
            
            # Alertes sur les coûts marketing
            if not marketing_data.empty and 'cpc' in marketing_data.columns:
                avg_cpc = marketing_data['cpc'].mean()
                if avg_cpc > analytics.targets['cpc_max']:
                    alerts.append(f"🔴 CPC élevé: {avg_cpc:.0f} Ar (objectif: {analytics.targets['cpc_max']:.0f} Ar)")
                    recommendations.append("Optimiser le ciblage et revoir la stratégie d'enchères")
            
            # Affichage des alertes
            if alerts:
                st.write("**🚨 Alertes:**")
                for alert in alerts:
                    st.warning(alert)
            
            if recommendations:
                st.write("**💡 Recommandations:**")
                for rec in recommendations:
                    st.info(rec)
            
            if not alerts and not recommendations:
                st.success("✅ Aucune alerte. Toutes les métriques sont dans les objectifs!")

        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du dashboard: {str(e)}")

    # Module Rapport Final
    elif selected_tab == "📑 Rapport Final":
        st.header("📑 Rapport Final - Analyse Complète TeeTech")
        
        if st.button("📋 Générer le Rapport Complet", type="primary"):
            with st.spinner("Génération du rapport en cours..."):
                try:
                    # Chargement de toutes les données
                    customers_data = pd.read_csv('output/customers_clean.csv') if os.path.exists('output/customers_clean.csv') else pd.DataFrame()
                    sales_data = pd.read_csv('output/sales_clean.csv') if os.path.exists('output/sales_clean.csv') else pd.DataFrame()
                    marketing_data = pd.read_csv('output/marketing_clean.csv') if os.path.exists('output/marketing_clean.csv') else pd.DataFrame()
                    segments_data = pd.read_csv('output/customer_segments.csv') if os.path.exists('output/customer_segments.csv') else pd.DataFrame()
                    
                    # Calcul des personas et KPIs
                    personas = {}
                    kpi_summary = {}
                    churn_data = None
                    
                    if not segments_data.empty and not customers_data.empty and not sales_data.empty:
                        try:
                            products_data = pd.read_csv('output/products_clean.csv')
                        except:
                            products_data = pd.DataFrame()
                        
                        personas = analytics.create_customer_personas(segments_data, customers_data, sales_data, products_data)
                    
                    if not marketing_data.empty:
                        st.caption("[Rapport] Aperçu marketing dtypes et échantillon")
                        try:
                            st.write(marketing_data.dtypes.astype(str))
                            st.write(marketing_data.head(3))
                        except Exception:
                            pass
                        try:
                            _, kpi_summary, _ = analytics.calculate_marketing_kpis(marketing_data)
                            # Vérifier types du résumé
                            if kpi_summary is not None:
                                ks_debug = {k: (type(v).__name__, v) for k, v in kpi_summary.items()}
                                st.caption("[Rapport] kpi_summary types")
                                st.write(ks_debug)
                        except Exception as e:
                            st.error(f"[Rapport] Erreur calcul KPIs marketing: {str(e)}")
                            raise
                    
                    if not customers_data.empty and not sales_data.empty:
                        churn_data, _, _ = analytics.build_predictive_model(customers_data, sales_data)
                    
                    # Génération du rapport
                    report_content = analytics.generate_comprehensive_report(
                        customers_data, sales_data, marketing_data, 
                        segments_data, personas, kpi_summary, churn_data
                    )
                    
                    st.success("✅ Rapport généré avec succès!")
                    
                    # Affichage du rapport
                    st.markdown("---")
                    st.markdown(report_content)
                    
                    # Options de téléchargement
                    st.markdown("---")
                    st.subheader("⬇️ Téléchargements")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Rapport Markdown
                        st.download_button(
                            "📄 Rapport Markdown",
                            report_content,
                            "rapport_teetech_complet.md",
                            "text/markdown"
                        )
                    
                    with col2:
                        # Données compilées
                        if not customers_data.empty and not sales_data.empty:
                            # Création d'un dataset consolidé
                            consolidated_data = sales_data.merge(
                                customers_data[['customer_id', 'age', 'city']], 
                                on='customer_id', 
                                how='left'
                            )
                            
                            consolidated_csv = consolidated_data.to_csv(index=False)
                            st.download_button(
                                "📊 Données Consolidées",
                                consolidated_csv,
                                "donnees_consolidees.csv",
                                "text/csv"
                            )
                    
                    with col3:
                        # Package complet (ZIP)
                        if st.button("📦 Package Complet (.zip)"):
                            # Création d'un fichier ZIP avec tous les outputs
                            zip_buffer = io.BytesIO()
                            
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                # Ajouter le rapport
                                zip_file.writestr("rapport_complet.md", report_content)
                                
                                # Ajouter tous les fichiers CSV de output/
                                output_files = glob.glob('output/*.csv')
                                for file_path in output_files:
                                    filename = os.path.basename(file_path)
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        zip_file.writestr(f"data/{filename}", f.read())
                                
                                # Ajouter les visualisations PNG s'il y en a
                                viz_files = glob.glob('output/*.png')
                                for file_path in viz_files:
                                    filename = os.path.basename(file_path)
                                    with open(file_path, 'rb') as f:
                                        zip_file.writestr(f"visualizations/{filename}", f.read())
                            
                            zip_buffer.seek(0)
                            
                            st.download_button(
                                "📦 Télécharger Package",
                                zip_buffer.getvalue(),
                                "teetech_analytics_package.zip",
                                "application/zip"
                            )
                    
                    # Résumé exécutif visuel
                    st.markdown("---")
                    st.subheader("📈 Résumé Exécutif Visuel")
                    
                    # Métriques clés
                    if not customers_data.empty or not sales_data.empty:
                        summary_cols = st.columns(4)
                        
                        with summary_cols[0]:
                            total_customers = len(customers_data) if not customers_data.empty else 0
                            st.metric("👥 Clients Total", f"{total_customers:,}")
                        
                        with summary_cols[1]:
                            has_sales = sales_data is not None and not sales_data.empty
                            total_revenue = sales_data['total_amount'].sum() if has_sales and 'total_amount' in sales_data.columns else 0
                            st.metric("💰 CA Total", f"{total_revenue:,.0f} Ar")
                        
                        with summary_cols[2]:
                            num_segments = segments_data['cluster'].nunique() if not segments_data.empty else 0
                            st.metric("🎯 Segments", f"{num_segments}")
                        
                        with summary_cols[3]:
                            if churn_data is not None:
                                high_risk = len(churn_data[churn_data['churn_risk'] == 'Élevé'])
                                st.metric("⚠️ Risque Churn", f"{high_risk}")
                            else:
                                st.metric("⚠️ Risque Churn", "N/A")
                    
                    # Graphiques de synthèse
                    if not segments_data.empty:
                        # Valeur par segment
                        segment_value = segments_data.groupby('cluster')['monetary'].mean()
                        fig_segment_summary = px.bar(
                            x=[f'Segment {i}' for i in segment_value.index],
                            y=segment_value.values,
                            title='Valeur Moyenne par Segment Client',
                            labels={'x': 'Segments', 'y': 'Valeur Moyenne (Ar)'}
                        )
                        st.plotly_chart(fig_segment_summary, use_container_width=True)
                    
                    # Recommandations prioritaires
                    st.markdown("---")
                    st.subheader("🎯 Recommandations Prioritaires")
                    
                    priority_recs = [
                        "🔴 **Urgent**: Mettre en place un programme de rétention pour les clients à risque élevé",
                        "🟡 **Court terme**: Optimiser les campagnes marketing sous-performantes (CPC > objectif)",
                        "🟢 **Moyen terme**: Développer des offres personnalisées par segment client",
                        "🔵 **Long terme**: Investir dans l'automatisation du marketing pour améliorer l'efficacité"
                    ]
                    
                    for rec in priority_recs:
                        st.write(rec)
                    
                    # ROI estimé des recommandations
                    st.markdown("---")
                    st.subheader("💹 ROI Estimé des Recommandations")
                    
                    roi_estimates = [
                        {"Action": "Programme de rétention", "Investissement": "100,000 Ar", "ROI Attendu": "300%", "Délai": "3 mois"},
                        {"Action": "Optimisation marketing", "Investissement": "50,000 Ar", "ROI Attendu": "200%", "Délai": "1 mois"},
                        {"Action": "Personnalisation", "Investissement": "200,000 Ar", "ROI Attendu": "400%", "Délai": "6 mois"},
                        {"Action": "Automatisation", "Investissement": "500,000 Ar", "ROI Attendu": "500%", "Délai": "12 mois"}
                    ]
                    
                    roi_df = pd.DataFrame(roi_estimates)
                    st.dataframe(roi_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de la génération du rapport: {str(e)}")
                    st.error("Assurez-vous d'avoir exécuté les modules précédents.")
        
        # Aperçu des données disponibles
        st.markdown("---")
        st.subheader("📂 Données Disponibles pour le Rapport")
        
        data_status = [
            {"Module": "M2 - Nettoyage", "Statut": "✅" if os.path.exists('output/customers_clean.csv') else "❌"},
            {"Module": "M3 - Segmentation", "Statut": "✅" if os.path.exists('output/customer_segments.csv') else "❌"},
            {"Module": "M4 - Personas", "Statut": "✅" if os.path.exists('output/customer_segments.csv') else "❌"},
            {"Module": "M5 - KPIs Marketing", "Statut": "✅" if os.path.exists('output/campaign_kpis.csv') else "❌"},
            {"Module": "M6 - Prédictif", "Statut": "✅" if os.path.exists('output/churn_predictions.csv') else "❌"}
        ]
        
        status_df = pd.DataFrame(data_status)
        st.dataframe(status_df, use_container_width=True)
        
        if status_df["Statut"].str.contains("❌").any():
            st.warning("⚠️ Certains modules n'ont pas été exécutés. Le rapport sera incomplet.")
            st.info("💡 Exécutez tous les modules pour obtenir un rapport complet.")

if __name__ == "__main__":
    main()