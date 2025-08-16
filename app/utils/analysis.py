"""
Module d'analyse pour l'application d'analyse marketing.

Ce module fournit des fonctions pour effectuer des analyses statistiques
et prédictives sur les données marketing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
# Try optional xgboost imports
try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore
    XGBRegressor = None  # type: ignore
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketingAnalyzer:
    """
    Classe pour effectuer des analyses marketing avancées.
    """
    
    def __init__(self, data: Dict[str, pd.DataFrame]):
        """
        Initialise l'analyseur avec les données chargées.
        
        Args:
            data: Dictionnaire des DataFrames chargés
        """
        self.data = data
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
    
    def prepare_rfm_data(self, 
                        customers_df: pd.DataFrame, 
                        orders_df: pd.DataFrame,
                        customer_id: str = 'customer_id',
                        order_date: str = 'order_date',
                        amount: str = 'amount') -> pd.DataFrame:
        """
        Prépare les données pour l'analyse RFM (Récence, Fréquence, Montant).
        
        Args:
            customers_df: DataFrame des clients
            orders_df: DataFrame des commandes
            customer_id: Nom de la colonne d'identifiant client
            order_date: Nom de la colonne de date de commande
            amount: Nom de la colonne de montant
            
        Returns:
            pd.DataFrame: Données RFM préparées
            
        Raises:
            ValueError: Si des colonnes obligatoires sont manquantes ou si les données sont invalides
        """
        # Faire une copie pour éviter les modifications sur le DataFrame original
        orders_df = orders_df.copy()
        customers_df = customers_df.copy() if customers_df is not None else pd.DataFrame()
        
        try:
            # Vérifier les colonnes requises dans les commandes
            required_order_cols = {customer_id, order_date, amount}
            missing_order_cols = required_order_cols - set(orders_df.columns)
            if missing_order_cols:
                raise ValueError(
                    f"Colonnes manquantes dans les données de commandes : {missing_order_cols}. "
                    f"Colonnes disponibles : {list(orders_df.columns)}"
                )
                
            # Vérifier que le DataFrame des commandes n'est pas vide
            if orders_df.empty:
                raise ValueError("Le fichier de commandes est vide.")
            
            # Vérifier et convertir la colonne de date
            if not pd.api.types.is_datetime64_any_dtype(orders_df[order_date]):
                try:
                    orders_df[order_date] = pd.to_datetime(orders_df[order_date], dayfirst=True, errors='coerce')
                    # Vérifier s'il y a des valeurs manquantes après la conversion
                    if orders_df[order_date].isnull().any():
                        st.warning("Certaines dates n'ont pas pu être converties et ont été définies comme manquantes (NaN).")
                except Exception as e:
                    raise ValueError(f"Erreur lors de la conversion des dates : {e}")
            
            # Vérifier et convertir la colonne de montant
            if not pd.api.types.is_numeric_dtype(orders_df[amount]):
                try:
                    orders_df[amount] = pd.to_numeric(orders_df[amount], errors='coerce')
                    if orders_df[amount].isnull().any():
                        st.warning("Certains montants n'ont pas pu être convertis en nombres et ont été définis comme manquants (NaN).")
                except Exception as e:
                    raise ValueError(f"Erreur lors de la conversion des montants : {e}")
            
            # Date de référence (dernière date de commande + 1 jour)
            max_date = orders_df[order_date].max()
            if pd.isna(max_date):
                raise ValueError("Aucune date valide trouvée dans la colonne de date des commandes.")
                
            max_date = max_date + pd.Timedelta(days=1)
            
            # Calculer les métriques RFM
            rfm = orders_df.groupby(customer_id).agg({
                order_date: lambda x: (max_date - x.max()).days,  # Récence
                customer_id: 'count',                             # Fréquence
                amount: 'sum'                                     # Montant
            }).rename(columns={
                order_date: 'recency',
                customer_id: 'frequency',
                amount: 'monetary_value'
            }).reset_index()
            
            # Vérifier les valeurs négatives ou aberrantes
            if (rfm['monetary_value'] < 0).any():
                st.warning("Attention : Des valeurs négatives ont été détectées dans les montants.")
            
            if (rfm['frequency'] <= 0).any():
                st.warning("Attention : Des fréquences nulles ou négatives ont été détectées.")
            
            # Fusionner avec les données clients si disponible
            if not customers_df.empty and customer_id in customers_df.columns:
                # Garder uniquement les colonnes non dupliquées (sauf la clé de jointure)
                customer_cols = [col for col in customers_df.columns if col != customer_id]
                customer_cols = [col for col in customer_cols if col not in rfm.columns]
                
                if customer_cols:
                    rfm = pd.merge(
                        rfm,
                        customers_df[[customer_id] + customer_cols],
                        on=customer_id,
                        how='left'
                    )
            
            return rfm
            
        except Exception as e:
            logger.error(f"Erreur lors de la préparation des données RFM : {e}")
            raise
    
    def perform_segmentation(self, 
                           data: pd.DataFrame,
                           features: List[str],
                           n_clusters: int = 4,
                           method: str = 'kmeans',
                           random_state: int = 42) -> Tuple[pd.DataFrame, dict]:
        """
        Effectue une segmentation des clients.
        
        Args:
            data: DataFrame contenant les données à segmenter
            features: Liste des colonnes à utiliser pour la segmentation
            n_clusters: Nombre de clusters à créer
            method: Méthode de segmentation ('kmeans', 'hierarchical', 'dbscan')
            random_state: Graine aléatoire pour la reproductibilité
            
        Returns:
            Tuple contenant le DataFrame avec les segments et les métriques d'évaluation
        """
        try:
            # Vérifier les colonnes requises
            missing_features = [f for f in features if f not in data.columns]
            if missing_features:
                raise ValueError(f"Colonnes manquantes pour la segmentation : {missing_features}")
            
            # Sélectionner et normaliser les caractéristiques
            X = data[features].copy()
            X_scaled = self.scaler.fit_transform(X)
            
            # Appliquer la méthode de segmentation sélectionnée
            if method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, 
                             random_state=random_state, 
                             n_init=10)
                labels = model.fit_predict(X_scaled)
                
                # Calculer les métriques d'évaluation
                metrics = {
                    'inertia': model.inertia_,  # Somme des carrés des distances
                    'silhouette': silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0,
                    'calinski_harabasz': calinski_harabasz_score(X_scaled, labels) if len(set(labels)) > 1 else 0,
                    'davies_bouldin': davies_bouldin_score(X_scaled, labels) if len(set(labels)) > 1 else 0
                }
                
                # Ajouter les centres des clusters
                centers = pd.DataFrame(
                    self.scaler.inverse_transform(model.cluster_centers_),
                    columns=features
                )
                metrics['centers'] = centers
                
            else:
                raise ValueError(f"Méthode de segmentation non prise en charge : {method}")
            
            # Ajouter les labels au DataFrame d'origine
            segmented_data = data.copy()
            segmented_data['segment'] = labels.astype(str)
            
            # Calculer les statistiques par segment
            segment_stats = segmented_data.groupby('segment')[features].agg(['mean', 'median', 'std', 'count'])
            metrics['segment_stats'] = segment_stats
            
            return segmented_data, metrics
            
        except Exception as e:
            logger.error(f"Erreur lors de la segmentation : {e}")
            raise
    
    def analyze_campaign_performance(self, 
                                   marketing_data: pd.DataFrame,
                                   sales_data: pd.DataFrame,
                                   campaign_id: str = 'campaign_id',
                                   date_column: str = 'date',
                                   spend_column: str = 'spend',
                                   revenue_column: str = 'revenue') -> Dict:
        """
        Analyse les performances des campagnes marketing.
        
        Args:
            marketing_data: Données des campagnes marketing
            sales_data: Données des ventes
            campaign_id: Nom de la colonne d'identifiant de campagne
            date_column: Nom de la colonne de date
            spend_column: Nom de la colonne de coût
            revenue_column: Nom de la colonne de revenu
            
        Returns:
            Dictionnaire contenant les métriques de performance
        """
        try:
            # Vérifier les colonnes requises
            required_marketing = {campaign_id, date_column, spend_column}
            if not required_marketing.issubset(marketing_data.columns):
                missing = required_marketing - set(marketing_data.columns)
                raise ValueError(f"Colonnes manquantes dans les données marketing : {missing}")
            
            # Calculer les métriques de base
            campaign_metrics = marketing_data.groupby(campaign_id).agg({
                spend_column: 'sum',
                revenue_column: 'sum' if revenue_column in marketing_data.columns else None
            }).reset_index()
            
            # Calculer le ROI si les données de revenus sont disponibles
            if revenue_column in marketing_data.columns:
                campaign_metrics['roi'] = (
                    (campaign_metrics[revenue_column] - campaign_metrics[spend_column]) / 
                    campaign_metrics[spend_column]
                ) * 100  # en pourcentage
            
            # Fusionner avec les données de vente si disponibles
            if not sales_data.empty and campaign_id in sales_data.columns:
                sales_by_campaign = sales_data.groupby(campaign_id).agg({
                    'order_id': 'nunique',  # Nombre de commandes
                    'quantity': 'sum',      # Quantité totale vendue
                    'amount': 'sum'         # Montant total des ventes
                }).reset_index()
                
                campaign_metrics = pd.merge(
                    campaign_metrics, 
                    sales_by_campaign, 
                    on=campaign_id, 
                    how='left'
                )
                
                # Calculer le coût par acquisition (CPA) si possible
                if 'order_id' in campaign_metrics.columns and spend_column in campaign_metrics.columns:
                    campaign_metrics['cpa'] = campaign_metrics[spend_column] / campaign_metrics['order_id']
            
            # Calculer les métriques globales
            result = {
                'campaign_metrics': campaign_metrics,
                'total_spend': campaign_metrics[spend_column].sum(),
                'avg_roi': campaign_metrics['roi'].mean() if 'roi' in campaign_metrics.columns else None,
                'total_orders': campaign_metrics['order_id'].sum() if 'order_id' in campaign_metrics.columns else None,
                'total_revenue': campaign_metrics[revenue_column].sum() if revenue_column in campaign_metrics.columns else None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des performances des campagnes : {e}")
            raise
    
    def predict_churn(self, 
                     data: pd.DataFrame,
                     target_column: str,
                     test_size: float = 0.2,
                     random_state: int = 42) -> Dict:
        """
        Prédit le taux de désabonnement (churn) des clients.
        
        Args:
            data: DataFrame contenant les données d'entraînement
            target_column: Nom de la colonne cible (0 = actif, 1 = churn)
            test_size: Proportion des données à utiliser pour le test
            random_state: Graine aléatoire pour la reproductibilité
            
        Returns:
            Dictionnaire contenant les résultats du modèle
        """
        try:
            # Vérifier la colonne cible
            if target_column not in data.columns:
                raise ValueError(f"Colonne cible manquante : {target_column}")
            
            # Préparer les données
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Diviser en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Entraîner le modèle
            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state
            )
            
            model.fit(X_train, y_train)
            
            # Faire des prédictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculer les métriques
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Importance des caractéristiques
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Enregistrer les résultats
            result = {
                'model': model,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'classification_report': report,
                'feature_importance': feature_importance,
                'accuracy': report['accuracy'],
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score']
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction du churn : {e}")
            raise
    
    def calculate_clv(self, 
                     customer_data: pd.DataFrame,
                     transaction_data: pd.DataFrame,
                     customer_id: str = 'customer_id',
                     date_column: str = 'date',
                     amount_column: str = 'amount',
                     prediction_years: int = 3) -> pd.DataFrame:
        """
        Calcule la valeur à vie du client (Customer Lifetime Value).
        
        Args:
            customer_data: Données des clients
            transaction_data: Données des transactions
            customer_id: Nom de la colonne d'identifiant client
            date_column: Nom de la colonne de date
            amount_column: Nom de la colonne de montant
            prediction_years: Nombre d'années pour la prédiction
            
        Returns:
            DataFrame avec la CLV pour chaque client
        """
        try:
            # Vérifier les colonnes requises
            required_transaction = {customer_id, date_column, amount_column}
            if not required_transaction.issubset(transaction_data.columns):
                missing = required_transaction - set(transaction_data.columns)
                raise ValueError(f"Colonnes manquantes dans les données de transactions : {missing}")
            
            # Convertir la date si nécessaire
            if not pd.api.types.is_datetime64_any_dtype(transaction_data[date_column]):
                transaction_data[date_column] = pd.to_datetime(transaction_data[date_column])
            
            # Calculer la valeur moyenne des commandes par client
            avg_order_value = transaction_data.groupby(customer_id)[amount_column].mean().reset_index()
            avg_order_value.columns = [customer_id, 'avg_order_value']
            
            # Calculer la fréquence d'achat (nombre de commandes par an)
            min_date = transaction_data[date_column].min()
            max_date = transaction_data[date_column].max()
            days_active = (max_date - min_date).days
            years_active = max(days_active / 365, 0.1)  # Éviter la division par zéro
            
            purchase_freq = transaction_data.groupby(customer_id)[date_column].count().reset_index()
            purchase_freq.columns = [customer_id, 'total_orders']
            purchase_freq['purchase_frequency'] = purchase_freq['total_orders'] / years_active
            
            # Calculer le taux de rétention (simplifié)
            # Dans une implémentation réelle, on utiliserait un modèle de survie
            retention_rate = 0.7  # Valeur par défaut
            
            # Calculer la durée de vie moyenne du client
            avg_customer_lifespan = 1 / (1 - retention_rate)  # en années
            
            # Calculer la CLV
            clv_data = pd.merge(avg_order_value, purchase_freq, on=customer_id)
            clv_data['clv'] = (
                clv_data['avg_order_value'] * 
                clv_data['purchase_frequency'] * 
                avg_customer_lifespan * 
                prediction_years
            )
            
            # Fusionner avec les données clients si disponibles
            if not customer_data.empty and customer_id in customer_data.columns:
                clv_data = pd.merge(clv_data, customer_data, on=customer_id, how='left')
            
            return clv_data
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la CLV : {e}")
            raise

def generate_marketing_recommendations(segment_data: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Génère des recommandations marketing basées sur les segments de clients.
    
    Args:
        segment_data: DataFrame contenant les données de segmentation
        
    Returns:
        Dictionnaire des recommandations par segment
    """
    recommendations = {}
    
    # Vérifier si les colonnes nécessaires sont présentes
    if 'segment' not in segment_data.columns:
        return {"Erreur": ["La colonne 'segment' est manquante dans les données."]}
    
    # Parcourir chaque segment
    for segment in segment_data['segment'].unique():
        segment_mask = segment_data['segment'] == segment
        segment_df = segment_data[segment_mask]
        
        segment_rec = []
        
        # Exemple de logique de recommandation (à adapter selon les données réelles)
        if 'recency' in segment_df.columns and 'frequency' in segment_df.columns:
            avg_recency = segment_df['recency'].median()
            avg_frequency = segment_df['frequency'].median()
            
            if avg_recency > 180:  # Clients inactifs depuis longtemps
                segment_rec.append("Lancer une campagne de réactivation avec des offres exclusives.")
            elif avg_frequency > 10:  # Clients très actifs
                segment_rec.append("Proposer un programme de fidélité premium avec des avantages exclusifs.")
            
            if 'monetary' in segment_df.columns:
                avg_monetary = segment_df['monetary'].median()
                if avg_monetary > segment_data['monetary'].quantile(0.75):  # Gros acheteurs
                    segment_rec.append("Offrir un service client personnalisé et des avantages VIP.")
        
        # Si pas de recommandations spécifiques, ajouter des recommandations génériques
        if not segment_rec:
            segment_rec = [
                f"Personnaliser les communications pour le segment {segment}.",
                f"Analyser les préférences d'achat du segment {segment} pour des recommandations plus précises."
            ]
        
        recommendations[f"Segment {segment}"] = segment_rec
    
    return recommendations
