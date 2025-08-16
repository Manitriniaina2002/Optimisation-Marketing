# ========================
# Module M3 - Segmentation Client
# Projet Marketing Analytics
# ========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Configuration du style des graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette('viridis')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)

# ========================
# 1. Chargement des données
# ========================

def load_data():
    """Charge les données nettoyées depuis le dossier data_clean."""
    try:
        customers = pd.read_csv('../data_clean/customers_clean.csv', encoding='utf-8')
        orders = pd.read_csv('../data_clean/orders_clean.csv', encoding='utf-8')
        tshirts = pd.read_csv('../data_clean/tshirts_clean.csv', encoding='utf-8')
        return customers, orders, tshirts
    except FileNotFoundError:
        print("Erreur: Les fichiers de données nettoyés sont introuvables.")
        print("Veuillez d'abord exécuter le notebook 01_explore.ipynb pour nettoyer les données.")
        return None, None, None

# ========================
# 2. Préparation des données
# ========================

def prepare_segmentation_data(customers, orders, tshirts):
    """Prépare les données pour la segmentation."""
    # Les données clients sont déjà dans le fichier orders_clean.csv
    # Nous allons utiliser directement les données de commandes qui contiennent déjà les informations clients
    
    # Conversion des dates
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    
    # Agrégation par client
    customer_data = orders.groupby('customer_id').agg({
        'name': 'first',
        'age': 'first',
        'gender': 'first',
        'city': 'first',
        'quantity': 'sum',
        'amount': 'sum',
        'order_date': ['min', 'max', 'count']
    })
    
    # Aplatissement des colonnes multi-niveaux
    customer_data.columns = ['_'.join(col).strip() for col in customer_data.columns.values]
    
    # Calcul de la récence (en jours depuis la dernière commande)
    last_date = customer_data['order_date_max'].max()
    customer_data['recency'] = (pd.to_datetime(last_date) - pd.to_datetime(customer_data['order_date_max'])).dt.days
    
    # Calcul de la fréquence
    customer_data['frequency'] = customer_data['order_date_count']
    
    # Calcul de la valeur monétaire
    customer_data['monetary'] = customer_data['amount_sum']
    
    # Sélection des variables pour la segmentation
    rfm_data = customer_data[['recency', 'frequency', 'monetary']]
    
    return rfm_data, customer_data

# ========================
# 3. Analyse RFM
# ========================

def calculate_rfm_scores(rfm_data):
    """Calcule les scores RFM pour chaque client."""
    # Fonction pour calculer les quintiles avec gestion des doublons
    def safe_qcut(series, labels):
        try:
            # Essayer de faire un qcut standard
            return pd.qcut(series, q=5, labels=labels, duplicates='drop')
        except ValueError:
            # Si toutes les valeurs sont identiques, attribuer le score médian (3)
            return pd.Series(3, index=series.index)
    
    # Création des quintiles (1 = plus bas, 5 = plus haut)
    rfm_data['R'] = safe_qcut(rfm_data['recency'], labels=[5, 4, 3, 2, 1])
    rfm_data['F'] = safe_qcut(rfm_data['frequency'], labels=[1, 2, 3, 4, 5])
    rfm_data['M'] = safe_qcut(rfm_data['monetary'], labels=[1, 2, 3, 4, 5])
    
    # Conversion en numérique
    rfm_data['R'] = rfm_data['R'].astype(int)
    rfm_data['F'] = rfm_data['F'].astype(int)
    rfm_data['M'] = rfm_data['M'].astype(int)
    
    # Score RFM global
    rfm_data['RFM_Score'] = rfm_data['R'] + rfm_data['F'] + rfm_data['M']
    
    return rfm_data

# ========================
# 4. Segmentation par K-means
# ========================

def perform_kmeans_clustering(rfm_data, n_clusters=5):
    """Effectue un clustering K-means sur les données RFM normalisées."""
    # Sélection des caractéristiques
    X = rfm_data[['recency', 'frequency', 'monetary']]
    
    # Normalisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Calcul du score de silhouette
    silhouette_avg = silhouette_score(X_scaled, clusters)
    print(f"Score de silhouette: {silhouette_avg:.2f}")
    
    # Ajout des clusters aux données
    rfm_data['Cluster'] = clusters
    
    return rfm_data, X_scaled, kmeans

# ========================
# 5. Visualisation des clusters
# ========================

def plot_clusters_3d(rfm_data, x='recency', y='frequency', z='monetary'):
    """Affiche une visualisation 3D des clusters."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        rfm_data[x], 
        rfm_data[y], 
        rfm_data[z],
        c=rfm_data['Cluster'],
        cmap='viridis',
        s=50,
        alpha=0.6
    )
    
    ax.set_xlabel(x.capitalize())
    ax.set_ylabel(y.capitalize())
    ax.set_zlabel(z.capitalize())
    ax.set_title('Segmentation des clients (3D)')
    
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.savefig('../reports/segmentation_3d.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_cluster_profiles(rfm_data):
    """Affiche les profils des clusters."""
    cluster_profile = rfm_data.groupby('Cluster').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'R': 'mean',
        'F': 'mean',
        'M': 'mean',
        'RFM_Score': 'mean'
    }).round(2)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        cluster_profile.T,
        cmap='viridis',
        annot=True,
        fmt='.2f',
        linewidths=0.5
    )
    plt.title('Profils des segments de clients')
    plt.tight_layout()
    plt.savefig('../reports/cluster_profiles.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cluster_profile

# ========================
# 6. Analyse des segments
# ========================

def analyze_segments(rfm_data, customer_data):
    """Analyse détaillée des segments de clients."""
    # Fusion avec les données clients
    segments = pd.merge(
        rfm_data[['Cluster', 'recency', 'frequency', 'monetary', 'RFM_Score']],
        customer_data[['age_first', 'gender_first', 'city_first']],
        left_index=True,
        right_index=True
    )
    
    # Statistiques descriptives par cluster
    segment_stats = segments.groupby('Cluster').agg({
        'age_first': 'mean',
        'gender_first': lambda x: x.mode()[0],
        'city_first': lambda x: x.mode()[0],
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'RFM_Score': 'mean'
    }).round(2)
    
    return segments, segment_stats

# ========================
# 7. Fonction principale
# ========================

import os
from pathlib import Path

def main():
    """Fonction principale pour la segmentation client."""
    print("Début de la segmentation client...")
    
    # Créer le répertoire data_processed s'il n'existe pas
    output_dir = Path("../data_processed")
    output_dir.mkdir(exist_ok=True)
    
    # Chargement des données
    print("\nChargement des données...")
    customers, orders, tshirts = load_data()
    
    if customers is None or orders is None or tshirts is None:
        return
    
    # Préparation des données
    print("\nPréparation des données...")
    rfm_data, customer_data = prepare_segmentation_data(customers, orders, tshirts)
    
    # Calcul des scores RFM
    print("\nCalcul des scores RFM...")
    rfm_data = calculate_rfm_scores(rfm_data)
    
    # Clustering avec K-means
    print("\nClustering des clients...")
    n_clusters = 5
    rfm_data, X_scaled, kmeans = perform_kmeans_clustering(rfm_data, n_clusters)
    
    # Visualisation des résultats
    print("\nGénération des visualisations...")
    plot_clusters_3d(rfm_data, 'recency', 'frequency', 'monetary')
    cluster_profile = plot_cluster_profiles(rfm_data)
    
    # Analyse des segments
    print("\nAnalyse des segments...")
    segments, segment_stats = analyze_segments(rfm_data, customer_data)
    
    # Sauvegarde des résultats
    print("\nSauvegarde des résultats...")
    segments.to_csv(output_dir / 'customer_segments.csv', index=True)
    rfm_data.to_csv(output_dir / 'rfm_data.csv', index=True)
    
    print("\n✅ Segmentation terminée avec succès !")
    print("\nRésumé des segments:")
    print(segment_stats)

if __name__ == "__main__":
    main()
