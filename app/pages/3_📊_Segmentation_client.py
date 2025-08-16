"""
Page de segmentation client pour l'application d'analyse marketing.
Permet de segmenter les clients en groupes homogènes basés sur leur comportement d'achat.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Ajouter le répertoire parent au chemin pour les imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.visualization import display_metrics, plot_distribution, plot_correlation_heatmap
from utils.analysis import MarketingAnalyzer
from config import APP_CONFIG

# Configuration de la page
st.set_page_config(
    page_title=f"{APP_CONFIG['app_name']} - Segmentation client",
    page_icon="📊",
    layout="wide"
)

# Titre de la page
st.title("📊 Segmentation client")
st.markdown("---")

# Vérifier que les données sont chargées
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Veuillez d'abord importer et valider les données dans l'onglet 'Importation des données'.")
    st.stop()

# Récupérer les données de la session
customers_df = st.session_state.get('customers_df')
orders_df = st.session_state.get('orders_df')

# Vérifier que les données nécessaires sont disponibles
if customers_df is None or orders_df is None:
    st.error("Les données clients et/ou commandes sont manquantes. Veuillez importer les données nécessaires.")
    st.stop()

# Afficher les colonnes disponibles pour le débogage
st.write("Colonnes disponibles dans les données de commandes:", orders_df.columns.tolist())
st.write("Types de données dans les commandes:", orders_df.dtypes)

# Vérifier les colonnes requises pour RFM
required_columns = ['customer_id', 'order_date', 'amount']
missing_columns = [col for col in required_columns if col not in orders_df.columns]
if missing_columns:
    st.error(f"Colonnes requises manquantes dans les données de commandes: {missing_columns}")
else:
    st.success("Toutes les colonnes requises sont présentes dans les données de commandes.")
    
    # Afficher des exemples de valeurs pour les colonnes requises
    st.write("\nExemples de valeurs pour les colonnes requises:")
    for col in required_columns:
        st.write(f"{col} (type: {orders_df[col].dtype}): {orders_df[col].head().tolist()}")

# Initialiser l'analyseur marketing
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = MarketingAnalyzer({
        'customers': customers_df,
        'orders': orders_df
    })

# Section de préparation des données RFM
with st.expander("📊 Préparation des données RFM", expanded=True):
    st.markdown("""
    La segmentation RFM (Récence, Fréquence, Montant) est une technique de segmentation des clients 
    basée sur leur comportement d'achat. Cette section prépare les données pour l'analyse RFM.
    """)
    
    # Vérifier si la colonne de date est au bon format
    if 'order_date' in orders_df.columns and not pd.api.types.is_datetime64_any_dtype(orders_df['order_date']):
        orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
    
    # Calculer les métriques RFM
    if st.button("Calculer les métriques RFM", key="calculate_rfm"):
        with st.spinner("Calcul des métriques RFM en cours..."):
            try:
                # Préparer les données RFM
                rfm_data = st.session_state.analyzer.prepare_rfm_data(
                    customers_df=customers_df,
                    orders_df=orders_df,
                    customer_id='customer_id',
                    order_date='order_date',
                    amount='amount'
                )
                
                # Enregistrer les données RFM dans la session
                st.session_state.rfm_data = rfm_data
                st.session_state.rfm_calculated = True
                
                # Afficher un aperçu des données RFM
                st.success("Métriques RFM calculées avec succès !")
                st.dataframe(rfm_data.head(), use_container_width=True)
                
                # Afficher les statistiques descriptives
                st.subheader("Statistiques descriptives RFM")
                st.dataframe(rfm_data[['recency', 'frequency', 'monetary_value']].rename(columns={'monetary_value': 'monetary'}).describe(), use_container_width=True)
                
            except Exception as e:
                st.error(f"Erreur lors du calcul des métriques RFM : {e}")
    
    elif 'rfm_data' in st.session_state:
        st.success("Métriques RFM déjà calculées.")
        st.dataframe(st.session_state.rfm_data.head(), use_container_width=True)
        
        # Afficher les statistiques descriptives
        st.subheader("Statistiques descriptives RFM")
        st.dataframe(st.session_state.rfm_data[['recency', 'frequency', 'monetary_value']].rename(columns={'monetary_value': 'monetary'}).describe(), use_container_width=True)

# Vérifier si les données RFM sont disponibles
if 'rfm_data' not in st.session_state:
    st.warning("Veuillez d'abord calculer les métriques RFM ci-dessus.")
    st.stop()

# Section de segmentation des clients
with st.expander("🔍 Segmentation des clients", expanded=True):
    st.markdown("""
    Cette section permet de segmenter les clients en groupes homogènes en utilisant l'algorithme K-means 
    sur les métriques RFM. Vous pouvez ajuster les paramètres de segmentation ci-dessous.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sélection des variables pour la segmentation
        # Ne proposer que les colonnes présentes dans le DataFrame RFM
        available_columns = [col for col in ['recency', 'frequency', 'monetary_value'] 
                           if col in st.session_state.rfm_data.columns]
        
        # S'assurer qu'il y a au moins une colonne disponible
        if not available_columns:
            st.error("Aucune colonne RFM valide trouvée. Veuillez vérifier vos données.")
            st.stop()
            
        # Définir les valeurs par défaut en fonction des colonnes disponibles
        default_columns = [col for col in ['recency', 'frequency', 'monetary_value'] 
                         if col in available_columns]
        
        selected_features = st.multiselect(
            "Variables pour la segmentation",
            options=available_columns,
            default=default_columns[:2] if len(default_columns) >= 2 else default_columns,
            key="segmentation_features"
        )
        
        # Vérifier qu'au moins une variable est sélectionnée
        if not selected_features:
            st.error("Veuillez sélectionner au moins une variable pour la segmentation.")
            st.stop()
        
        # Nombre de segments
        n_clusters = st.slider(
            "Nombre de segments",
            min_value=2,
            max_value=8,
            value=4,
            step=1,
            key="n_clusters"
        )
    
    with col2:
        # Options avancées
        st.markdown("**Options avancées**")
        
        # Normalisation des données
        normalize_data = st.checkbox(
            "Normaliser les données",
            value=True,
            help="Mettre à l'échelle les variables pour qu'elles aient une moyenne de 0 et un écart-type de 1."
        )
        
        # Algorithme de segmentation
        algorithm = st.selectbox(
            "Algorithme de segmentation",
            ["K-means", "Agglomérative", "DBSCAN"],
            index=0,
            key="segmentation_algorithm"
        )
    
    # Bouton pour lancer la segmentation
    if st.button("Lancer la segmentation", type="primary", key="run_segmentation"):
        with st.spinner("Segmentation en cours..."):
            try:
                # Préparer les données pour la segmentation
                X = st.session_state.rfm_data[selected_features].copy()
                
                # Normaliser les données si nécessaire
                if normalize_data:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                
                # Appliquer l'algorithme de segmentation sélectionné
                if algorithm == "K-means":
                    from sklearn.cluster import KMeans
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = model.fit_predict(X)
                    
                elif algorithm == "Agglomérative":
                    from sklearn.cluster import AgglomerativeClustering
                    model = AgglomerativeClustering(n_clusters=n_clusters)
                    labels = model.fit_predict(X)
                    
                elif algorithm == "DBSCAN":
                    from sklearn.cluster import DBSCAN
                    model = DBSCAN(eps=0.5, min_samples=5)
                    labels = model.fit_predict(X)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                # Ajouter les labels au DataFrame RFM
                st.session_state.rfm_data['segment'] = labels.astype(str)
                
                # Calculer les métriques d'évaluation
                from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
                
                if len(set(labels)) > 1:  # Au moins 2 clusters non bruit
                    silhouette = silhouette_score(X, labels)
                    calinski = calinski_harabasz_score(X, labels)
                    davies = davies_bouldin_score(X, labels)
                else:
                    silhouette = calinski = davies = None
                
                # Enregistrer les résultats
                st.session_state.segmentation_results = {
                    'model': model,
                    'n_clusters': n_clusters,
                    'features': selected_features,
                    'metrics': {
                        'silhouette': silhouette,
                        'calinski_harabasz': calinski,
                        'davies_bouldin': davies
                    }
                }
                
                st.success(f"Segmentation terminée avec succès ! {n_clusters} segments identifiés.")
                
            except Exception as e:
                st.error(f"Erreur lors de la segmentation : {e}")
    
    # Afficher les résultats de la segmentation si disponibles
    if 'segmentation_results' in st.session_state:
        results = st.session_state.segmentation_results
        
        # Afficher les métriques d'évaluation
        st.subheader("Qualité de la segmentation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Score de silhouette", 
                     f"{results['metrics']['silhouette']:.3f}" if results['metrics']['silhouette'] is not None else "N/A",
                     help="Mesure de la cohésion et de la séparation des clusters (entre -1 et 1, plus c'est élevé, mieux c'est)")
        
        with col2:
            st.metric("Score de Calinski-Harabasz", 
                     f"{results['metrics']['calinski_harabasz']:.1f}" if results['metrics']['calinski_harabasz'] is not None else "N/A",
                     help="Rapport entre la dispersion inter-cluster et intra-cluster (plus c'est élevé, mieux c'est)")
        
        with col3:
            st.metric("Indice de Davies-Bouldin", 
                     f"{results['metrics']['davies_bouldin']:.3f}" if results['metrics']['davies_bouldin'] is not None else "N/A",
                     help="Mesure de similarité moyenne entre les clusters (plus c'est bas, mieux c'est)")
        
        # Afficher la répartition des segments
        st.subheader("Répartition des segments")
        
        # Compter le nombre de clients par segment
        segment_counts = st.session_state.rfm_data['segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Nombre de clients']
        
        # Trier les segments par taille décroissante
        segment_counts = segment_counts.sort_values('Nombre de clients', ascending=False)
        
        # Afficher le graphique à barres
        fig = px.bar(
            segment_counts,
            x='Segment',
            y='Nombre de clients',
            color='Segment',
            title="Répartition des clients par segment",
            text='Nombre de clients',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        # Personnaliser le graphique
        fig.update_traces(
            texttemplate='%{text:,}',
            textposition='outside',
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5
        )
        
        fig.update_layout(
            xaxis_title="Segment",
            yaxis_title="Nombre de clients",
            showlegend=False,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Afficher les caractéristiques des segments
        st.subheader("Caractéristiques des segments")
        
        # Calculer les statistiques par segment
        segment_stats = st.session_state.rfm_data.groupby('segment').agg({
            'recency': ['mean', 'median', 'min', 'max'],
            'frequency': ['mean', 'median', 'min', 'max'],
            'monetary_value': ['mean', 'median', 'min', 'max', 'count']
        }).round(2)
        
        # Afficher les statistiques dans des onglets
        tab1, tab2, tab3 = st.tabs([" Vue d'ensemble", " Graphiques", " Détails"])
        
        with tab1:
            # Afficher les moyennes par segment
            st.markdown("**Moyennes par segment**")
            mean_stats = st.session_state.rfm_data.groupby('segment').agg({
                'recency': 'mean',
                'frequency': 'mean',
                'monetary_value': 'mean'
            }).reset_index()
            
            # Renommer les colonnes pour l'affichage
            mean_stats = mean_stats.rename(columns={
                'recency': 'Récence (jours)',
                'frequency': 'Fréquence',
                'monetary_value': 'Valeur monétaire (€)'
            })
            
            # Afficher le tableau
            st.dataframe(
                mean_stats,
                column_config={
                    'segment': 'Segment',
                    'Récence (jours)': 'Récence (jours)',
                    'Fréquence': 'Fréquence',
                    'Valeur monétaire (€)': 'Valeur monétaire (€)'
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Afficher les profils des segments
            st.markdown("**Profils des segments**")
            
            # Définir les profils en fonction des métriques RFM
            for segment in sorted(st.session_state.rfm_data['segment'].unique()):
                seg_data = st.session_state.rfm_data[st.session_state.rfm_data['segment'] == segment]
                
                # Calculer les moyennes
                recency = seg_data['recency'].mean()
                frequency = seg_data['frequency'].mean()
                monetary = seg_data['monetary_value'].mean()
                
                # Déterminer le profil
                if recency < 30 and frequency > 5 and monetary > 500:
                    profile = "Clients fidèles à forte valeur"
                elif recency < 60 and frequency > 3 and monetary > 200:
                    profile = "Clients réguliers"
                elif recency > 180:
                    profile = "Clients inactifs"
                elif monetary < 100:
                    profile = "Petits acheteurs"
                else:
                    profile = "Autres clients"
                
                # Afficher la carte du segment
                with st.expander(f"Segment {segment} - {profile}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Récence moyenne", f"{recency:.1f} jours")
                    with col2:
                        st.metric("Fréquence moyenne", f"{frequency:.1f} achats")
                    with col3:
                        st.metric("Valeur moyenne", f"{monetary:.2f} €")
                    
                    # Recommandations pour le segment
                    st.markdown("**Recommandations**")
                    
                    if profile == "Clients fidèles à forte valeur":
                        st.markdown("""
                        - Offrir des avantages exclusifs (accès anticipé, cadeaux personnalisés)
                        - Programme de fidélité premium
                        - Services personnalisés et dédiés
                        """)
                    elif profile == "Clients réguliers":
                        st.markdown("""
                        - Inciter à des achats plus fréquents
                        - Offres croisées sur des produits complémentaires
                        - Programme de parrainage
                        """)
                    elif profile == "Clients inactifs":
                        st.markdown("""
                        - Campagnes de réactivation
                        - Offres spéciales pour les faire revenir
                        - Enquête de satisfaction pour comprendre leur départ
                        """)
                    elif profile == "Petits acheteurs":
                        st.markdown("""
                        - Inciter à des achats plus importants (seuils de livraison gratuite)
                        - Offres groupées
                        - Échantillons gratuits pour encourager des achats plus importants
                        """)
                    else:
                        st.markdown("""
                        - Analyser plus en détail le comportement d'achat
                        - Personnaliser les offres en fonction de l'historique
                        - Tester différentes approches marketing
                        """)
        
        with tab2:
            # Graphique radar pour comparer les segments
            st.markdown("**Comparaison des segments (Radar)**")
            
            # Préparer les données pour le graphique radar
            # Utiliser les colonnes disponibles dans mean_stats
            available_metrics = [col for col in ['Récence (jours)', 'Fréquence', 'Valeur monétaire (€)'] 
                              if col in mean_stats.columns]
            
            if not available_metrics:
                st.warning("Aucune métrique disponible pour le graphique radar.")
                st.stop()
            
            # Créer une copie avec les noms de colonnes originaux pour le traitement
            radar_data = mean_stats.rename(columns={
                'Récence (jours)': 'recency',
                'Fréquence': 'frequency',
                'Valeur monétaire (€)': 'monetary_value'
            }).melt(
                id_vars=['segment'],
                value_vars=[col for col in ['recency', 'frequency', 'monetary_value'] 
                          if col in mean_stats.columns or 
                             {'Récence (jours)': 'recency', 
                              'Fréquence': 'frequency', 
                              'Valeur monétaire (€)': 'monetary_value'}.get(col) in mean_stats.columns],
                var_name='metric',
                value_name='value'
            )
            
            # Normaliser les valeurs entre 0 et 1 pour une meilleure visualisation
            for metric in radar_data['metric'].unique():
                metric_data = radar_data[radar_data['metric'] == metric]
                if not metric_data.empty:
                    min_val = metric_data['value'].min()
                    max_val = metric_data['value'].max()
                    if max_val > min_val:  # Éviter la division par zéro
                        radar_data.loc[radar_data['metric'] == metric, 'normalized'] = (
                            (radar_data[radar_data['metric'] == metric]['value'] - min_val) / (max_val - min_val) * 100
                        )
                    else:
                        radar_data.loc[radar_data['metric'] == metric, 'normalized'] = 50  # Valeur moyenne si pas de variation
            
            # Créer le graphique radar
            fig_radar = go.Figure()
            
            for segment in radar_data['segment'].unique():
                segment_data = radar_data[radar_data['segment'] == segment]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=segment_data['normalized'],
                    theta=segment_data['metric'],
                    name=f'Segment {segment}',
                    fill='toself',
                    line=dict(color=px.colors.qualitative.Plotly[int(segment) % len(px.colors.qualitative.Plotly)])
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                height=600,
                title="Comparaison des segments (valeurs normalisées)"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Graphique 3D pour visualiser les segments
            st.markdown("**Visualisation 3D des segments**")
            
            if len(selected_features) >= 3:
                fig_3d = px.scatter_3d(
                    st.session_state.rfm_data,
                    x=selected_features[0],
                    y=selected_features[1],
                    z=selected_features[2],
                    color='segment',
                    title=f"Segmentation 3D - {selected_features[0]} vs {selected_features[1]} vs {selected_features[2]}",
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                
                fig_3d.update_layout(
                    scene=dict(
                        xaxis_title=selected_features[0],
                        yaxis_title=selected_features[1],
                        zaxis_title=selected_features[2]
                    ),
                    height=700
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning("Sélectionnez au moins 3 variables pour la visualisation 3D.")
        
        with tab3:
            # Afficher les statistiques détaillées
            st.markdown("**Statistiques détaillées par segment**")
            
            # Sélectionner la métrique à analyser
            metric = st.selectbox(
                "Sélectionnez une métrique",
                ['recency', 'frequency', 'monetary'],
                key="metric_selector"
            )
            
            # Afficher la distribution de la métrique sélectionnée par segment
            fig_dist = px.histogram(
                st.session_state.rfm_data,
                x=metric,
                color='segment',
                marginal="box",
                title=f"Distribution de {metric} par segment",
                nbins=30,
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            
            fig_dist.update_layout(
                barmode='overlay',
                height=500,
                showlegend=True
            )
            
            # Ajuster l'opacité des barres
            fig_dist.update_traces(opacity=0.6)
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Afficher un tableau croisé des statistiques
            st.markdown("**Tableau croisé des statistiques**")
            
            # Calculer les statistiques par segment
            # Vérifier les colonnes disponibles
            available_columns = st.session_state.rfm_data.columns
            agg_dict = {}
            
            # Ajouter les colonnes disponibles au dictionnaire d'agrégation
            if 'recency' in available_columns:
                agg_dict['recency'] = ['mean', 'median', 'std', 'min', 'max']
            if 'frequency' in available_columns:
                agg_dict['frequency'] = ['mean', 'median', 'std', 'min', 'max']
            if 'monetary_value' in available_columns:
                agg_dict['monetary_value'] = ['mean', 'median', 'std', 'min', 'max', 'count']
            
            # Vérifier qu'il y a des colonnes à agréger
            if not agg_dict:
                st.warning("Aucune colonne RFM valide trouvée pour les statistiques par segment.")
                st.stop()
                
            # Calculer les statistiques
            stats_by_segment = st.session_state.rfm_data.groupby('segment').agg(agg_dict).round(2)
            
            # Afficher le tableau avec mise en forme conditionnelle
            st.dataframe(
                stats_by_segment.style.background_gradient(cmap='YlGnBu', axis=0),
                use_container_width=True
            )
            
            # Télécharger les résultats de la segmentation
            st.markdown("**Exporter les résultats**")
            
            if st.button("Télécharger les segments clients", key="download_segments"):
                # Créer un fichier CSV avec les segments
                csv = st.session_state.rfm_data.to_csv(index=False)
                
                # Générer un nom de fichier avec la date
                from datetime import datetime
                filename = f"segments_clients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                # Proposer le téléchargement
                st.download_button(
                    label="Télécharger au format CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv"
                )

# Navigation entre les pages
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("← Précédent", use_container_width=True):
        st.switch_page("pages/2_🔍_Exploration_des_données.py")

with col3:
    if st.button("Suivant →", type="primary", use_container_width=True):
        st.switch_page("pages/4_📈_Analyse_des_performances.py")

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
        font-size: 1.3em;
        font-weight: 600;
    }
    
    /* Style des boutons de navigation */
    .stButton>button {
        border-radius: 20px;
        padding: 0.5rem 1rem;
        margin: 10px 0;
    }
    
    /* Style des cartes de segment */
    .segment-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #4e79a7;
    }
    
    .segment-card h4 {
        margin-top: 0;
        color: #2c3e50;
    }
    
    .segment-card p {
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)
