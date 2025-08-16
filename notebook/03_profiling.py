#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module M4 - Profilage des Segments
---------------------------------
Ce module permet d'analyser et de visualiser les caractéristiques des segments clients
définis dans l'étape précédente de segmentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
from datetime import datetime
import os
import sys
from pathlib import Path

# Configuration des chemins
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data_clean'  # Changé de 'Data' à 'data_clean'
PROCESSED_DIR = BASE_DIR / 'data_processed'
REPORTS_DIR = BASE_DIR / 'reports'

# Création des dossiers s'ils n'existent pas
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Configuration des styles
plt.style.use('seaborn-v0_8')
sns.set_style('whitegrid')
sns.set_palette('viridis')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)

# ========================
# 1. Chargement des données
# ========================

def load_segmentation_data():
    """
    Charge les données de segmentation depuis les fichiers générés précédemment.
    
    Returns:
        tuple: Un tuple contenant (segments, customers, orders, tshirts, marketing)
    """
    try:
        # Chargement des segments de clients
        segments = pd.read_csv(PROCESSED_DIR / 'customer_segments.csv')
        
        # Chargement des données clients nettoyées
        customers = pd.read_csv(DATA_DIR / 'customers_clean.csv', parse_dates=['signup_date'])
        
        # Chargement des données de commandes nettoyées (depuis data_clean)
        orders = pd.read_csv(DATA_DIR / 'orders_clean.csv', parse_dates=['order_date'])
        
        # Chargement des données de produits
        tshirts = pd.read_csv(DATA_DIR / 'tshirts_clean.csv')
        
        # Chargement des données marketing
        marketing = pd.read_csv(DATA_DIR / 'marketing_clean.csv', parse_dates=['date'])
        
        return segments, customers, orders, tshirts, marketing
    
    except FileNotFoundError as e:
        print(f"Erreur lors du chargement des fichiers : {e}")
        print(f"Répertoire de données: {DATA_DIR}")
        print(f"Répertoire traité: {PROCESSED_DIR}")
        print("Fichiers disponibles dans data_processed:", os.listdir(PROCESSED_DIR))
        print("Assurez-vous d'avoir exécuté les étapes précédentes de nettoyage et segmentation.")
        sys.exit(1)

# ========================
# 2. Analyse démographique
# ========================

def analyze_demographics(segments, customers):
    """
    Analyse la composition démographique de chaque segment.
    
    Args:
        segments (pd.DataFrame): Données des segments clients
        customers (pd.DataFrame): Données des clients
        
    Returns:
        pd.DataFrame: DataFrame enrichi avec les données démographiques
    """
    try:
        print("\n=== DEBUG: Entrée dans analyze_demographics ===")
        print("Colonnes de segments:", segments.columns.tolist())
        print("Colonnes de customers:", customers.columns.tolist())
        
        # Vérifier les valeurs uniques de customer_id dans les deux DataFrames
        print("\nValeurs de customer_id dans segments (5 premières):", segments['customer_id'].head().tolist())
        print("Valeurs de customer_id dans customers (5 premières):", customers['customer_id'].head().tolist())
        
        # Vérifier les types de données
        print("\nType de customer_id dans segments:", segments['customer_id'].dtype)
        print("Type de customer_id dans customers:", customers['customer_id'].dtype)
        
        # Fusion avec les données clients complètes
        segments_full = segments.merge(
            customers[['customer_id', 'age', 'gender', 'city', 'signup_date']],
            on='customer_id',
            how='left'
        )
        
        print("\n=== Après la fusion ===")
        print("Nombre de lignes après fusion:", len(segments_full))
        print("Colonnes après fusion:", segments_full.columns.tolist())
        
        if 'signup_date' in segments_full.columns:
            # Calcul de l'ancienneté (en jours)
            max_date = segments_full['signup_date'].max()
            segments_full['tenure_days'] = (max_date - segments_full['signup_date']).dt.days
            print("Ancienneté calculée avec succès")
        else:
            print("ATTENTION: La colonne 'signup_date' est manquante après la fusion")
        
        return segments_full
    
    except Exception as e:
        import traceback
        print(f"\n=== ERREUR DANS analyze_demographics ===")
        print(f"Type d'erreur: {type(e).__name__}")
        print(f"Message d'erreur: {str(e)}")
        print("\nStack trace:")
        traceback.print_exc()
        return None

# ========================
# 3. Comportement d'achat
# ========================

def analyze_purchase_behavior(segments, orders, tshirts):
    """
    Analyse le comportement d'achat par segment.
    
    Args:
        segments (pd.DataFrame): Données des segments clients
        orders (pd.DataFrame): Données des commandes
        tshirts (pd.DataFrame): Données des produits
        
    Returns:
        pd.DataFrame: Données de commandes enrichies avec les segments
    """
    try:
        print("\n=== DEBUG: Entrée dans analyze_purchase_behavior ===")
        print("Colonnes de segments:", segments.columns.tolist())
        print("Colonnes de orders:", orders.columns.tolist())
        print("Colonnes de tshirts:", tshirts.columns.tolist())
        
        # Vérifier que la colonne 'segment' existe dans segments
        if 'segment' not in segments.columns:
            print("Erreur : la colonne 'segment' est introuvable dans les données de segments")
            print("Colonnes disponibles dans segments:", segments.columns.tolist())
            return None
            
        # Fusion des données de commandes et produits
        orders_products = orders.merge(
            tshirts,
            on='tshirt_id',
            how='left'
        )
        
        # Fusion avec les segments en utilisant la colonne 'segment'
        orders_segments = orders_products.merge(
            segments[['customer_id', 'segment']],
            on='customer_id',
            how='inner'
        )
        
        print("\n=== Après la fusion ===")
        print("Nombre de lignes après fusion:", len(orders_segments))
        print("Colonnes après fusion:", orders_segments.columns.tolist())
        
        return orders_segments
    
    except Exception as e:
        print(f"Erreur lors de l'analyse du comportement d'achat : {e}")
        return None

# ========================
# 4. Visualisation des segments
# ========================

def plot_segment_distribution(segments_full):
    """
    Affiche la distribution des segments.
    
    Args:
        segments_full (pd.DataFrame): Données des segments avec informations démographiques
    """
    try:
        # Comptage des clients par segment
        segment_counts = segments_full['segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Nombre de clients']
        
        # Création du graphique avec Plotly
        fig = px.bar(
            segment_counts,
            x='Segment',
            y='Nombre de clients',
            title='Distribution des segments clients',
            color='Segment',
            text='Nombre de clients',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        # Personnalisation du graphique
        fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        fig.update_layout(
            xaxis_title='Segment',
            yaxis_title='Nombre de clients',
            showlegend=False,
            height=500
        )
        
        # Sauvegarde du graphique
        fig.write_html(str(REPORTS_DIR / 'segment_distribution.html'))
        
        return fig
    
    except Exception as e:
        print(f"Erreur lors de la création du graphique de distribution : {e}")
        return None

def plot_demographic_profiles(segments_full):
    """
    Affiche les profils démographiques par segment.
    
    Args:
        segments_full (pd.DataFrame): Données des segments avec informations démographiques
    """
    try:
        print("\n=== DEBUG: Entrée dans plot_demographic_profiles ===")
        print("Colonnes disponibles:", segments_full.columns.tolist())
        print("Valeurs uniques de 'segment':", segments_full['segment'].unique())
        
        # Vérifier les colonnes requises
        required_columns = ['segment', 'age', 'gender', 'city', 'tenure_days']
        missing_columns = [col for col in required_columns if col not in segments_full.columns]
        
        if missing_columns:
            print(f"ATTENTION: Colonnes manquantes pour l'analyse démographique: {missing_columns}")
            return None
            
        # Création d'une figure avec plusieurs sous-graphiques
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Âge moyen par segment',
                'Répartition par genre',
                'Ancienneté moyenne (jours)',
                'Top villes par segment'
            ),
            specs=[[{"type": "box"}, {"type": "pie"}],
                  [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Distribution d'âge par segment (boxplot)
        for segment in segments_full['segment'].unique():
            age_data = segments_full[segments_full['segment'] == segment]['age'].dropna()
            if not age_data.empty:
                fig.add_trace(
                    go.Box(
                        y=age_data,
                        name=str(segment),  # Convertir en string pour éviter les problèmes de type
                        boxpoints=False,
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # 2. Répartition par genre (camembert)
        if 'gender' in segments_full.columns and not segments_full['gender'].isna().all():
            gender_dist = segments_full.groupby(['segment', 'gender']).size().reset_index(name='count')
            for segment in segments_full['segment'].unique():
                segment_data = gender_dist[gender_dist['segment'] == segment]
                if not segment_data.empty:
                    fig.add_trace(
                        go.Pie(
                            labels=segment_data['gender'].astype(str),  # Convertir en string
                            values=segment_data['count'],
                            name=str(segment),  # Convertir en string
                            showlegend=False,
                            textinfo='percent+label',
                            hole=0.4
                        ),
                        row=1, col=2
                    )
        
        # 3. Ancienneté moyenne par segment
        if 'tenure_days' in segments_full.columns and not segments_full['tenure_days'].isna().all():
            tenure_avg = segments_full.groupby('segment')['tenure_days'].mean().reset_index()
            if not tenure_avg.empty:
                fig.add_trace(
                    go.Bar(
                        x=tenure_avg['segment'].astype(str),  # Convertir en string
                        y=tenure_avg['tenure_days'],
                        text=tenure_avg['tenure_days'].round(0).astype(int),  # Convertir en int
                        textposition='auto',
                        marker_color=px.colors.qualitative.Pastel[:len(tenure_avg)],
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # 4. Top villes par segment
        if 'city' in segments_full.columns and not segments_full['city'].isna().all():
            top_cities = segments_full.groupby(['segment', 'city']).size().reset_index(name='count')
            top_cities = top_cities.sort_values(['segment', 'count'], ascending=[True, False])
            top_cities = top_cities.groupby('segment').head(3)
            
            for segment in segments_full['segment'].unique():
                segment_cities = top_cities[top_cities['segment'] == segment]
                if not segment_cities.empty:
                    fig.add_trace(
                        go.Bar(
                            x=segment_cities['city'].astype(str),  # Convertir en string
                            y=segment_cities['count'],
                            name=str(segment),  # Convertir en string
                            showlegend=True,
                            text=segment_cities['count'].astype(str),  # Convertir en string
                            textposition='auto'
                        ),
                        row=2, col=2
                    )
        
        # Mise à jour du layout
        fig.update_layout(
            height=800,
            width=1200,
            title_text="Profils Démographiques par Segment",
            showlegend=True,
            legend_title_text="Segments"
        )
        
        # Mise à jour des axes et des titres
        fig.update_xaxes(title_text="Segments", row=2, col=1)
        fig.update_yaxes(title_text="Ancienneté (jours)", row=2, col=1)
        fig.update_xaxes(title_text="Villes", row=2, col=2)
        fig.update_yaxes(title_text="Nombre de clients", row=2, col=2)
        
        # Créer le répertoire de rapports s'il n'existe pas
        REPORTS_DIR.mkdir(exist_ok=True)
        
        # Sauvegarde du graphique
        output_path = REPORTS_DIR / 'demographic_profiles.html'
        fig.write_html(str(output_path))
        print(f"Graphique démographique enregistré sous : {output_path}")
        
        return fig
    
    except Exception as e:
        import traceback
        print(f"\n=== ERREUR DANS plot_demographic_profiles ===")
        print(f"Type d'erreur: {type(e).__name__}")
        print(f"Message d'erreur: {str(e)}")
        print("\nStack trace:")
        traceback.print_exc()
        return None
        return None

# ========================
# 5. Analyse des préférences par segment
# ========================

def analyze_product_preferences(orders_segments):
    """
    Analyse les préférences de produits par segment.
    
    Args:
        orders_segments (pd.DataFrame): Données de commandes avec segments
        
    Returns:
        dict: Dictionnaire contenant les DataFrames d'analyse
    """
    try:
        # Copie des données pour éviter les modifications accidentelles
        data = orders_segments.copy()
        
        # Vérifier les colonnes disponibles
        print("\n=== DEBUG: analyse_product_preferences ===")
        print("Colonnes disponibles dans orders_segments:", data.columns.tolist())
        
        # Identifier les colonnes de produits (celles qui ont des suffixes _x ou _y)
        product_columns = {
            'category': 'category_y' if 'category_y' in data.columns else 'category_x' if 'category_x' in data.columns else None,
            'style': 'style_y' if 'style_y' in data.columns else 'style_x' if 'style_x' in data.columns else None,
            'size': 'size_y' if 'size_y' in data.columns else 'size_x' if 'size_x' in data.columns else None,
            'color': 'color_y' if 'color_y' in data.columns else 'color_x' if 'color_x' in data.columns else None
        }
        
        # Vérifier que toutes les colonnes nécessaires sont présentes
        missing_columns = [col for col, actual_col in product_columns.items() if actual_col is None]
        missing_columns.extend([col for col in ['segment', 'order_id', 'price'] if col not in data.columns])
        
        if missing_columns:
            print(f"ATTENTION: Colonnes manquantes pour l'analyse des préférences: {missing_columns}")
            print("Colonnes de produits disponibles:", {k: v for k, v in product_columns.items() if v is not None})
            return {}
            
        # Renommer les colonnes pour faciliter l'analyse
        column_mapping = {v: k for k, v in product_columns.items() if v is not None}
        data_renamed = data.rename(columns=column_mapping)
        
        # 1. Catégories préférées par segment
        category_pref = data_renamed.groupby(['segment', 'category'])['order_id']\
            .count()\
            .reset_index()\
            .sort_values(['segment', 'order_id'], ascending=[True, False])
        
        # 2. Styles préférés par segment
        style_pref = data_renamed.groupby(['segment', 'style'])['order_id']\
            .count()\
            .reset_index()\
            .sort_values(['segment', 'order_id'], ascending=[True, False])
        
        # 3. Tailles préférées par segment
        size_pref = data_renamed.groupby(['segment', 'size'])['order_id']\
            .count()\
            .reset_index()\
            .sort_values(['segment', 'order_id'], ascending=[True, False])
        
        # 4. Couleurs préférées par segment
        color_pref = data_renamed.groupby(['segment', 'color'])['order_id']\
            .count()\
            .reset_index()\
            .sort_values(['segment', 'order_id'], ascending=[True, False])
        
        # 5. Panier moyen par segment
        avg_basket = data.groupby('segment')['price']\
            .agg(['mean', 'median', 'count'])\
            .reset_index()
        
        return {
            'category_pref': category_pref,
            'style_pref': style_pref,
            'size_pref': size_pref,
            'color_pref': color_pref,
            'avg_basket': avg_basket
        }
    
    except Exception as e:
        print(f"Erreur lors de l'analyse des préférences produits : {e}")
        return {}

# ========================
# 6. Analyse de la valeur client
# ========================

def analyze_customer_value(segments, orders):
    """
    Analyse la valeur client par segment.
    
    Args:
        segments (pd.DataFrame): Données des segments
        orders (pd.DataFrame): Données des commandes
        
    Returns:
        pd.DataFrame: Données de valeur client par segment
    """
    try:
        # Calcul du chiffre d'affaires par client
        customer_revenue = orders.groupby('customer_id')['price'].sum().reset_index()
        
        # Fusion avec les segments
        customer_value = segments.merge(
            customer_revenue,
            on='customer_id',
            how='left'
        )
        
        # Remplissage des valeurs manquantes par 0
        customer_value['price'] = customer_value['price'].fillna(0)
        
        # Calcul des métriques par segment
        value_by_segment = customer_value.groupby('segment').agg(
            total_revenue=pd.NamedAgg(column='price', aggfunc='sum'),
            avg_revenue=pd.NamedAgg(column='price', aggfunc='mean'),
            median_revenue=pd.NamedAgg(column='price', aggfunc='median'),
            customer_count=pd.NamedAgg(column='price', aggfunc='count')
        ).reset_index()
        
        # Calcul du pourcentage de revenu
        value_by_segment['revenue_pct'] = (
            value_by_segment['total_revenue'] / value_by_segment['total_revenue'].sum() * 100
        ).round(2)
        
        return value_by_segment
    
    except Exception as e:
        print(f"Erreur lors de l'analyse de la valeur client : {e}")
        return pd.DataFrame()

# ========================
# 7. Génération du rapport
# ========================

def generate_segment_report(segments_full, preferences, customer_value):
    """
    Génère un rapport HTML complet d'analyse des segments.
    
    Args:
        segments_full (pd.DataFrame): Données complètes des segments
        preferences (dict): Dictionnaire contenant les préférences par segment
        customer_value (pd.DataFrame): Données de valeur client
    """
    try:
        # Préparation des données pour le rapport
        report_date = datetime.now().strftime('%d/%m/%Y')
        total_customers = len(segments_full)
        num_segments = segments_full['segment'].nunique()
        total_revenue = customer_value['total_revenue'].sum() if not customer_value.empty else 0
        
        # Préparation des tableaux de données pour le rapport
        category_table = preferences.get('category_pref', pd.DataFrame()).to_html(index=False, classes='data-table')
        style_table = preferences.get('style_pref', pd.DataFrame()).to_html(index=False, classes='data-table')
        size_table = preferences.get('size_pref', pd.DataFrame()).to_html(index=False, classes='data-table')
        color_table = preferences.get('color_pref', pd.DataFrame()).to_html(index=False, classes='data-table')
        basket_table = preferences.get('avg_basket', pd.DataFrame()).to_html(index=False, classes='data-table')
        value_table = customer_value.to_html(index=False, classes='data-table')
        
        # Création du contenu HTML
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport d'Analyse des Segments Clients</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .section {{ margin-bottom: 40px; }}
                .section h2 {{ 
                    color: #2c3e50; 
                    border-bottom: 2px solid #3498db; 
                    padding-bottom: 5px; 
                }}
                .metrics {{ 
                    display: flex; 
                    flex-wrap: wrap; 
                    gap: 20px; 
                    margin: 20px 0;
                }}
                .metric-card {{ 
                    background: white; 
                    border-left: 4px solid #3498db; 
                    padding: 15px; 
                    flex: 1; 
                    min-width: 200px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-card h3 {{ margin: 0 0 10px 0; color: #7f8c8d; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin: 20px 0;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
                th, td {{ 
                    border: 1px solid #ddd; 
                    padding: 12px; 
                    text-align: left; 
                }}
                th {{ 
                    background-color: #f2f2f2; 
                    position: sticky;
                    top: 0;
                }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                tr:hover {{ background-color: #f1f1f1; }}
                .plot-container {{ 
                    margin: 30px 0;
                    border: 1px solid #eee;
                    padding: 15px;
                    border-radius: 5px;
                }}
                .plot-container h3 {{ 
                    margin-top: 0;
                    color: #2c3e50;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Rapport d'Analyse des Segments Clients</h1>
                    <p>Date du rapport: {report_date}</p>
                </div>
                
                <div class="section">
                    <h2>1. Vue d'ensemble</h2>
                    <p>Ce rapport présente une analyse approfondie des segments clients identifiés lors de l'étape de segmentation.</p>
                    
                    <div class="metrics">
                        <div class="metric-card">
                            <h3>Nombre total de clients</h3>
                            <div class="metric-value">{total_customers}</div>
                        </div>
                        <div class="metric-card">
                            <h3>Nombre de segments</h3>
                            <div class="metric-value">{num_segments}</div>
                        </div>
                        <div class="metric-card">
                            <h3>Chiffre d'affaires total</h3>
                            <div class="metric-value">{total_revenue:,.2f} €</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>2. Distribution des segments</h2>
                    <div class="plot-container">
                        <iframe src="segment_distribution.html" width="100%" height="500" frameborder="0"></iframe>
                    </div>
                </div>
                
                <div class="section">
                    <h2>3. Profils Démographiques</h2>
                    <div class="plot-container">
                        <iframe src="demographic_profiles.html" width="100%" height="850" frameborder="0"></iframe>
                    </div>
                </div>
                
                <div class="section">
                    <h2>4. Préférences par Segment</h2>
                    
                    <h3>4.1 Catégories préférées</h3>
                    {category_table}
                    
                    <h3>4.2 Styles préférés</h3>
                    {style_table}
                    
                    <h3>4.3 Tailles préférées</h3>
                    {size_table}
                    
                    <h3>4.4 Couleurs préférées</h3>
                    {color_table}
                    
                    <h3>4.5 Panier moyen</h3>
                    {basket_table}
                </div>
                
                <div class="section">
                    <h2>5. Analyse de la Valeur Client</h2>
                    {value_table}
                </div>
                
                <div class="section">
                    <h2>6. Recommandations Stratégiques</h2>
                    
                    <h3>6.1 Pour les clients à haute valeur</h3>
                    <ul>
                        <li>Mettre en place des programmes de fidélité premium</li>
                        <li>Proposer des offres personnalisées et exclusives</li>
                        <li>Maintenir un contact régulier avec du contenu de qualité</li>
                    </ul>
                    
                    <h3>6.2 Pour les clients à potentiel</h3>
                    <ul>
                        <li>Identifier les opportunités d'upselling et de cross-selling</li>
                        <li>Améliorer l'expérience client pour augmenter la fréquence d'achat</li>
                        <li>Mettre en place des campagnes de rétention ciblées</li>
                    </ul>
                    
                    <h3>6.3 Pour les clients à risque</h3>
                    <ul>
                        <li>Identifier les facteurs de désabonnement</li>
                        <li>Mettre en place des actions correctives</li>
                        <li>Enquêter sur les raisons de la baisse d'engagement</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>7. Conclusion</h2>
                    <p>Cette analyse approfondie des segments clients fournit des informations précieuses pour orienter les stratégies marketing et commerciales. En comprenant les caractéristiques, les préférences et le comportement d'achat de chaque segment, il est possible de personnaliser les offres et les communications pour maximiser la satisfaction et la fidélisation des clients.</p>
                </div>
            </div>
        </body>
        </html>
        """.format(
            report_date=report_date,
            total_customers=total_customers,
            num_segments=num_segments,
            total_revenue=total_revenue,
            category_table=category_table,
            style_table=style_table,
            size_table=size_table,
            color_table=color_table,
            basket_table=basket_table,
            value_table=value_table
        )
        
        # Écriture du rapport dans un fichier
        report_path = REPORTS_DIR / 'segment_analysis_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
    
    except Exception as e:
        print(f"Erreur lors de la génération du rapport : {e}")
        return None

# ========================
# 8. Fonction principale
# ========================

def main():
    """Fonction principale pour exécuter l'analyse de profilage des segments."""
    print("Démarrage de l'analyse de profilage des segments...")
    
    # 1. Chargement des données
    print("1/5 - Chargement des données...")
    segments, customers, orders, tshirts, marketing = load_segmentation_data()
    
    # 2. Analyse démographique
    print("2/5 - Analyse démographique...")
    # Renommer la colonne 'Cluster' en 'segment' pour la compatibilité avec les fonctions existantes
    segments_renamed = segments.rename(columns={'Cluster': 'segment'})
    segments_full = analyze_demographics(segments_renamed, customers)
    
    # 3. Analyse du comportement d'achat
    print("3/5 - Analyse du comportement d'achat...")
    orders_segments = analyze_purchase_behavior(segments_renamed, orders, tshirts)
    
    if orders_segments is None:
        print("Erreur lors de l'analyse du comportement d'achat. Vérifiez les logs ci-dessus.")
        return
    
    # 4. Génération des visualisations
    print("4/5 - Génération des visualisations...")
    try:
        if segments_full is not None:
            plot_segment_distribution(segments_full)
            plot_demographic_profiles(segments_full)
        else:
            print("Avertissement : Impossible de générer les visualisations, segments_full est None")
    except Exception as e:
        print(f"Erreur lors de la génération des visualisations : {e}")
    
    # 5. Analyse des préférences
    preferences = {}
    if orders_segments is not None:
        try:
            preferences = analyze_product_preferences(orders_segments) or {}
        except Exception as e:
            print(f"Erreur lors de l'analyse des préférences : {e}")
    else:
        print("Avertissement : Impossible d'analyser les préférences, orders_segments est None")
    
    # 6. Analyse de la valeur client
    customer_value = None
    if segments is not None:
        try:
            customer_value = analyze_customer_value(segments_renamed, orders)
        except Exception as e:
            print(f"Erreur lors de l'analyse de la valeur client : {e}")
    else:
        print("Avertissement : Impossible d'analyser la valeur client, segments est None")
    
    # 7. Génération du rapport
    print("5/5 - Génération du rapport...")
    report_path = generate_segment_report(segments_full, preferences, customer_value)
    
    print("\nAnalyse terminée avec succès !")
    print(f"Rapport généré : {report_path}")
    print("\nProchaines étapes recommandées :")
    print("1. Examiner le rapport HTML généré dans le dossier 'reports'")
    print("2. Consulter les recommandations stratégiques")
    print("3. Partager les résultats avec les équipes concernées")

if __name__ == "__main__":
    main()
