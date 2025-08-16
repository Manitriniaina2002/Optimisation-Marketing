"""
Module 7: Stratégie Marketing Digitale (M7)

Ce module développe une stratégie marketing digitale basée sur les segments clients identifiés
et les prédictions de churn. Il propose des actions personnalisées pour chaque segment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Configuration des chemins
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data_processed'
REPORTS_DIR = BASE_DIR / 'reports'

# Création des répertoires si nécessaire
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_segment_data():
    """Charge les données des segments clients et les prédictions de churn."""
    try:
        # Charger les segments clients
        segments_path = DATA_DIR / 'customer_segments.csv'
        segments_df = pd.read_csv(segments_path)
        
        # Charger les prédictions de churn
        churn_predictions_path = DATA_DIR / 'churn_predictions.csv'
        if os.path.exists(churn_predictions_path):
            churn_df = pd.read_csv(churn_predictions_path)
            # Fusionner avec les segments
            segments_df = segments_df.merge(
                churn_df[['customer_id', 'churn_probability', 'churn_risk']],
                on='customer_id',
                how='left'
            )
        
        return segments_df
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return None

def analyze_segment_characteristics(segments_df):
    """Analyse les caractéristiques clés de chaque segment."""
    if segments_df is None or segments_df.empty:
        return {}
    
    segment_analysis = {}
    
    # Vérifier si la colonne 'Cluster' existe, sinon utiliser une valeur par défaut
    cluster_col = 'Cluster' if 'Cluster' in segments_df.columns else 'cluster'
    
    for cluster_id in segments_df[cluster_col].unique():
        cluster_data = segments_df[segments_df[cluster_col] == cluster_id]
        
        # Calculer les métriques clés
        segment_analysis[cluster_id] = {
            'count': len(cluster_data),
            'avg_clv': cluster_data['monetary'].mean() if 'monetary' in cluster_data.columns else 0,
            'avg_purchase_freq': cluster_data['frequency'].mean() if 'frequency' in cluster_data.columns else 0,
            'avg_order_value': cluster_data['monetary'].mean() if 'monetary' in cluster_data.columns else 0,
            'recency': cluster_data['recency'].mean() if 'recency' in cluster_data.columns else 0,
            'rfm_score': cluster_data['RFM_Score'].mean() if 'RFM_Score' in cluster_data.columns else 0,
            'avg_age': cluster_data['age_first'].mean() if 'age_first' in cluster_data.columns else 0,
            'gender_distribution': cluster_data['gender_first'].value_counts().to_dict() if 'gender_first' in cluster_data.columns else {}
        }
    
    return segment_analysis

def generate_segment_strategies(segment_analysis):
    """Génère des stratégies personnalisées pour chaque segment."""
    strategies = {}
    
    # Calculer la CLV moyenne globale pour référence
    avg_clv_global = sum(metrics['avg_clv'] for metrics in segment_analysis.values()) / len(segment_analysis) if segment_analysis else 0
    
    for segment_id, metrics in segment_analysis.items():
        # Stratégie de base pour chaque segment
        strategy = {
            'objectifs': [],
            'canaux_prioritaires': [],
            'actions_specifiques': [],
            'budget_alloue': 0,
            'kpis': []
        }
        
        # Personnalisation basée sur les métriques du segment
        if metrics['avg_clv'] > avg_clv_global:  # Si CLV supérieure à la moyenne globale
            strategy['objectifs'].append("Maximiser la valeur à vie")
            strategy['actions_specifiques'].append("Programme de fidélité premium")
            strategy['canaux_prioritaires'].extend(['Email personnalisé', 'Téléphone', 'Événements exclusifs'])
            strategy['budget_alloue'] = metrics['avg_clv'] * 0.2  # 20% de la CLV moyenne du segment
        else:
            strategy['objectifs'].append("Augmenter la fréquence d'achat")
            strategy['actions_specifiques'].append("Offres de réduction sur le prochain achat")
            strategy['canaux_prioritaires'].extend(['Emailing', 'Réseaux sociaux', 'Publicité ciblée'])
            strategy['budget_alloue'] = metrics['avg_clv'] * 0.1  # 10% de la CLV moyenne du segment
        
        # Stratégie basée sur la fréquence d'achat
        if metrics.get('avg_purchase_freq', 0) < 1:  # Si fréquence d'achat inférieure à 1 par mois
            strategy['objectifs'].append("Augmenter la fréquence d'achat")
            strategy['actions_specifiques'].append("Programme de fidélité avec récompenses")
            strategy['kpis'].append("Augmenter la fréquence d'achat de 30% sur 6 mois")
        
        # Stratégie basée sur le score RFM
        if metrics.get('rfm_score', 0) < 5:  # Si score RFM bas
            strategy['objectifs'].append("Relancer les clients inactifs")
            strategy['actions_specifiques'].append("Campagne de relance personnalisée")
            strategy['kpis'].append("Taux de réactivation de 15% dans les 3 mois")
        
        strategies[segment_id] = strategy
    
    return strategies

def generate_marketing_report(segment_analysis, strategies):
    """Génère un rapport détaillé de la stratégie marketing."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stratégie Marketing Digitale - Rapport</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 30px; }
            .section { margin-bottom: 40px; }
            .segment-card { 
                border: 1px solid #ddd; 
                border-radius: 5px; 
                padding: 20px; 
                margin-bottom: 20px;
                background-color: #f9f9f9;
            }
            .metrics { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }
            .metric-card { 
                background: white; 
                border-left: 4px solid #3498db; 
                padding: 10px 15px; 
                flex: 1; 
                min-width: 150px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .metric-card h4 { margin: 0 0 5px 0; color: #7f8c8d; }
            .metric-value { font-size: 18px; font-weight: bold; color: #2c3e50; }
            .strategy-details { margin-top: 15px; }
            .strategy-details h4 { margin-bottom: 5px; }
            ul { margin-top: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Stratégie Marketing Digitale</h1>
                <p>Basée sur l'analyse des segments clients et des prédictions de churn</p>
            </div>
    """
    
    # Ajouter une section pour chaque segment (Cluster)
    for cluster_id, metrics in segment_analysis.items():
        strategy = strategies.get(cluster_id, {})
        
        # Créer la section du segment
        segment_section = f"""
            <div class="section">
                <h2>Segment {cluster_id}</h2>
                <div class="segment-card">
                    <h3>Caractéristiques du Segment</h3>
                    <div class="metrics">
                        <div class="metric-card">
                            <h4>Nombre de clients</h4>
                            <div class="metric-value">{metrics['count']}</div>
                        </div>
                        <div class="metric-card">
                            <h4>CLV Moyenne</h4>
                            <div class="metric-value">{metrics['avg_clv']:,.2f} €</div>
                        </div>
                        <div class="metric-card">
                            <h4>Fréquence d'achat</h4>
                            <div class="metric-value">{metrics['avg_purchase_freq']:.2f}/mois</div>
                        </div>
                        <div class="metric-card">
                            <h4>Panier moyen</h4>
                            <div class="metric-value">{metrics['avg_order_value']:,.2f} €</div>
                        </div>
        """
        
        # Ajouter la métrique de churn si disponible
        if 'churn_rate' in metrics:
            segment_section += f"""
                        <div class="metric-card">
                            <h4>Risque de churn</h4>
                            <div class="metric-value">{churn_rate:.1%}</div>
                        </div>
            """.format(churn_rate=metrics['churn_rate'])
            
        segment_section += """
                    </div>
                    
                    <div class="strategy-details">
                        <h3>Stratégie Recommandée</h3>
                        <h4>Objectifs</h4>
                        <ul>
        """
        
        # Ajouter les objectifs
        segment_section += "".join([f"<li>{obj}</li>" for obj in strategy.get('objectifs', [])])
        
        segment_section += """
                        </ul>
                        
                        <h4>Canaux Prioritaires</h4>
                        <ul>
        """
        
        # Ajouter les canaux prioritaires
        segment_section += "".join([f"<li>{canal}</li>" for canal in strategy.get('canaux_prioritaires', [])])
        
        segment_section += """
                        </ul>
                        
                        <h4>Actions Spécifiques</h4>
                        <ul>
        """
        
        # Ajouter les actions spécifiques
        segment_section += "".join([f"<li>{action}</li>" for action in strategy.get('actions_specifiques', [])])
        
        # Ajouter le budget et les KPIs
        segment_section += f"""
                        </ul>
                        
                        <h4>Budget Alloué</h4>
                        <p>{strategy['budget_alloue']:,.2f} € par client</p>
                        
                        <h4>KPIs de Succès</h4>
                        <ul>
        """.format(budget=strategy.get('budget_alloue', 0))
        
        # Ajouter les KPIs
        for kpi in strategy.get('kpis', ["Augmentation de la CLV de 15% sur 6 mois"]):
            html_content += f"<li>{kpi}</li>"
        
        html_content += """
                        </ul>
                    </div>
                </div>
            </div>
        """
    
    # Ajouter une conclusion
    html_content += """
            <div class="section">
                <h2>Recommandations Générales</h2>
                <div class="segment-card">
                    <h3>Stratégie Globale</h3>
                    <p>Basée sur l'analyse des segments, voici les recommandations globales :</p>
                    <ul>
                        <li>Mettre en place un programme de fidélité personnalisé pour chaque segment</li>
                        <li>Développer des campagnes de remarketing ciblées pour les segments à fort potentiel</li>
                        <li>Créer des parcours clients personnalisés en fonction du cycle de vie du client</li>
                        <li>Mettre en place un système d'alerte pour les clients à haut risque de churn</li>
                        <li>Optimiser le budget marketing en allouant plus de ressources aux segments les plus rentables</li>
                    </ul>
                    
                    <h3>Prochaines Étapes</h3>
                    <ol>
                        <li>Valider les stratégies avec les équipes concernées</li>
                        <li>Définir un calendrier de mise en œuvre</li>
                        <li>Mettre en place un système de suivi des performances</li>
                        <li>Réviser et ajuster les stratégies tous les trimestres</li>
                    </ol>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Écrire le rapport dans un fichier
    report_path = REPORTS_DIR / 'digital_marketing_strategy.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return report_path

def main():
    print("Démarrage de l'élaboration de la stratégie marketing digitale...")
    
    # 1. Charger les données des segments
    print("Chargement des données des segments clients...")
    segments_df = load_segment_data()
    
    if segments_df is None or segments_df.empty:
        print("Erreur: Impossible de charger les données des segments clients.")
        return
    
    # 2. Analyser les caractéristiques des segments
    print("Analyse des caractéristiques des segments...")
    segment_analysis = analyze_segment_characteristics(segments_df)
    
    if not segment_analysis:
        print("Erreur: Impossible d'analyser les segments.")
        return
    
    # 3. Générer des stratégies personnalisées
    print("Génération des stratégies par segment...")
    strategies = generate_segment_strategies(segment_analysis)
    
    # 4. Générer le rapport
    print("Génération du rapport de stratégie marketing...")
    report_path = generate_marketing_report(segment_analysis, strategies)
    
    print(f"\nStratégie marketing digitale élaborée avec succès!")
    print(f"Rapport généré : {report_path}")
    print("\nProchaines étapes recommandées :")
    print("1. Examiner le rapport HTML généré dans le dossier 'reports'")
    print("2. Présenter la stratégie aux parties prenantes")
    print("3. Mettre en œuvre les actions recommandées")
    print("4. Suivre les indicateurs de performance clés")

if __name__ == "__main__":
    main()
