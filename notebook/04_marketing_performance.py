# ========================
# Module M5 - Analyse des Performances Marketing
# Projet Marketing Analytics
# ========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Configuration des styles
plt.style.use('seaborn-v0_8')  # Utilisation du style seaborn compatible
sns.set_palette('viridis')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)

# ========================
# 1. Chargement des données
# ========================

def load_marketing_data():
    """Charge les données nécessaires pour l'analyse marketing."""
    try:
        # Définition des chemins des fichiers
        data_dir = Path(__file__).parent.parent / 'data'
        processed_dir = Path(__file__).parent.parent / 'data_processed'
        
        # Chargement des données marketing
        marketing = pd.read_csv(data_dir / 'marketing.csv', sep=';', encoding='utf-8')
        
        # Chargement des segments de clients
        segments = pd.read_csv(processed_dir / 'customer_segments.csv', index_col=0)
        
        # Chargement des données de commandes
        orders = pd.read_csv(data_dir / 'orders.csv', sep=';', encoding='utf-8')
        
        # Conversion des dates
        marketing['date'] = pd.to_datetime(marketing['date'], dayfirst=True)
        orders['order_date'] = pd.to_datetime(orders['order_date'], dayfirst=True)
        
        return marketing, segments, orders
    
    except FileNotFoundError as e:
        print(f"Erreur lors du chargement des fichiers : {e}")
        return None, None, None

# ========================
# 2. Analyse des indicateurs clés
# ========================

def calculate_kpis(marketing, orders, segments):
    """Calcule les indicateurs clés de performance marketing."""
    # Agrégation des indicateurs par campagne
    campaign_metrics = marketing.groupby('campaign_id').agg({
        'impressions': 'sum',
        'clicks': 'sum',
        'spend': 'sum',
        'conversions': 'sum',
        'revenue': 'sum'
    }).reset_index()
    
    # Calcul des KPIs
    campaign_metrics['CTR'] = (campaign_metrics['clicks'] / campaign_metrics['impressions']) * 100  # en %
    campaign_metrics['CPC'] = campaign_metrics['spend'] / campaign_metrics['clicks']  # Coût par clic
    campaign_metrics['CPM'] = (campaign_metrics['spend'] / campaign_metrics['impressions']) * 1000  # Coût pour 1000 impressions
    campaign_metrics['conversion_rate'] = (campaign_metrics['conversions'] / campaign_metrics['clicks']) * 100  # en %
    
    # Calcul du ROI (Retour sur Investissement)
    campaign_metrics['ROI'] = (
        (campaign_metrics['revenue'] - campaign_metrics['spend']) / 
        campaign_metrics['spend']
    ) * 100  # en %
    
    return campaign_metrics

# ========================
# 3. Analyse par segment client
# ========================

def analyze_segment_performance(orders, segments, marketing):
    """Analyse les performances par segment client."""
    # Fusion des commandes avec les segments
    orders_segments = orders.merge(
        segments[['Cluster']],
        left_on='customer_id',
        right_index=True,
        how='left'
    )
    
    # Agrégation des commandes par segment
    segment_metrics = orders_segments.groupby('Cluster').agg({
        'order_id': 'count',
        'price': 'sum'
    }).rename(columns={
        'order_id': 'orders',
        'price': 'revenue'
    })
    
    # Calcul du panier moyen
    segment_metrics['avg_order_value'] = segment_metrics['revenue'] / segment_metrics['orders']
    
    # Calcul du taux de conversion (approximatif)
    # Nombre total de clients uniques par segment
    unique_customers = orders_segments.groupby('Cluster')['customer_id'].nunique()
    segment_metrics['conversion_rate'] = (
        segment_metrics['orders'] / unique_customers
    ) * 100  # en %
    
    # Fusion avec les données de dépenses par segment (approximation)
    # Ici, nous supposons que les dépenses sont réparties uniformément entre les segments
    # Dans un cas réel, il faudrait avoir le tracking des campagnes par segment
    total_customers = segments.shape[0]
    segment_metrics['customers'] = segments['Cluster'].value_counts().sort_index()
    segment_metrics['spend'] = (
        marketing['spend'].sum() * 
        (segment_metrics['customers'] / total_customers)
    )
    
    # Calcul du ROI par segment
    segment_metrics['ROI'] = (
        (segment_metrics['revenue'] - segment_metrics['spend']) / 
        segment_metrics['spend']
    ) * 100  # en %
    
    return segment_metrics

# ========================
# 4. Analyse temporelle
# ========================

def analyze_temporal_trends(marketing, orders):
    """Analyse les tendances temporelles des performances marketing."""
    # Agrégation par jour
    daily_marketing = marketing.groupby('date').agg({
        'impressions': 'sum',
        'clicks': 'sum',
        'spend': 'sum',
        'conversions': 'sum',
        'revenue': 'sum'
    })
    
    # Calcul des indicateurs journaliers
    daily_marketing['CTR'] = (daily_marketing['clicks'] / daily_marketing['impressions']) * 100
    daily_marketing['CPC'] = daily_marketing['spend'] / daily_marketing['clicks']
    
    # Agrégation des commandes par jour
    daily_orders = orders.groupby('order_date').agg({
        'order_id': 'count',
        'price': 'sum'
    }).rename(columns={
        'order_id': 'orders',
        'price': 'revenue'
    })
    
    # Renommer les colonnes de revenus pour éviter les conflits
    daily_marketing = daily_marketing.rename(columns={'revenue': 'campaign_revenue'})
    daily_orders = daily_orders.rename(columns={'revenue': 'sales_revenue'})
    
    # Fusion des données marketing et ventes
    daily_data = daily_marketing.join(daily_orders, how='outer').sort_index()
    
    # Calcul du revenu total (campagne + ventes)
    daily_data['total_revenue'] = daily_data[['campaign_revenue', 'sales_revenue']].sum(axis=1)
    
    return daily_data

# ========================
# 5. Visualisation des résultats
# ========================

def plot_campaign_performance(campaign_metrics):
    """Crée des visualisations pour les performances des campagnes."""
    # Création d'une figure avec sous-graphiques
    fig = make_subplots(
        rows=2, 
        cols=2,
        subplot_titles=(
            'Taux de clics (CTR) par campagne',
            'Coût par acquisition (CPA)',
            'Taux de conversion',
            'Retour sur investissement (ROI)'
        )
    )
    
    # 1. Taux de clics (CTR)
    fig.add_trace(
        go.Bar(
            x=campaign_metrics['campaign_id'],
            y=campaign_metrics['CTR'],
            name='CTR (%)',
            marker_color='#1f77b4'
        ),
        row=1, col=1
    )
    
    # 2. Coût par acquisition (CPA)
    fig.add_trace(
        go.Bar(
            x=campaign_metrics['campaign_id'],
            y=campaign_metrics['CPC'],
            name='CPC (€)',
            marker_color='#ff7f0e'
        ),
        row=1, col=2
    )
    
    # 3. Taux de conversion
    fig.add_trace(
        go.Bar(
            x=campaign_metrics['campaign_id'],
            y=campaign_metrics['conversion_rate'],
            name='Taux de conversion (%)',
            marker_color='#2ca02c'
        ),
        row=2, col=1
    )
    
    # 4. Retour sur investissement (ROI)
    fig.add_trace(
        go.Bar(
            x=campaign_metrics['campaign_id'],
            y=campaign_metrics['ROI'],
            name='ROI (%)',
            marker_color='#d62728'
        ),
        row=2, col=2
    )
    
    # Mise en page
    fig.update_layout(
        title_text='Performances des Campagnes Marketing',
        height=800,
        showlegend=False
    )
    
    # Ajout des étiquettes d'axe
    fig.update_xaxes(title_text="Campagne", row=1, col=1)
    fig.update_xaxes(title_text="Campagne", row=1, col=2)
    fig.update_xaxes(title_text="Campagne", row=2, col=1)
    fig.update_xaxes(title_text="Campagne", row=2, col=2)
    
    fig.update_yaxes(title_text="CTR (%)", row=1, col=1)
    fig.update_yaxes(title_text="CPC (€)", row=1, col=2)
    fig.update_yaxes(title_text="Taux de conversion (%)", row=2, col=1)
    fig.update_yaxes(title_text="ROI (%)", row=2, col=2)
    
    # Création du dossier reports s'il n'existe pas
    reports_dir = Path(__file__).parent.parent / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    # Enregistrement du graphique
    output_path = reports_dir / 'campaign_performance.html'
    fig.write_html(str(output_path))
    
    return fig

def plot_segment_performance(segment_metrics):
    """Crée des visualisations pour les performances par segment."""
    # Création d'une figure avec sous-graphiques
    fig = make_subplots(
        rows=2, 
        cols=2,
        subplot_titles=(
            'Revenu par segment',
            'Panier moyen par segment',
            'Taux de conversion par segment',
            'ROI par segment'
        )
    )
    
    # 1. Revenu par segment
    fig.add_trace(
        go.Bar(
            x=segment_metrics.index,
            y=segment_metrics['revenue'],
            name='Revenu (€)',
            marker_color='#1f77b4'
        ),
        row=1, col=1
    )
    
    # 2. Panier moyen par segment
    fig.add_trace(
        go.Bar(
            x=segment_metrics.index,
            y=segment_metrics['avg_order_value'],
            name='Panier moyen (€)',
            marker_color='#ff7f0e'
        ),
        row=1, col=2
    )
    
    # 3. Taux de conversion par segment
    fig.add_trace(
        go.Bar(
            x=segment_metrics.index,
            y=segment_metrics['conversion_rate'],
            name='Taux de conversion (%)',
            marker_color='#2ca02c'
        ),
        row=2, col=1
    )
    
    # 4. ROI par segment
    fig.add_trace(
        go.Bar(
            x=segment_metrics.index,
            y=segment_metrics['ROI'],
            name='ROI (%)',
            marker_color='#d62728'
        ),
        row=2, col=2
    )
    
    # Mise en page
    fig.update_layout(
        title_text='Performances par Segment Client',
        height=800,
        showlegend=False
    )
    
    # Ajout des étiquettes d'axe
    fig.update_xaxes(title_text="Segment", row=1, col=1)
    fig.update_xaxes(title_text="Segment", row=1, col=2)
    fig.update_xaxes(title_text="Segment", row=2, col=1)
    fig.update_xaxes(title_text="Segment", row=2, col=2)
    
    fig.update_yaxes(title_text="Revenu (€)", row=1, col=1)
    fig.update_yaxes(title_text="Panier moyen (€)", row=1, col=2)
    fig.update_yaxes(title_text="Taux de conversion (%)", row=2, col=1)
    fig.update_yaxes(title_text="ROI (%)", row=2, col=2)
    
    # Création du dossier reports s'il n'existe pas
    reports_dir = Path(__file__).parent.parent / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    # Enregistrement du graphique
    output_path = reports_dir / 'segment_performance.html'
    fig.write_html(str(output_path))
    
    return fig

def plot_temporal_trends(daily_data):
    """Crée des visualisations pour les tendances temporelles."""
    # Création d'une figure avec sous-graphiques
    fig = make_subplots(
        rows=3, 
        cols=1,
        subplot_titles=(
            'Trafic et engagement',
            'Coûts et dépenses',
            'Conversions et revenus'
        ),
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    # 1. Trafic et engagement
    fig.add_trace(
        go.Scatter(
            x=daily_data.index,
            y=daily_data['impressions'],
            name='Impressions',
            line=dict(color='#1f77b4')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily_data.index,
            y=daily_data['clicks'],
            name='Clics',
            line=dict(color='#ff7f0e')
        ),
        row=1, col=1
    )
    
    # 2. Coûts et dépenses
    fig.add_trace(
        go.Scatter(
            x=daily_data.index,
            y=daily_data['spend'],
            name='Dépenses (€)',
            line=dict(color='#d62728')
        ),
        row=2, col=1
    )
    
    # 3. Conversions et revenus
    fig.add_trace(
        go.Bar(
            x=daily_data.index,
            y=daily_data['conversions'],
            name='Conversions',
            marker_color='#2ca02c',
            opacity=0.6
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily_data.index,
            y=daily_data['total_revenue'],
            name='Revenu total (€)',
            line=dict(color='#9467bd')
        ),
        row=3, col=1
    )
    
    # Mise en page
    fig.update_layout(
        title_text='Tendances Temporelles des Performances Marketing',
        height=900,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Mise en forme des axes
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    fig.update_yaxes(title_text="Volume", row=1, col=1, type="log")
    fig.update_yaxes(title_text="Coût (€)", row=2, col=1)
    fig.update_yaxes(title_text="Montant", row=3, col=1)
    
    # Création du dossier reports s'il n'existe pas
    reports_dir = Path(__file__).parent.parent / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    # Enregistrement du graphique
    output_path = reports_dir / 'temporal_trends.html'
    fig.write_html(str(output_path))
    
    return fig

# ========================
# 6. Génération du rapport
# ========================

def generate_marketing_report(campaign_metrics, segment_metrics, daily_data):
    """Génère un rapport complet d'analyse des performances marketing."""
    # Création du rapport HTML
    report = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rapport d'Analyse des Performances Marketing</title>
        <style>
            body {{{{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}}}
            .container {{{{ max-width: 1200px; margin: 0 auto; }}}}
            h1, h2, h3 {{{{ color: #2c3e50; }}}}
            .section {{{{ margin-bottom: 40px; }}}}
            .chart {{{{ margin: 20px 0; }}}}
            table {{{{ width: 100%; border-collapse: collapse; margin: 20px 0; }}}}
            th, td {{{{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}}}
            th {{{{ background-color: #f2f2f2; }}}}
            .highlight {{{{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}}}
            .kpi-container {{{{ display: flex; justify-content: space-between; margin: 20px 0; }}}}
            .kpi {{{{ text-align: center; padding: 15px; background: #f8f9fa; border-radius: 5px; flex: 1; margin: 0 5px; }}}}
            .kpi-value {{{{ font-size: 24px; font-weight: bold; color: #2c3e50; }}}}
            .kpi-label {{{{ font-size: 14px; color: #7f8c8d; }}}}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Rapport d'Analyse des Performances Marketing</h1>
            <p>Date de génération: {date}</p>
            
            <div class="section">
                <h2>1. Vue d'ensemble</h2>
                <p>Ce rapport présente une analyse détaillée des performances des campagnes marketing et de leur impact sur les différents segments de clients.</p>
                
                <div class="kpi-container">
                    <div class="kpi">
                        <div class="kpi-value">{total_impressions:,.0f}</div>
                        <div class="kpi-label">Impressions totales</div>
                    </div>
                    <div class="kpi">
                        <div class="kpi-value">{total_clicks:,.0f}</div>
                        <div class="kpi-label">Clics totaux</div>
                    </div>
                    <div class="kpi">
                        <div class="kpi-value">{avg_ctr:.2f}%</div>
                        <div class="kpi-label">Taux de clics moyen</div>
                    </div>
                    <div class="kpi">
                        <div class="kpi-value">{total_conversions:,.0f}</div>
                        <div class="kpi-label">Conversions totales</div>
                    </div>
                    <div class="kpi">
                        <div class="kpi-value">{total_revenue:,.0f} €</div>
                        <div class="kpi-label">Revenu total</div>
                    </div>
                    <div class="kpi">
                        <div class="kpi-value" style="color: {roi_color};">{avg_roi:.1f}%</div>
                        <div class="kpi-label">ROI moyen</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>2. Performances par Campagne</h2>
                <div class="chart">
                    <iframe src="campaign_performance.html" width="100%" height="800px"></iframe>
                </div>
                <div class="highlight">
                    <h3>Tableau de bord des campagnes</h3>
                    {campaign_table}
                </div>
            </div>
            
            <div class="section">
                <h2>3. Analyse par Segment Client</h2>
                <div class="chart">
                    <iframe src="segment_performance.html" width="100%" height="800px"></iframe>
                </div>
                <div class="highlight">
                    <h3>Tableau de bord des segments</h3>
                    {segment_table}
                </div>
            </div>
            
            <div class="section">
                <h2>4. Tendances Temporelles</h2>
                <div class="chart">
                    <iframe src="temporal_trends.html" width="100%" height="900px"></iframe>
                </div>
            </div>
            
            <div class="section">
                <h2>5. Recommandations Stratégiques</h2>
                <h3>Opportunités d'optimisation</h3>
                <ul>
                    {recommendations}
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Préparation des données pour le rapport
    from datetime import datetime
    
    # Calcul des indicateurs clés
    total_impressions = campaign_metrics['impressions'].sum()
    total_clicks = campaign_metrics['clicks'].sum()
    total_conversions = campaign_metrics['conversions'].sum()
    total_revenue = campaign_metrics['revenue'].sum()
    total_spend = campaign_metrics['spend'].sum()
    
    avg_ctr = (total_clicks / total_impressions) * 100 if total_impressions > 0 else 0
    avg_roi = ((total_revenue - total_spend) / total_spend) * 100 if total_spend > 0 else 0
    
    # Détermination de la couleur du ROI
    roi_color = '#2ecc71' if avg_roi >= 0 else '#e74c3c'
    
    # Formatage des tableaux
    campaign_table = campaign_metrics[[
        'campaign_id', 'impressions', 'clicks', 'CTR', 'CPC', 
        'conversion_rate', 'spend', 'revenue', 'ROI'
    ]].round(2).to_html(index=False)
    
    segment_table = segment_metrics[[
        'customers', 'orders', 'revenue', 'avg_order_value', 
        'conversion_rate', 'spend', 'ROI'
    ]].round(2).to_html()
    
    # Génération des recommandations
    recommendations = ""
    
    # Recommandation basée sur le ROI moyen
    if avg_roi < 100:
        recommendations += """
        <li><strong>Optimisation des dépenses publicitaires :</strong> 
        Le ROI moyen des campagnes est de {:.1f}%, ce qui indique une marge d'amélioration. 
        Envisagez de réallouer le budget des campagnes les moins performantes vers celles qui génèrent le meilleur retour sur investissement.
        </li>
        """.format(avg_roi)
    
    # Recommandation basée sur le CTR moyen
    if avg_ctr < 2.0:  # Seuil arbitraire pour un bon CTR
        recommendations += """
        <li><strong>Amélioration du taux de clics :</strong> 
        Le taux de clics moyen est de {:.2f}%, ce qui est en dessous de la moyenne du secteur. 
        Testez différents appels à l'action, titres et images pour améliorer l'engagement.
        </li>
        """.format(avg_ctr)
    
    # Recommandation basée sur les segments
    best_segment = segment_metrics['ROI'].idxmax()
    worst_segment = segment_metrics['ROI'].idxmin()
    
    recommendations += """
    <li><strong>Ciblage des segments performants :</strong> 
    Le segment {0} présente le meilleur ROI ({1:.1f}%). 
    Envisagez d'augmenter le ciblage et le budget alloué à ce segment.
    </li>
    """.format(best_segment, segment_metrics.loc[best_segment, 'ROI'])
    
    recommendations += """
    <li><strong>Réévaluation des segments sous-performants :</strong> 
    Le segment {0} présente le ROI le plus faible ({1:.1f}%). 
    Analysez les raisons de cette sous-performance et envisagez des stratégies spécifiques pour améliorer la conversion ou réduire les coûts d'acquisition.
    </li>
    """.format(worst_segment, segment_metrics.loc[worst_segment, 'ROI'])
    
    # Recommandation basée sur la saisonnalité
    if 'revenue' in daily_data.columns and len(daily_data) > 30:
        # Vérifier si les 7 derniers jours sont supérieurs à la moyenne
        last_week = daily_data['revenue'].tail(7).mean()
        avg_revenue = daily_data['revenue'].mean()
        
        if last_week > avg_revenue * 1.2:  # 20% de plus que la moyenne
            recommendations += """
            <li><strong>Capitalisation sur la tendance positive :</strong> 
            Les ventes des 7 derniers jours sont {:.1f}% supérieures à la moyenne. 
            Envisagez d'augmenter les dépenses publicitaires pour capitaliser sur cette tendance positive.
            </li>
            """.format(((last_week / avg_revenue) - 1) * 100)
    
    # Remplissage du modèle
    report = report.format(
        date=datetime.now().strftime("%d/%m/%Y %H:%M"),
        total_impressions=total_impressions,
        total_clicks=total_clicks,
        total_conversions=total_conversions,
        total_revenue=total_revenue,
        total_cost=total_spend,  # Utilisation de total_spend au lieu de total_cost
        avg_ctr=avg_ctr,
        avg_roi=avg_roi,
        roi_color=roi_color,
        campaign_table=campaign_table,
        segment_table=segment_table,
        recommendations=recommendations
    )
    
    # Création du dossier reports s'il n'existe pas
    reports_dir = Path(__file__).parent.parent / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    # Sauvegarde du rapport
    report_path = reports_dir / 'marketing_performance_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report

# ========================
# 7. Fonction principale
# ========================

def main():
    print("Début de l'analyse des performances marketing...")
    
    # 1. Chargement des données
    print("\nChargement des données...")
    marketing, segments, orders = load_marketing_data()
    
    if marketing is None:
        return
    
    # 2. Calcul des indicateurs clés
    print("\nCalcul des indicateurs clés...")
    campaign_metrics = calculate_kpis(marketing, orders, segments)
    
    # 3. Analyse par segment
    print("\nAnalyse par segment...")
    segment_metrics = analyze_segment_performance(orders, segments, marketing)
    
    # 4. Analyse temporelle
    print("\nAnalyse des tendances temporelles...")
    daily_data = analyze_temporal_trends(marketing, orders)
    
    # 5. Génération des visualisations
    print("\nGénération des visualisations...")
    plot_campaign_performance(campaign_metrics)
    plot_segment_performance(segment_metrics)
    plot_temporal_trends(daily_data)
    
    # 6. Génération du rapport
    print("\nGénération du rapport...")
    report = generate_marketing_report(campaign_metrics, segment_metrics, daily_data)
    
    print("\nAnalyse des performances marketing terminée avec succès!")
    print("Rapport généré : ../reports/marketing_performance_report.html")

if __name__ == "__main__":
    main()
