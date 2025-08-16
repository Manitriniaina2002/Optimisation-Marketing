"""
Page d'analyse des performances marketing pour l'application d'analyse marketing.
Permet d'analyser l'efficacité des campagnes marketing et le retour sur investissement.
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

from utils.visualization import display_metrics, plot_distribution, plot_time_series
from utils.analysis import MarketingAnalyzer
from config import APP_CONFIG

# Configuration de la page
st.set_page_config(
    page_title=f"{APP_CONFIG['app_name']} - Analyse des performances",
    page_icon="📈",
    layout="wide"
)

# Titre de la page
st.title("📈 Analyse des performances marketing")
st.markdown("---")

# Vérifier que les données sont chargées
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Veuillez d'abord importer et valider les données dans l'onglet 'Importation des données'.")
    st.stop()

# Récupérer les données de la session
marketing_df = st.session_state.get('marketing_df')
orders_df = st.session_state.get('orders_df')
customers_df = st.session_state.get('customers_df')

# Vérifier que les données nécessaires sont disponibles
if marketing_df is None:
    st.error("Les données marketing sont manquantes. Veuillez importer les données nécessaires.")
    st.stop()

# Initialiser l'analyseur marketing
if 'marketing_analyzer' not in st.session_state:
    st.session_state.marketing_analyzer = MarketingAnalyzer({
        'marketing': marketing_df,
        'orders': orders_df if orders_df is not None else pd.DataFrame(),
        'customers': customers_df if customers_df is not None else pd.DataFrame()
    })

# Section d'aperçu des performances globales
with st.expander("📊 Aperçu des performances globales", expanded=True):
    st.markdown("""
    Cette section fournit une vue d'ensemble des performances marketing globales, 
    y compris les indicateurs clés de performance (KPI) et les tendances.
    """)
    
    # Vérifier si les colonnes nécessaires sont présentes
    required_columns = ['spend']
    missing_columns = [col for col in required_columns if col not in marketing_df.columns]
    
    if missing_columns:
        st.error(f"Colonnes manquantes dans les données marketing : {', '.join(missing_columns)}")
    else:
        # Calculer les KPI globaux
        total_spend = marketing_df['spend'].sum()
        total_campaigns = len(marketing_df)
        
        # Calculer le ROI si les données de revenus sont disponibles
        if 'revenue' in marketing_df.columns:
            total_revenue = marketing_df['revenue'].sum()
            total_roi = ((total_revenue - total_spend) / total_spend) * 100 if total_spend > 0 else 0
        else:
            total_revenue = None
            total_roi = None
        
        # Afficher les KPI dans des colonnes
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Budget total dépensé", f"{total_spend:,.2f} €")
        
        with col2:
            st.metric("Nombre de campagnes", total_campaigns)
        
        with col3:
            if total_revenue is not None:
                st.metric("Revenu total généré", f"{total_revenue:,.2f} €")
        
        with col4:
            if total_roi is not None:
                st.metric("ROI global", f"{total_roi:.1f}%")
        
        # Afficher l'évolution des dépenses dans le temps si une colonne de date est disponible
        if 'start_date' in marketing_df.columns:
            # Convertir en datetime si nécessaire
            if not pd.api.types.is_datetime64_any_dtype(marketing_df['start_date']):
                marketing_df['start_date'] = pd.to_datetime(marketing_df['start_date'])
            
            # Grouper par période (mois par défaut)
            marketing_df['month'] = marketing_df['start_date'].dt.to_period('M').dt.to_timestamp()
            monthly_spend = marketing_df.groupby('month')['spend'].sum().reset_index()
            
            # Créer le graphique d'évolution des dépenses
            fig = px.line(
                monthly_spend,
                x='month',
                y='spend',
                title="Évolution des dépenses marketing par mois",
                labels={'month': 'Mois', 'spend': 'Dépenses (€)'},
                markers=True
            )
            
            # Ajouter une ligne de tendance
            if len(monthly_spend) > 1:
                z = np.polyfit(range(len(monthly_spend)), monthly_spend['spend'], 1)
                p = np.poly1d(z)
                fig.add_scatter(
                    x=monthly_spend['month'],
                    y=p(range(len(monthly_spend))),
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Tendance'
                )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Afficher la répartition des dépenses par canal si disponible
        if 'channel' in marketing_df.columns:
            st.subheader("Répartition des dépenses par canal")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Graphique en camembert
                channel_spend = marketing_df.groupby('channel')['spend'].sum().reset_index()
                fig_pie = px.pie(
                    channel_spend,
                    values='spend',
                    names='channel',
                    title="Répartition des dépenses par canal",
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Tableau des dépenses par canal
                channel_spend['percentage'] = (channel_spend['spend'] / channel_spend['spend'].sum()) * 100
                channel_spend = channel_spend.sort_values('spend', ascending=False)
                
                st.markdown("**Dépenses par canal**")
                for _, row in channel_spend.iterrows():
                    st.metric(
                        label=row['channel'],
                        value=f"{row['spend']:,.2f} €",
                        delta=f"{row['percentage']:.1f}% du total"
                    )

# Section d'analyse détaillée des campagnes
with st.expander("🔍 Analyse détaillée des campagnes", expanded=True):
    st.markdown("""
    Analysez les performances détaillées de chaque campagne marketing et comparez-les 
    pour identifier les meilleures pratiques et opportunités d'optimisation.
    """)
    
    # Sélectionner les colonnes à afficher dans le tableau
    default_columns = ['campaign_name', 'channel', 'start_date', 'end_date', 'spend']
    available_columns = [col for col in default_columns if col in marketing_df.columns]
    
    # Ajouter des métriques de performance si disponibles
    performance_metrics = ['impressions', 'clicks', 'conversions', 'revenue', 'roi', 'cpa', 'ctr', 'conversion_rate']
    available_metrics = [col for col in performance_metrics if col in marketing_df.columns]
    
    # Sélection des colonnes à afficher
    selected_columns = st.multiselect(
        "Colonnes à afficher",
        options=available_columns + available_metrics,
        default=available_columns + available_metrics[:3],
        key="campaign_columns"
    )
    
    # Filtrer les données
    filtered_df = marketing_df[selected_columns].copy()
    
    # Trier par date si disponible
    if 'start_date' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('start_date', ascending=False)
    
    # Afficher le tableau avec les performances
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400
    )
    
    # Télécharger les données
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Télécharger les données au format CSV",
        data=csv,
        file_name="performances_campagnes.csv",
        mime="text/csv"
    )
    
    # Analyse comparative des canaux
    st.subheader("Comparaison des canaux marketing")
    
    if 'channel' in marketing_df.columns and len(available_metrics) > 0:
        # Sélectionner la métrique pour la comparaison
        metric_comparison = st.selectbox(
            "Métrique pour la comparaison",
            available_metrics,
            key="metric_comparison"
        )
        
        # Calculer les agrégations par canal
        channel_comparison = marketing_df.groupby('channel').agg({
            'spend': 'sum',
            metric_comparison: ['sum', 'mean', 'count']
        }).reset_index()
        
        # Aplatir les colonnes multi-niveaux
        channel_comparison.columns = ['_'.join(col).strip('_') for col in channel_comparison.columns.values]
        
        # Calculer le coût par métrique (si pertinent)
        if metric_comparison + '_sum' in channel_comparison.columns:
            channel_comparison[f'cost_per_{metric_comparison}'] = (
                channel_comparison['spend_sum'] / channel_comparison[f'{metric_comparison}_sum']
            )
        
        # Afficher le tableau de comparaison
        st.dataframe(
            channel_comparison,
            use_container_width=True,
            height=300
        )
        
        # Graphique de comparaison
        fig_comparison = px.bar(
            channel_comparison,
            x='channel',
            y=f'{metric_comparison}_sum',
            title=f"Comparaison des canaux par {metric_comparison}",
            labels={'channel': 'Canal', f'{metric_comparison}_sum': metric_comparison.capitalize()},
            color='channel'
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)

# Section d'analyse du retour sur investissement (ROI)
if 'revenue' in marketing_df.columns and 'spend' in marketing_df.columns:
    with st.expander("💰 Analyse du retour sur investissement (ROI)", expanded=True):
        st.markdown("""
        Analysez le retour sur investissement (ROI) de vos campagnes marketing pour identifier 
        les canaux et les stratégies les plus rentables.
        """)
        
        # Calculer le ROI par campagne
        marketing_df['roi'] = ((marketing_df['revenue'] - marketing_df['spend']) / marketing_df['spend']) * 100
        
        # Afficher le ROI moyen par canal
        if 'channel' in marketing_df.columns:
            roi_by_channel = marketing_df.groupby('channel').agg({
                'spend': 'sum',
                'revenue': 'sum',
                'roi': 'mean'
            }).reset_index()
            
            # Calculer le ROI global par canal
            roi_by_channel['total_roi'] = (
                (roi_by_channel['revenue'] - roi_by_channel['spend']) / roi_by_channel['spend']
            ) * 100
            
            # Trier par ROI décroissant
            roi_by_channel = roi_by_channel.sort_values('total_roi', ascending=False)
            
            # Afficher le graphique de ROI par canal
            fig_roi = px.bar(
                roi_by_channel,
                x='channel',
                y='total_roi',
                title="ROI moyen par canal",
                labels={'channel': 'Canal', 'total_roi': 'ROI (%)'},
                text_auto='.1f',
                color='channel'
            )
            
            fig_roi.update_traces(
                texttemplate='%{text}%',
                textposition='outside'
            )
            
            st.plotly_chart(fig_roi, use_container_width=True)
            
            # Tableau détaillé du ROI par canal
            st.subheader("Détail du ROI par canal")
            
            # Calculer des métriques supplémentaires
            roi_by_channel['revenue_per_euro'] = roi_by_channel['revenue'] / roi_by_channel['spend']
            roi_by_channel['pourcentage_du_budget'] = (
                (roi_by_channel['spend'] / roi_by_channel['spend'].sum()) * 100
            )
            
            # Formater les colonnes pour l'affichage
            display_columns = {
                'channel': 'Canal',
                'spend': 'Budget (€)',
                'revenue': 'Revenu (€)',
                'total_roi': 'ROI (%)',
                'revenue_per_euro': 'Revenu par € dépensé',
                'pourcentage_du_budget': '% du budget total'
            }
            
            # Sélectionner et formater les colonnes à afficher
            display_df = roi_by_channel[list(display_columns.keys())].copy()
            display_df.columns = [display_columns[col] for col in display_df.columns]
            
            # Formater les valeurs numériques
            for col in ['Budget (€)', 'Revenu (€)']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f} €")
            
            if 'ROI (%)' in display_df.columns:
                display_df['ROI (%)'] = display_df['ROI (%)'].apply(lambda x: f"{x:.1f}%")
            
            if 'Revenu par € dépensé' in display_df.columns:
                display_df['Revenu par € dépensé'] = display_df['Revenu par € dépensé'].apply(lambda x: f"{x:.2f} €")
            
            if '% du budget total' in display_df.columns:
                display_df['% du budget total'] = display_df['% du budget total'].apply(lambda x: f"{x:.1f}%")
            
            # Afficher le tableau formaté
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Recommandations basées sur le ROI
            st.subheader("Recommandations")
            
            if not roi_by_channel.empty:
                best_channel = roi_by_channel.iloc[0]['channel']
                worst_channel = roi_by_channel.iloc[-1]['channel']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **✅ Canal le plus performant: {best_channel}**  
                    - Augmentez progressivement le budget alloué
                    - Analysez ce qui fonctionne bien pour reproduire cette réussite
                    - Testez des variations de messages créatifs
                    """)
                
                with col2:
                    st.markdown(f"""
                    **⚠️ Canal le moins performant: {worst_channel}**  
                    - Réduisez ou réallouez le budget
                    - Analysez les raisons de la sous-performance
                    - Testez de nouvelles approches créatives ou ciblages
                    """)

# Section d'analyse des tendances temporelles
with st.expander("📅 Analyse des tendances temporelles", expanded=False):
    st.markdown("""
    Analysez l'évolution des performances marketing dans le temps pour identifier 
    des tendances, des saisonnalités et des opportunités d'optimisation.
    """)
    
    if 'start_date' in marketing_df.columns:
        # Sélectionner la période d'analyse
        date_col = 'start_date'
        
        # Convertir en datetime si nécessaire
        if not pd.api.types.is_datetime64_any_dtype(marketing_df[date_col]):
            marketing_df[date_col] = pd.to_datetime(marketing_df[date_col])
        
        # Sélectionner la métrique à analyser
        metric_options = ['spend', 'impressions', 'clicks', 'conversions', 'revenue']
        available_metrics = [m for m in metric_options if m in marketing_df.columns]
        
        if available_metrics:
            selected_metric = st.selectbox(
                "Métrique à analyser",
                available_metrics,
                key="trend_metric"
            )
            
            # Agrégation par période (jour, semaine, mois)
            period = st.radio(
                "Période d'agrégation",
                ['Jour', 'Semaine', 'Mois'],
                horizontal=True,
                key="trend_period"
            )
            
            # Préparer les données pour l'analyse des tendances
            trend_df = marketing_df.copy()
            
            # Grouper par période
            if period == 'Jour':
                trend_df['period'] = trend_df[date_col].dt.date
            elif period == 'Semaine':
                trend_df['period'] = trend_df[date_col].dt.to_period('W').dt.to_timestamp()
            else:  # Mois
                trend_df['period'] = trend_df[date_col].dt.to_period('M').dt.to_timestamp()
            
            # Agrégation des données
            trend_agg = trend_df.groupby('period').agg({
                selected_metric: 'sum',
                'spend': 'sum'
            }).reset_index()
            
            # Calculer le coût par métrique si ce n'est pas déjà le coût
            if selected_metric != 'spend':
                trend_agg[f'cpm_{selected_metric}'] = (trend_agg['spend'] / trend_agg[selected_metric]) * 1000
            
            # Afficher le graphique des tendances
            fig_trend = px.line(
                trend_agg,
                x='period',
                y=selected_metric,
                title=f"Évolution des {selected_metric} par {period.lower()}",
                labels={'period': 'Période', selected_metric: selected_metric.capitalize()},
                markers=True
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Afficher le coût par métrique si pertinent
            if selected_metric != 'spend' and f'cpm_{selected_metric}' in trend_agg.columns:
                fig_cpm = px.line(
                    trend_agg,
                    x='period',
                    y=f'cpm_{selected_metric}',
                    title=f"Coût pour 1000 {selected_metric} (CPM) par {period.lower()}",
                    labels={'period': 'Période', f'cpm_{selected_metric}': f'Coût pour 1000 {selected_metric} (€)'},
                    markers=True
                )
                
                st.plotly_chart(fig_cpm, use_container_width=True)
                
                # Afficher des recommandations basées sur les tendances
                st.subheader("Analyse des tendances")
                
                # Calculer la variation sur la période
                if len(trend_agg) > 1:
                    first_value = trend_agg[selected_metric].iloc[0]
                    last_value = trend_agg[selected_metric].iloc[-1]
                    variation = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0
                    
                    # Calculer la tendance (haussière ou baissière)
                    z = np.polyfit(range(len(trend_agg)), trend_agg[selected_metric], 1)
                    trend = "à la hausse" if z[0] > 0 else "à la baisse" if z[0] < 0 else "stable"
                    
                    # Afficher l'analyse
                    st.markdown(f"""
                    - **Tendance générale** : {trend} sur la période analysée
                    - **Variation totale** : {variation:.1f}% entre {trend_agg['period'].iloc[0].strftime('%d/%m/%Y')} et {trend_agg['period'].iloc[-1].strftime('%d/%m/%Y')}
                    - **Moyenne par {period.lower()}** : {trend_agg[selected_metric].mean():.1f}
                    - **Maximum** : {trend_agg[selected_metric].max():.1f} atteint le {trend_agg.loc[trend_agg[selected_metric].idxmax(), 'period'].strftime('%d/%m/%Y')}
                    """)
                    
                    # Recommandations basées sur la tendance
                    if trend == "à la hausse":
                        st.success("""
                        **Recommandations** :
                        - Capitalisez sur cette tendance positive en maintenant ou en augmentant les investissements
                        - Analysez les facteurs contribuant à cette croissance pour les reproduire
                        - Surveillez le retour sur investissement pour éviter les rendements décroissants
                        """)
                    elif trend == "à la baisse":
                        st.warning("""
                        **Recommandations** :
                        - Identifiez les causes de cette baisse (saisonnalité, concurrence, etc.)
                        - Envisagez de réallouer le budget vers des canaux plus performants
                        - Testez de nouvelles approches créatives ou ciblages
                        """)
        else:
            st.warning("Aucune métrique disponible pour l'analyse des tendances.")
    else:
        st.warning("Les données de date sont nécessaires pour l'analyse des tendances temporelles.")

# Section d'analyse d'impact des campagnes
if orders_df is not None and 'order_date' in orders_df.columns and 'amount' in orders_df.columns:
    with st.expander("📊 Impact des campagnes sur les ventes", expanded=False):
        st.markdown("""
        Analysez l'impact des campagnes marketing sur les ventes pour évaluer leur efficacité réelle.
        """)
        
        # Convertir les dates si nécessaire
        if not pd.api.types.is_datetime64_any_dtype(orders_df['order_date']):
            orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
        
        # Agrégation des ventes par jour
        daily_sales = orders_df.groupby('order_date')['amount'].sum().reset_index()
        
        # Fusionner avec les données marketing si possible
        if 'start_date' in marketing_df.columns and 'end_date' in marketing_df.columns:
            # Créer une série temporelle avec les périodes de campagnes
            campaign_periods = []
            for _, row in marketing_df.iterrows():
                start = pd.to_datetime(row['start_date'])
                end = pd.to_datetime(row['end_date']) if pd.notnull(row['end_date']) else start + pd.Timedelta(days=1)
                campaign_periods.append((start, end, row.get('campaign_name', 'Campagne sans nom')))
            
            # Créer un graphique des ventes avec les périodes de campagnes
            fig_sales = go.Figure()
            
            # Ajouter les ventes
            fig_sales.add_trace(go.Scatter(
                x=daily_sales['order_date'],
                y=daily_sales['amount'],
                name='Ventes journalières',
                line=dict(color='#1f77b4')
            ))
            
            # Ajouter les périodes de campagnes
            for i, (start, end, name) in enumerate(campaign_periods):
                fig_sales.add_vrect(
                    x0=start,
                    x1=end,
                    fillcolor="lightgreen",
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                    annotation_text=name,
                    annotation_position="top left"
                )
            
            # Mise en forme du graphique
            fig_sales.update_layout(
                title="Impact des campagnes sur les ventes",
                xaxis_title="Date",
                yaxis_title="Montant des ventes (€)",
                showlegend=True,
                height=500
            )
            
            st.plotly_chart(fig_sales, use_container_width=True)
            
            # Calculer l'impact des campagnes
            st.subheader("Analyse d'impact des campagnes")
            
            impact_results = []
            
            for start, end, name in campaign_periods:
                # Période avant la campagne (même durée que la campagne)
                campaign_duration = (end - start).days + 1
                before_start = start - pd.Timedelta(days=campaign_duration)
                before_end = start - pd.Timedelta(days=1)
                
                # Ventes pendant la campagne
                campaign_sales = daily_sales[
                    (daily_sales['order_date'] >= start) & 
                    (daily_sales['order_date'] <= end)
                ]['amount'].sum()
                
                # Ventes avant la campagne (période de référence)
                before_sales = daily_sales[
                    (daily_sales['order_date'] >= before_start) & 
                    (daily_sales['order_date'] <= before_end)
                ]['amount'].sum()
                
                # Calculer l'impact
                if before_sales > 0:
                    impact = ((campaign_sales - before_sales) / before_sales) * 100
                else:
                    impact = 0
                
                impact_results.append({
                    'Campagne': name,
                    'Début': start.strftime('%d/%m/%Y'),
                    'Fin': end.strftime('%d/%m/%Y'),
                    'Durée (jours)': campaign_duration,
                    'Ventes pendant (€)': f"{campaign_sales:,.2f}",
                    'Ventes avant (€)': f"{before_sales:,.2f}",
                    'Impact (%)': f"{impact:.1f}%"
                })
            
            # Afficher les résultats dans un tableau
            if impact_results:
                impact_df = pd.DataFrame(impact_results)
                st.dataframe(
                    impact_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Télécharger les résultats
                csv = impact_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Télécharger l'analyse d'impact",
                    data=csv,
                    file_name="impact_campagnes.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Aucune campagne trouvée avec des données de vente correspondantes.")
        else:
            st.warning("Les dates de début et de fin des campagnes sont nécessaires pour cette analyse.")

# Navigation entre les pages
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("← Précédent", use_container_width=True):
        st.switch_page("pages/3_📊_Segmentation_client.py")

with col3:
    if st.button("Suivant →", type="primary", use_container_width=True):
        st.switch_page("pages/5_🔮_Analyse_prédictive.py")

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
    
    /* Style des tableaux */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Style des cartes d'information */
    .info-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #4e79a7;
    }
    
    .info-card h4 {
        margin-top: 0;
        color: #2c3e50;
    }
    
    /* Style des graphiques */
    .stPlotlyChart {
        border-radius: 10px;
        border: 1px solid #e1e5ed;
    }
    </style>
""", unsafe_allow_html=True)
