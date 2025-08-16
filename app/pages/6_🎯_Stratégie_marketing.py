"""
Page de stratégie marketing pour l'application d'analyse marketing.
Permet de définir et de suivre des stratégies marketing basées sur les segments clients.
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

from utils.visualization import display_metrics, plot_segment_comparison
from utils.analysis import MarketingAnalyzer
from config import APP_CONFIG

# Configuration de la page
st.set_page_config(
    page_title=f"{APP_CONFIG['app_name']} - Stratégie Marketing",
    page_icon="🎯",
    layout="wide"
)

# Titre de la page
st.title("🎯 Stratégie Marketing")
st.markdown("---")

# Vérifier que les données sont chargées
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Veuillez d'abord importer et valider les données dans l'onglet 'Importation des données'.")
    st.stop()

# Vérifier que la segmentation a été effectuée
if 'rfm_data' not in st.session_state:
    st.warning("Veuvez d'abord effectuer la segmentation des clients dans l'onglet 'Segmentation client'.")
    st.stop()

# Récupérer les données de la session
customers_df = st.session_state.get('customers_df')
orders_df = st.session_state.get('orders_df')
marketing_df = st.session_state.get('marketing_df')
rfm_data = st.session_state.get('rfm_data')

# Initialiser l'analyseur marketing
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = MarketingAnalyzer({
        'customers': customers_df,
        'orders': orders_df,
        'marketing': marketing_df if marketing_df is not None else pd.DataFrame(),
        'rfm': rfm_data
    })

# Section d'analyse des segments
with st.expander("📊 Analyse des segments", expanded=True):
    st.markdown("""
    Analysez les différents segments de clients pour définir des stratégies ciblées.
    """)
    
    # Vérifier que les segments sont disponibles
    if 'segment' not in rfm_data.columns:
        st.error("Aucun segment trouvé. Veuillez d'abord effectuer la segmentation des clients.")
        st.stop()
    
    # Afficher les statistiques des segments
    st.subheader("Vue d'ensemble des segments")
    
    # Calculer les métriques de base par segment
    segment_stats_df = rfm_data.groupby('segment').agg({
        'recency': ['mean', 'count'],
        'frequency': ['mean'],
        'monetary_value': ['mean', 'count']
    }).round(2)
    
    # Renommer les colonnes pour un affichage plus clair
    segment_stats_df.columns = [
        'récence_moyenne',
        'nb_clients',
        'fréquence_moyenne',
        'monetary_value_mean',
        'monetary_value_count'
    ]
    
    # Renommer les colonnes pour l'affichage
    display_names = {
        'récence_moyenne': 'Récence moyenne (jours)',
        'nb_clients': 'Nombre de clients',
        'fréquence_moyenne': 'Fréquence moyenne',
        'monetary_value_mean': 'Valeur moyenne (€)',
        'monetary_value_count': 'Nombre de clients'
    }
    # Inverser le mapping pour retrouver les colonnes internes depuis les libellés affichés
    display_to_internal = {v: k for k, v in display_names.items()}
    
    # Afficher les statistiques
    st.dataframe(
        segment_stats_df.sort_values('monetary_value_mean', ascending=False),
        use_container_width=True
    )
    
    # Graphique de comparaison des segments
    st.subheader("Comparaison des segments")
    
    # Sélection des métriques à comparer
    metrics = st.multiselect(
        "Métriques à comparer",
        options=list(display_to_internal.keys()),
        default=['Récence moyenne (jours)', 'Valeur moyenne (€)'],
        key="segment_metrics"
    )
    
    if metrics:
        fig = go.Figure()
        for metric in metrics:
            col_name = display_to_internal.get(metric)
            if col_name and col_name in segment_stats_df.columns:
                fig.add_trace(go.Bar(
                    x=segment_stats_df.index,
                    y=segment_stats_df[col_name],
                    name=metric,
                    text=segment_stats_df[col_name],
                    textposition='auto'
                ))
        
        fig.update_layout(
            title="Comparaison des segments",
            xaxis_title="Segment",
            yaxis_title="Valeur",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Analyse SWOT des segments
    st.subheader("Analyse SWOT par segment")
    
    selected_segment = st.selectbox(
        "Sélectionnez un segment",
        options=sorted(rfm_data['segment'].unique()),
        key="swot_segment"
    )
    
    # Calculer un profil moyen du segment sélectionné
    seg_df = rfm_data[rfm_data['segment'] == selected_segment]
    segment_profile = seg_df[['recency', 'frequency', 'monetary_value']].mean()
    
    # Indicateurs relatifs (comparaison aux médianes globales)
    is_high_value = segment_profile['monetary_value'] > rfm_data['monetary_value'].median()
    is_frequent = segment_profile['frequency'] > rfm_data['frequency'].median()
    is_recent = segment_profile['recency'] < rfm_data['recency'].median()
    
    swot_analysis = {
        'Forces': [
            {"text": "Fidélité élevée des clients", "priority": "Haute", "action": "Renforcer la fidélisation"} if is_frequent 
            else {"text": "Potentiel de fidélisation", "priority": "Moyenne", "action": "Développer la fidélité"},
            {"text": "Valeur par commande élevée", "priority": "Haute", "action": "Optimiser le panier moyen"} if is_high_value 
            else {"text": "Marge de progression sur la valeur", "priority": "Moyenne", "action": "Augmenter la valeur moyenne"},
            {"text": "Dernier achat récent", "priority": "Haute", "action": "Capitaliser sur l'engagement"} if is_recent 
            else {"text": "Potentiel de réactivation", "priority": "Haute", "action": "Relancer les clients"},
            {"text": "Potentiel de vente croisée", "priority": "Moyenne", "action": "Proposer des produits complémentaires"}
        ],
        'Faiblesses': [
            {"text": "Sensibilité aux prix", "priority": "Moyenne", "action": "Adapter la politique tarifaire"} if is_high_value 
            else {"text": "Valeur par client faible", "priority": "Haute", "action": "Augmenter la valeur client"},
            {"text": "Dernier achat ancien", "priority": "Haute", "action": "Relance ciblée"} if not is_recent 
            else {"text": "Potentiel de désengagement", "priority": "Moyenne", "action": "Renforcer l'engagement"},
            {"text": "Fréquence d'achat faible", "priority": "Haute", "action": "Stimuler la fréquence"} if not is_frequent 
            else {"text": "Risque de lassitude", "priority": "Moyenne", "action": "Varier l'offre"},
            {"text": "Coût d'acquisition élevé", "priority": "Haute", "action": "Optimiser les canaux d'acquisition"}
        ],
        'Opportunités': [
            {"text": "Développement de produits premium", "priority": "Haute", "action": "Étudier les besoins"} if is_high_value 
            else {"text": "Développement de produits d'entrée de gamme", "priority": "Moyenne", "action": "Étudier la demande"},
            {"text": "Programme de fidélisation premium", "priority": "Haute", "action": "Créer un programme VIP"} if is_frequent 
            else {"text": "Programme de fidélisation basique", "priority": "Moyenne", "action": "Mettre en place un programme"},
            {"text": "Personnalisation avancée", "priority": "Moyenne", "action": "Développer l'offre personnalisée"} if is_high_value 
            else {"text": "Standardisation des offres", "priority": "Basse", "action": "Optimiser les coûts"},
            {"text": "Expansion sur de nouveaux marchés", "priority": "Moyenne", "action": "Étude de marché"}
        ],
        'Menaces': [
            {"text": "Concurrence sur les prix", "priority": "Haute", "action": "Renforcer la différenciation"} if is_high_value 
            else {"text": "Concurrence sur la valeur", "priority": "Moyenne", "action": "Améliorer la proposition de valeur"},
            {"text": "Changement des préférences", "priority": "Moyenne", "action": "Surveiller les tendances"},
            {"text": "Économie en récession", "priority": "Haute", "action": "Adapter l'offre"} if is_high_value 
            else {"text": "Pression sur les marges", "priority": "Moyenne", "action": "Optimiser les coûts"},
            {"text": "Nouvelles réglementations", "priority": "Basse", "action": "Veille réglementaire"}
        ]
    }
    
    # Fonction pour afficher une carte SWOT avec style conditionnel
    def display_swot_card(category, items, icon):
        colors = {
            'Forces': '#2ecc71',
            'Faiblesses': '#e74c3c',
            'Opportunités': '#3498db',
            'Menaces': '#f39c12'
        }
        color_val = colors[category]
        
        with st.container():
            st.markdown(f"""
            <div style="
                border-left: 5px solid {color_val};
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 15px;
                margin: 10px 0;
            ">
                <h4 style="margin-top: 0; color: {color_val};">{icon} {category}</h4>
            """, unsafe_allow_html=True)
            
            for item in items:
                priority_icons = {
                    'Haute': '🔥',
                    'Moyenne': '⚠️',
                    'Basse': 'ℹ️'
                }
                
                st.markdown(f"""
                <div style="
                    background-color: white;
                    border-radius: 5px;
                    padding: 10px;
                    margin: 5px 0;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span>{item['text']}</span>
                        <span style="color: #7f8c8d; font-size: 0.9em;">{priority_icons[item['priority']]} {item['priority']}</span>
                    </div>
                    <div style="margin-top: 5px;">
                        <button 
                                data-action="{item['action']}"
                                style="
                                    background-color: {color_val};
                                    color: white;
                                    border: none;
                                    border-radius: 3px;
                                    padding: 3px 8px;
                                    font-size: 0.8em;
                                    cursor: pointer;
                                ">
                            {item['action']}
                        </button>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Afficher l'analyse SWOT dans des colonnes
    col1, col2 = st.columns(2)
    
    with col1:
        display_swot_card('Forces', swot_analysis['Forces'], '✅')
        display_swot_card('Opportunités', swot_analysis['Opportunités'], '🔍')
    
    with col2:
        display_swot_card('Faiblesses', swot_analysis['Faiblesses'], '⚠️')
        display_swot_card('Menaces', swot_analysis['Menaces'], '⚠️')
    
    # Ajouter un graphique radar pour visualiser les caractéristiques du segment
    st.subheader("Profil du segment")
    
    categories = ['Récence', 'Fréquence', 'Valeur']
    # Sécuriser contre division par zéro
    rec_range = rfm_data['recency'].max() - rfm_data['recency'].min()
    freq_range = rfm_data['frequency'].max() - rfm_data['frequency'].min()
    val_range = rfm_data['monetary_value'].max() - rfm_data['monetary_value'].min()
    
    values = [
        ((rfm_data['recency'].max() - segment_profile['recency']) / rec_range * 100) if rec_range > 0 else 50,
        ((segment_profile['frequency'] - rfm_data['frequency'].min()) / freq_range * 100) if freq_range > 0 else 50,
        ((segment_profile['monetary_value'] - rfm_data['monetary_value'].min()) / val_range * 100) if val_range > 0 else 50
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + values[:1],
        theta=categories + categories[:1],
        fill='toself',
        name=selected_segment,
        line=dict(color='#3498db')
    ))
    
    avg_values = [50, 50, 50, 50]
    fig.add_trace(go.Scatterpolar(
        r=avg_values,
        theta=categories + [categories[0]],
        name='Moyenne',
        line=dict(color='#95a5a6', dash='dash')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                ticks='',
                showline=False,
                showgrid=True
            )
        ),
        showlegend=True,
        title=f"Profil RFM du segment {selected_segment}",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Ajouter des indicateurs de performance clés (KPIs)
    st.subheader("Indicateurs clés")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Score de récence",
            value=f"{values[0]:.0f}/100",
            delta=f"{values[0] - 50:+.0f} vs moyenne"
        )
    
    with col2:
        st.metric(
            label="Score de fréquence",
            value=f"{values[1]:.0f}/100",
            delta=f"{values[1] - 50:+.0f} vs moyenne"
        )
    
    with col3:
        st.metric(
            label="Score de valeur",
            value=f"{values[2]:.0f}/100",
            delta=f"{values[2] - 50:+.0f} vs moyenne"
        )

# Section de définition des objectifs
with st.expander("🎯 Définition des objectifs", expanded=True):
    st.markdown("""
    Définissez des objectifs SMART (Spécifiques, Mesurables, Atteignables, Réalistes, Temporels) 
    pour chaque segment de clients.
    """)
    
    # Sélection du segment pour définir les objectifs
    segment_for_goals = st.selectbox(
        "Sélectionnez un segment pour définir les objectifs",
        options=sorted(rfm_data['segment'].unique()),
        key="goals_segment"
    )
    
    # Formulaire pour définir les objectifs SMART
    with st.form(f"objectifs_{segment_for_goals}"):
        st.subheader(f"Objectifs pour le segment {segment_for_goals}")
        
        # Objectif spécifique
        specific_goal = st.text_input(
            "Objectif spécifique",
            value=f"Augmenter les ventes du segment {segment_for_goals}",
            key=f"specific_{segment_for_goals}"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Objectif mesurable
            metric = st.selectbox(
                "Métrique à mesurer",
                options=["Chiffre d'affaires", "Nombre de clients", "Panier moyen", "Taux de rétention"],
                key=f"metric_{segment_for_goals}"
            )
            
            # Valeur cible
            target_value = st.number_input(
                "Valeur cible",
                min_value=0.0,
                value=10000.0,
                step=100.0,
                key=f"target_{segment_for_goals}"
            )
        
        with col2:
            # Unité de mesure
            unit = st.selectbox(
                "Unité de mesure",
                options=["€", "%", "nombre", "clients"],
                key=f"unit_{segment_for_goals}"
            )
            
            # Échéance
            deadline = st.date_input(
                "Date limite",
                value=datetime.now() + timedelta(days=90),
                key=f"deadline_{segment_for_goals}"
            )
        
        # Description détaillée
        description = st.text_area(
            "Description détaillée de l'objectif",
            value=f"Définir une stratégie pour atteindre {target_value} {unit} de {metric.lower()} pour le segment {segment_for_goals} d'ici le {deadline.strftime('%d/%m/%Y')}.",
            key=f"desc_{segment_for_goals}"
        )
        
        # Bouton pour enregistrer l'objectif
        if st.form_submit_button("Enregistrer l'objectif", type="primary"):
            # Initialiser le stockage des objectifs si nécessaire
            if 'marketing_goals' not in st.session_state:
                st.session_state.marketing_goals = {}
            
            # Enregistrer l'objectif
            goal_id = f"{segment_for_goals}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            st.session_state.marketing_goals[goal_id] = {
                'segment': segment_for_goals,
                'specific': specific_goal,
                'metric': metric,
                'target_value': target_value,
                'unit': unit,
                'deadline': deadline,
                'description': description,
                'created_at': datetime.now(),
                'status': 'En cours'
            }
            
            st.success(f"Objectif pour le segment {segment_for_goals} enregistré avec succès !")
    
    # Afficher les objectifs existants
    if 'marketing_goals' in st.session_state and st.session_state.marketing_goals:
        st.subheader("Objectifs enregistrés")
        
        # Filtrer les objectifs par segment
        segment_goals = {
            k: v for k, v in st.session_state.marketing_goals.items() 
            if v['segment'] == segment_for_goals
        }
        
        if segment_goals:
            for goal_id, goal in segment_goals.items():
                with st.expander(f"{goal['specific']} - {goal['status']}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Description**  \n{goal['description']}")
                        
                        st.markdown(
                            f"**Progression**  \n"
                            f"Objectif : {goal['target_value']} {goal['unit']} "
                            f"d'ici le {goal['deadline'].strftime('%d/%m/%Y')}"
                        )
                        
                        # Afficher une barre de progression (exemple statique)
                        progress = st.slider(
                            "Progression actuelle",
                            min_value=0,
                            max_value=100,
                            value=30,
                            key=f"progress_{goal_id}",
                            label_visibility="collapsed"
                        )
                        
                        # Calculer les jours restants
                        days_left = (goal['deadline'] - datetime.now().date()).days
                        st.caption(f"{days_left} jours restants")
                    
                    with col2:
                        # Statut de l'objectif
                        new_status = st.selectbox(
                            "Statut",
                            options=["En cours", "Atteint", "En retard", "Abandonné"],
                            index=["En cours", "Atteint", "En retard", "Abandonné"].index(goal['status']),
                            key=f"status_{goal_id}",
                            label_visibility="collapsed"
                        )
                        
                        # Mettre à jour le statut si nécessaire
                        if new_status != goal['status']:
                            st.session_state.marketing_goals[goal_id]['status'] = new_status
                            st.rerun()
                        
                        # Bouton pour supprimer l'objectif
                        if st.button("Supprimer", key=f"delete_{goal_id}"):
                            del st.session_state.marketing_goals[goal_id]
                            st.rerun()
        else:
            st.info(f"Aucun objectif défini pour le segment {segment_for_goals}.")
    else:
        st.info("Aucun objectif défini pour le moment.")

# Section de recommandations stratégiques
with st.expander("💡 Recommandations stratégiques", expanded=True):
    st.markdown("""
    Découvrez des recommandations stratégiques personnalisées pour chaque segment de clients.
    """)
    
    # Sélection du segment pour les recommandations
    segment_for_recommendations = st.selectbox(
        "Sélectionnez un segment pour les recommandations",
        options=sorted(rfm_data['segment'].unique()),
        key="recommendations_segment"
    )
    
    # Générer des recommandations basées sur les caractéristiques du segment
    segment_data = rfm_data[rfm_data['segment'] == segment_for_recommendations].iloc[0]
    
    # Déterminer le type de segment (basé sur RFM)
    segment_type = ""
    if segment_data['recency'] < rfm_data['recency'].median() and \
       segment_data['frequency'] > rfm_data['frequency'].median() and \
       segment_data['monetary_value'] > rfm_data['monetary_value'].median():
        segment_type = "Clients fidèles à forte valeur"
    elif segment_data['recency'] < rfm_data['recency'].median() and \
         segment_data['frequency'] > rfm_data['frequency'].median():
        segment_type = "Clients fidèles"
    elif segment_data['recency'] > rfm_data['recency'].median() and \
         segment_data['monetary_value'] > rfm_data['monetary_value'].median():
        segment_type = "Clients dormants à forte valeur"
    elif segment_data['frequency'] < rfm_data['frequency'].median() and \
         segment_data['monetary_value'] < rfm_data['monetary_value'].median():
        segment_type = "Nouveaux clients ou clients occasionnels"
    else:
        segment_type = "Segment moyen"
    
    # Afficher les caractéristiques du segment
    st.subheader(f"Caractéristiques du segment {segment_for_recommendations}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Type de segment", segment_type)
    
    with col2:
        st.metric("Récence moyenne", f"{segment_data['recency']:.1f} jours")
    
    with col3:
        st.metric("Fréquence moyenne", f"{segment_data['frequency']:.1f} achats")
    
    # Générer des recommandations basées sur le type de segment
    st.subheader("Recommandations stratégiques")
    
    if segment_type == "Clients fidèles à forte valeur":
        st.markdown("""
        ### Stratégie recommandée : Fidélisation et développement
        
        Ces clients sont vos meilleurs clients. L'objectif est de maintenir leur fidélité et d'augmenter leur valeur à vie.
        
        **Actions recommandées :**
        - 💎 **Programme VIP** : Créez un programme exclusif avec des avantages spéciaux
        - 🎁 **Cadeaux personnalisés** : Offrez des cadeaux d'anniversaire ou de fidélité
        - 📞 **Service client premium** : Offrez un accès prioritaire au support
        - 🔄 **Vente croisée** : Proposez des produits complémentaires de gamme supérieure
        - 📊 **Écoute client** : Mettez en place des entretiens clients pour comprendre leurs besoins
        """)
    
    elif segment_type == "Clients fidèles":
        st.markdown("""
        ### Stratégie recommandée : Augmentation de la valeur client
        
        Ces clients sont fidèles mais dépensent moins que vos meilleurs clients. L'objectif est d'augmenter leur panier moyen.
        
        **Actions recommandées :**
        - 🛍️ **Packs et offres groupées** : Proposez des offres groupées pour augmenter le panier moyen
        - ⬆️ **Vente incitative** : Formez votre équipe à la vente incitative
        - 🎯 **Recommandations personnalisées** : Utilisez l'IA pour des recommandations pertinentes
        - 💳 **Programme de fidélité** : Incitez à des achats plus importants avec des récompenses
        - 📧 **Email marketing ciblé** : Envoyez des offres personnalisées basées sur l'historique d'achat
        """)
    
    elif segment_type == "Clients dormants à forte valeur":
        st.markdown("""
        ### Stratégie recommandée : Reconquête
        
        Ces clients ont une forte valeur mais n'ont pas acheté récemment. L'objectif est de les faire revenir.
        
        **Actions recommandées :**
        - ✉️ **Campagnes de réactivation** : Envoyez des emails personnalisés avec des offres spéciales
        - 📞 **Appels de reconquête** : Contactez-les personnellement pour comprendre leur absence
        - 🔄 **Offres exclusives** : Proposez des remises ou des avantages pour leur retour
        - 📱 **Retargeting publicitaire** : Ciblez-les avec des publicités sur les réseaux sociaux
        - ❓ **Enquête de satisfaction** : Demandez-leur pourquoi ils ne sont pas revenus
        """)
    
    elif segment_type == "Nouveaux clients ou clients occasionnels":
        st.markdown("""
        ### Stratégie recommandée : Fidélisation et éducation
        
        Ces clients sont nouveaux ou n'achètent qu'occasionnellement. L'objectif est de les convertir en clients réguliers.
        
        **Actions recommandées :**
        - 👋 **Email de bienvenue** : Envoyez une série d'emails pour les guider
        - 🎁 **Offre de bienvenue** : Proposez une réduction sur leur prochain achat
        - 📚 **Contenu éducatif** : Partagez des conseils et tutoriels sur l'utilisation de vos produits
        - 🤝 **Programme de parrainage** : Incitez-les à parrainer des amis contre récompense
        - ⏰ **Rappels intelligents** : Envoyez des rappels basés sur leur cycle d'achat
        """)
    
    else:  # Segment moyen
        st.markdown("""
        ### Stratégie recommandée : Amélioration continue
        
        Ces clients ont un potentiel inexploité. L'objectif est de les faire progresser vers des segments plus rentables.
        
        **Actions recommandées :**
        - 📊 **Analyse comportementale** : Identifiez les modèles d'achat et les opportunités
        - 🎯 **Segmentation avancée** : Affinez vos segments pour un ciblage plus précis
        - 🔄 **Tests A/B** : Testez différentes approches pour voir ce qui fonctionne le mieux
        - 📈 **Optimisation du parcours client** : Simplifiez le processus d'achat
        - 💬 **Feedback client** : Recueillez des avis pour améliorer l'expérience client
        """)
    
    # Recommandations génériques pour tous les segments
    st.markdown("""
    ### Recommandations générales
    
    - **Personnalisation** : Utilisez le prénom du client dans les communications
    - **Automatisation** : Mettez en place des parcours automatisés basés sur le comportement
    - **Omnicanal** : Assurez une expérience cohérente sur tous les canaux
    - **Analyse concurrentielle** : Surveillez les offres de vos concurrents
    - **Amélioration continue** : Mesurez les résultats et ajustez votre stratégie en conséquence
    """)

# Section de suivi des performances
with st.expander("📈 Suivi des performances", expanded=True):
    st.markdown("""
    Suivez les performances de vos stratégies marketing par segment.
    """)
    
    # Vérifier s'il y a des objectifs définis
    if 'marketing_goals' not in st.session_state or not st.session_state.marketing_goals:
        st.info("Aucun objectif défini pour le moment. Définissez des objectifs dans la section 'Définition des objectifs'.")
    else:
        # Afficher un tableau de bord des objectifs
        st.subheader("Tableau de bord des objectifs")
        
        # Convertir les objectifs en DataFrame pour l'affichage
        goals_list = []
        for goal_id, goal in st.session_state.marketing_goals.items():
            goals_list.append({
                'Segment': goal['segment'],
                'Objectif': goal['specific'],
                'Métrique': f"{goal['target_value']} {goal['unit']}",
                'Échéance': goal['deadline'].strftime('%d/%m/%Y'),
                'Statut': goal['status'],
                'Jours restants': (goal['deadline'] - datetime.now().date()).days
            })
        
        goals_df = pd.DataFrame(goals_list)
        
        # Afficher le tableau des objectifs
        st.dataframe(
            goals_df,
            column_config={
                'Jours restants': st.column_config.ProgressColumn(
                    "Jours restants",
                    help="Jours restants avant l'échéance",
                    format="%d",
                    min_value=0,
                    max_value=365,
                )
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Graphique des objectifs par statut
        st.subheader("Répartition des objectifs par statut")
        
        if not goals_df.empty:
            status_counts = goals_df['Statut'].value_counts().reset_index()
            status_counts.columns = ['Statut', 'Nombre d\'objectifs']
            
            fig_status = px.pie(
                status_counts,
                values='Nombre d\'objectifs',
                names='Statut',
                title="Répartition des objectifs par statut",
                hole=0.4,
                color='Statut',
                color_discrete_map={
                    'Atteint': '#2ecc71',
                    'En cours': '#3498db',
                    'En retard': '#e74c3c',
                    'Abandonné': '#95a5a6'
                }
            )
            
            st.plotly_chart(fig_status, use_container_width=True)
        
        # Graphique des objectifs par segment
        st.subheader("Objectifs par segment")
        
        if not goals_df.empty:
            segment_goals = goals_df.groupby('Segment')['Objectif'].count().reset_index()
            segment_goals.columns = ['Segment', 'Nombre d\'objectifs']
            
            fig_segment = px.bar(
                segment_goals,
                x='Segment',
                y='Nombre d\'objectifs',
                title="Nombre d'objectifs par segment",
                color='Segment',
                text='Nombre d\'objectifs'
            )
            
            fig_segment.update_traces(
                textposition='outside',
                marker_line_color='rgb(8,48,107)',
                marker_line_width=1.5
            )
            
            st.plotly_chart(fig_segment, use_container_width=True)

# Navigation entre les pages
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("← Précédent", use_container_width=True):
        st.session_state.current_step = 5
        st.rerun()

with col3:
    if st.button("Suivant →", type="primary", use_container_width=True):
        st.session_state.current_step = 7
        st.rerun()

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
    
    /* Style des listes à puces */
    ul {
        padding-left: 20px;
    }
    
    li {
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)
