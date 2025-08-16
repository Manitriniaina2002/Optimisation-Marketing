"""
Module de visualisation pour l'application d'analyse marketing.

Ce module fournit des fonctions pour créer des visualisations interactives
à partir des données marketing et des résultats d'analyse.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Union, Tuple
import numpy as np

# Configuration des couleurs pour les graphiques
COLOR_PALETTE = [
    '#1f77b4',  # Bleu
    '#ff7f0e',  # Orange
    '#2ca02c',  # Vert
    '#d62728',  # Rouge
    '#9467bd',  # Violet
    '#8c564b',  # Marron
    '#e377c2',  # Rose
    '#7f7f7f',  # Gris
    '#bcbd22',  # Olive
    '#17becf'   # Cyan
]

def set_plotly_template():
    """Configure le thème par défaut pour les graphiques Plotly."""
    import plotly.io as pio
    
    # Configuration du template personnalisé
    pio.templates["custom"] = go.layout.Template(
        layout=go.Layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Arial, sans-serif", size=12, color="#2c3e50"),
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(showgrid=True, gridcolor="#e1e5ed", gridwidth=1, linecolor="#e1e5ed"),
            yaxis=dict(showgrid=True, gridcolor="#e1e5ed", gridwidth=1, linecolor="#e1e5ed"),
            colorway=COLOR_PALETTE,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.5)",
                bordercolor="#e1e5ed",
                borderwidth=1
            )
        )
    )
    pio.templates.default = "plotly_white+custom"

# Configurer le template au chargement du module
set_plotly_template()

def plot_distribution(
    data: pd.DataFrame,
    column: str,
    title: str = None,
    xaxis_title: str = None,
    yaxis_title: str = "Fréquence",
    color: str = None,
    nbins: int = None,
    height: int = 400
) -> None:
    """
    Affiche la distribution d'une variable numérique ou catégorielle.
    
    Args:
        data: DataFrame contenant les données
        column: Nom de la colonne à visualiser
        title: Titre du graphique
        xaxis_title: Titre de l'axe X
        yaxis_title: Titre de l'axe Y
        color: Colonne pour la couleur des barres
        nbins: Nombre de bacs pour les variables numériques
        height: Hauteur du graphique en pixels
    """
    if title is None:
        title = f"Distribution de {column}"
    if xaxis_title is None:
        xaxis_title = column
    
    # Vérifier si la colonne est numérique
    is_numeric = pd.api.types.is_numeric_dtype(data[column])
    
    fig = None
    
    try:
        if is_numeric:
            # Histogramme pour les variables numériques
            fig = px.histogram(
                data,
                x=column,
                color=color,
                nbins=nbins,
                title=title,
                marginal="box",
                opacity=0.7,
                height=height
            )
            
            # Ajouter des lignes verticales pour la moyenne et la médiane
            mean_val = data[column].mean()
            median_val = data[column].median()
            
            fig.add_vline(
                x=mean_val, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Moyenne: {mean_val:.2f}",
                annotation_position="top right"
            )
            
            fig.add_vline(
                x=median_val, 
                line_dash="dot", 
                line_color="green",
                annotation_text=f"Médiane: {median_val:.2f}",
                annotation_position="top right"
            )
            
        else:
            # Diagramme à barres pour les variables catégorielles
            value_counts = data[column].value_counts().reset_index()
            value_counts.columns = [column, 'count']
            
            fig = px.bar(
                value_counts,
                x=column,
                y='count',
                color=color,
                title=title,
                text='count',
                height=height
            )
            
            # Améliorer la lisibilité des étiquettes
            fig.update_traces(
                texttemplate='%{text:.2s}',
                textposition='outside'
            )
            
            # Rotation des étiquettes si nécessaire
            if len(value_counts) > 5:
                fig.update_xaxes(tickangle=45)
        
        # Mise en forme commune
        fig.update_layout(
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            showlegend=color is not None,
            hovermode='x unified',
            height=height
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erreur lors de la création du graphique : {str(e)}")

def plot_correlation_heatmap(
    data: pd.DataFrame,
    title: str = "Matrice de corrélation",
    height: int = 600,
    method: str = 'pearson'
) -> None:
    """
    Affiche une heatmap de corrélation pour les variables numériques.
    
    Args:
        data: DataFrame contenant les données
        title: Titre du graphique
        height: Hauteur du graphique en pixels
        method: Méthode de calcul de corrélation ('pearson', 'kendall', 'spearman')
    """
    # Sélectionner uniquement les colonnes numériques
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Pas assez de variables numériques pour la matrice de corrélation.")
        return
    
    try:
        # Calculer la matrice de corrélation
        corr_matrix = data[numeric_cols].corr(method=method)
        
        # Créer la heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            title=title,
            height=height
        )
        
        # Personnaliser la mise en page
        fig.update_layout(
            xaxis_title="Variables",
            yaxis_title="Variables",
            coloraxis_colorbar=dict(
                title="Corrélation",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"]
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erreur lors de la création de la matrice de corrélation : {str(e)}")

def plot_time_series(
    data: pd.DataFrame,
    date_column: str,
    value_column: str,
    group_column: str = None,
    title: str = "Série temporelle",
    xaxis_title: str = "Date",
    yaxis_title: str = "Valeur",
    height: int = 500
) -> None:
    """
    Affiche une série temporelle interactive.
    
    Args:
        data: DataFrame contenant les données
        date_column: Nom de la colonne de date
        value_column: Nom de la colonne de valeurs
        group_column: Colonne pour grouper les séries (optionnel)
        title: Titre du graphique
        xaxis_title: Titre de l'axe X
        yaxis_title: Titre de l'axe Y
        height: Hauteur du graphique en pixels
    """
    try:
        fig = px.line(
            data,
            x=date_column,
            y=value_column,
            color=group_column,
            title=title,
            height=height
        )
        
        # Améliorer la mise en page
        fig.update_layout(
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            hovermode='x unified',
            showlegend=group_column is not None,
            legend_title=group_column if group_column else None
        )
        
        # Ajouter des sélecteurs de plage de dates
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erreur lors de la création de la série temporelle : {str(e)}")

def plot_segments(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: str = None,
    size_column: str = None,
    hover_data: list = None,
    title: str = "Segmentation client",
    xaxis_title: str = None,
    yaxis_title: str = None,
    height: int = 600
) -> None:
    """
    Affiche une visualisation des segments clients.
    
    Args:
        data: DataFrame contenant les données de segmentation
        x_column: Nom de la colonne pour l'axe X
        y_column: Nom de la colonne pour l'axe Y
        color_column: Colonne pour la couleur des points
        size_column: Colonne pour la taille des points
        hover_data: Colonnes supplémentaires à afficher au survol
        title: Titre du graphique
        xaxis_title: Titre de l'axe X
        yaxis_title: Titre de l'axe Y
        height: Hauteur du graphique en pixels
    """
    if xaxis_title is None:
        xaxis_title = x_column
    if yaxis_title is None:
        yaxis_title = y_column
    
    try:
        fig = px.scatter(
            data,
            x=x_column,
            y=y_column,
            color=color_column,
            size=size_column,
            hover_data=hover_data,
            title=title,
            height=height,
            opacity=0.7
        )
        
        # Améliorer la mise en page
        fig.update_layout(
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            showlegend=color_column is not None,
            legend_title=color_column if color_column else None,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erreur lors de la création de la visualisation des segments : {str(e)}")

def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: List[str] = None,
    title: str = "Matrice de confusion",
    width: int = 700,
    height: int = 600
) -> go.Figure:
    """
    Affiche une matrice de confusion interactive.
    
    Args:
        conf_matrix: Matrice de confusion (2D array)
        class_names: Noms des classes
        title: Titre du graphique
        width: Largeur du graphique en pixels
        height: Hauteur du graphique en pixels
        
    Returns:
        Figure Plotly
    """
    if class_names is None:
        class_names = [f"Classe {i}" for i in range(conf_matrix.shape[0])]
    
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=class_names,
        y=class_names,
        text=[[str(y) for y in x] for x in conf_matrix],
        texttemplate="%{text}",
        textfont={"size": 12},
        colorscale='Blues',
        showscale=True,
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Prédit",
        yaxis_title="Réel",
        width=width,
        height=height,
        xaxis=dict(tickmode='array', tickvals=list(range(len(class_names))), ticktext=class_names),
        yaxis=dict(tickmode='array', tickvals=list(range(len(class_names))), ticktext=class_names)
    )
    
    return fig

def plot_feature_importance(
    feature_importance: Dict[str, float],
    title: str = "Importance des caractéristiques",
    max_features: int = 20,
    height: int = 500
) -> go.Figure:
    """
    Affiche un graphique d'importance des caractéristiques.
    
    Args:
        feature_importance: Dictionnaire des importances des caractéristiques
        title: Titre du graphique
        max_features: Nombre maximum de caractéristiques à afficher
        height: Hauteur du graphique en pixels
        
    Returns:
        Figure Plotly
    """
    # Trier les caractéristiques par importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Limiter le nombre de caractéristiques
    if len(sorted_features) > max_features:
        sorted_features = sorted_features[:max_features]
    
    features, importance = zip(*sorted_features)
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker_color=COLOR_PALETTE[0]
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Score d'importance",
        yaxis_title="Caractéristiques",
        height=height,
        margin=dict(l=150, r=50, t=50, b=50)
    )
    
    return fig

def plot_segment_comparison(
    data: pd.DataFrame,
    segment_column: str,
    metrics: List[str],
    title: str = "Comparaison des segments",
    height: int = 600
) -> go.Figure:
    """
    Affiche une comparaison des segments selon différentes métriques.
    
    Args:
        data: DataFrame contenant les données
        segment_column: Nom de la colonne des segments
        metrics: Liste des métriques à comparer
        title: Titre du graphique
        height: Hauteur du graphique en pixels
        
    Returns:
        Figure Plotly
    """
    fig = go.Figure()
    
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Box(
            y=data[metric],
            x=data[segment_column],
            name=metric,
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(size=4, opacity=0.6),
            line=dict(width=1),
            fillcolor=f'rgba(31, 119, 180, 0.1)'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Segments",
        yaxis_title="Valeur des métriques",
        boxmode='group',
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def display_metrics(
    metrics: Dict[str, Union[int, float, str]],
    columns: int = 4,
    title: str = "Métriques clés"
) -> None:
    """
    Affiche des métriques clés dans une grille.
    
    Args:
        metrics: Dictionnaire des métriques à afficher (clé: valeur)
        columns: Nombre de colonnes pour la grille
        title: Titre de la section
    """
    if title:
        st.subheader(title)
    
    # Créer une grille de colonnes
    cols = st.columns(columns)
    
    # Afficher chaque métrique dans une colonne
    for i, (name, value) in enumerate(metrics.items()):
        with cols[i % columns]:
            # Formater les valeurs numériques
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # Formater les grands nombres avec des séparateurs de milliers
                if abs(value) >= 1000:
                    formatted_value = f"{value:,.0f}".replace(",", " ")
                # Afficher 2 décimales pour les nombres décimaux
                else:
                    formatted_value = f"{value:,.2f}"
            else:
                formatted_value = str(value)
            
            # Afficher la métrique dans une carte
            st.metric(label=name, value=formatted_value)
