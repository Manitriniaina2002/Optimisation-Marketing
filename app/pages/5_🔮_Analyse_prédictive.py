"""
Page d'analyse prédictive pour l'application d'analyse marketing.
Permet de réaliser des prédictions sur le comportement des clients et la valeur à vie.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import traceback
from datetime import datetime, timedelta
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
# from xgboost import XGBClassifier  # lazy-imported when selected
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import label_binarize

# Ajouter le répertoire parent au chemin pour les imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.visualization import plot_confusion_matrix, plot_feature_importance
from utils.analysis import MarketingAnalyzer
from config import APP_CONFIG

# Configuration de la page
st.set_page_config(
    page_title=f"{APP_CONFIG['app_name']} - Analyse prédictive",
    page_icon="🔮",
    layout="wide"
)

# Titre de la page
st.title("🔮 Analyse prédictive")
st.markdown("---")

# Vérifier que les données sont chargées
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Veuillez d'abord importer et valider les données dans l'onglet 'Importation des données'.")
    st.stop()

# Récupérer les données de la session
customers_df = st.session_state.get('customers_df')
orders_df = st.session_state.get('orders_df')
marketing_df = st.session_state.get('marketing_df')

# Travailler sur une copie pour éviter les SettingWithCopyWarning
if customers_df is not None:
    customers_df = customers_df.copy()

# Vérifier que les données nécessaires sont disponibles
if customers_df is None or orders_df is None:
    st.error("Les données clients et/ou commandes sont manquantes. Veuillez importer les données nécessaires.")
    st.stop()

# Initialiser l'analyseur marketing
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = MarketingAnalyzer({
        'customers': customers_df,
        'orders': orders_df,
        'marketing': marketing_df if marketing_df is not None else pd.DataFrame()
    })

# Section de préparation des données
with st.expander("📊 Préparation des données", expanded=True):
    st.markdown("""
    ### 🛠️ Configuration de l'analyse prédictive
    
    Cette section vous permet de configurer votre analyse prédictive en sélectionnant les variables pertinentes
    et en définissant les paramètres de préparation des données.
    """)
    
    # Indicateur visuel d'étape
    st.progress(0.2, text="Étape 1/4 - Configuration de l'analyse")
    
    # Vérifier si les segments sont disponibles
    if 'rfm_data' not in st.session_state or 'segment' not in st.session_state.rfm_data.columns:
        st.warning("Veuvez d'abord effectuer la segmentation des clients dans l'onglet précédent.")
        st.stop()
    
    # Préparer les données pour la modélisation
    st.subheader("Sélection des variables")
    
    # Détection robuste des variables numériques
    numeric_cols = []
    for col in customers_df.columns:
        if col == 'customer_id':
            continue
            
        # Vérifier si la colonne est déjà numérique
        if pd.api.types.is_numeric_dtype(customers_df[col]):
            numeric_cols.append(col)
        # Sinon, essayer de convertir en numérique
        else:
            try:
                # Essayer de convertir en numérique
                pd.to_numeric(customers_df[col], errors='raise')
                numeric_cols.append(col)
                # Convertir la colonne en numérique
                customers_df[col] = pd.to_numeric(customers_df[col], errors='coerce')
            except (ValueError, TypeError):
                pass  # La colonne n'est pas numérique
    
    # Détection des variables catégorielles
    categorical_cols = []
    for col in customers_df.columns:
        if col in numeric_cols or col == 'customer_id':
            continue
            
        # Si la colonne a un nombre limité de valeurs uniques, la considérer comme catégorielle
        unique_ratio = customers_df[col].nunique() / len(customers_df)
        if unique_ratio < 0.5:  # Moins de 50% de valeurs uniques
            categorical_cols.append(col)
    
    # Afficher un avertissement si aucune variable numérique n'est détectée
    if not numeric_cols:
        st.warning("⚠️ Aucune variable numérique détectée. Vérifiez le format de vos données.")
        
        # Afficher les types de données détectés pour le débogage (version compatible Arrow)
        st.write("Types de données détectés dans le DataFrame :")
        try:
            dtypes_str = customers_df.dtypes.astype(str)
            dtypes_df = pd.DataFrame({
                'Colonne': dtypes_str.index,
                'Type': dtypes_str.values
            })
            st.dataframe(dtypes_df, use_container_width=True)
        except Exception:
            st.write({col: str(dt) for col, dt in customers_df.dtypes.items()})
        
        # Afficher un échantillon des données
        st.write("Aperçu des données :")
        st.dataframe(customers_df.head())
    
    # Afficher les colonnes détectées pour le débogage
    st.session_state.debug_numeric_cols = numeric_cols
    st.session_state.debug_categorical_cols = categorical_cols
    
    # Afficher les colonnes détectées (en mode débogage)
    with st.expander("🔍 Détection des colonnes (débogage)", expanded=False):
        st.write("Colonnes numériques détectées :", numeric_cols)
        st.write("Colonnes catégorielles détectées :", categorical_cols)
        st.write("Types de données :")
        try:
            dtypes_str = customers_df.dtypes.astype(str)
            dtypes_df = pd.DataFrame({
                'Colonne': dtypes_str.index,
                'Type': dtypes_str.values
            })
            st.dataframe(dtypes_df, use_container_width=True)
        except Exception:
            st.write({col: str(dt) for col, dt in customers_df.dtypes.items()})
    
    # Sélection des variables pour la modélisation
    selected_numeric = st.multiselect(
        "Variables numériques",
        options=numeric_cols,
        default=numeric_cols[:3] if len(numeric_cols) > 0 else [],
        help="Sélectionnez les variables numériques à inclure dans le modèle"
    )
    
    selected_categorical = st.multiselect(
        "Variables catégorielles",
        options=categorical_cols,
        default=categorical_cols[:2] if len(categorical_cols) > 0 else [],
        help="Sélectionnez les variables catégorielles à inclure dans le modèle"
    )
    
    # Sélection de la variable cible
    target_options = ['segment']  # Par défaut, on prédit le segment
    
    # Ajouter d'autres options de cible si disponibles
    if 'churn' in customers_df.columns:
        target_options.append('churn')
    if 'lifetime_value' in customers_df.columns:
        target_options.append('lifetime_value')
    
    target_variable = st.selectbox(
        "Variable cible",
        options=target_options,
        index=0,
        help="Sélectionnez la variable à prédire"
    )
    
    # Options de prétraitement
    st.subheader("Options de prétraitement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        handle_missing = st.selectbox(
            "Gestion des valeurs manquantes",
            options=["Supprimer les lignes", "Remplacer par la médiane/mode", "Imputer avec KNN"],
            index=1
        )
        
        scale_features = st.checkbox(
            "Mettre à l'échelle les variables numériques",
            value=True,
            help="Standardiser les variables numériques (moyenne=0, écart-type=1)"
        )
    
    with col2:
        test_size = st.slider(
            "Taille de l'ensemble de test",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Proportion des données à utiliser pour le test"
        )
        
        random_state = st.number_input(
            "Graine aléatoire",
            min_value=0,
            max_value=100,
            value=42,
            help="Pour la reproductibilité des résultats"
        )
    
    # Bouton pour préparer les données
    if st.button("Préparer les données", type="primary"):
        with st.spinner("Préparation des données en cours..."):
            try:
                # Fusionner les données RFM avec les données clients
                rfm_data = st.session_state.rfm_data
                
                # Vérifier que les colonnes nécessaires sont présentes dans rfm_data
                required_rfm_columns = ['customer_id', 'recency', 'frequency', 'monetary_value', 'segment']
                missing_rfm_cols = [col for col in required_rfm_columns if col not in rfm_data.columns]
                if missing_rfm_cols:
                    st.error(f"Colonnes manquantes dans les données RFM : {', '.join(missing_rfm_cols)}")
                    st.stop()
                
                # Sélectionner les colonnes nécessaires
                features = ['customer_id'] + selected_numeric + selected_categorical
                
                # Vérifier que les colonnes sélectionnées existent dans customers_df
                missing_cols = [col for col in features if col not in customers_df.columns]
                if missing_cols:
                    st.error(f"Colonnes manquantes dans les données clients : {', '.join(missing_cols)}")
                    st.stop()
                
                # Vérifier le type de la colonne customer_id dans les deux DataFrames
                if rfm_data['customer_id'].dtype != customers_df['customer_id'].dtype:
                    st.warning("Les types de la colonne 'customer_id' diffèrent entre les données RFM et clients. Conversion en type commun...")
                    rfm_data['customer_id'] = rfm_data['customer_id'].astype(str)
                    customers_df['customer_id'] = customers_df['customer_id'].astype(str)
                
                # Créer le jeu de données complet
                try:
                    model_data = pd.merge(
                        rfm_data[required_rfm_columns],
                        customers_df[features],
                        on='customer_id',
                        how='left'
                    )
                except Exception as e:
                    st.error(f"Erreur lors de la fusion des données RFM et clients : {e}")
                    st.error(f"Types de colonnes dans rfm_data: {rfm_data[required_rfm_columns].dtypes}")
                    st.error(f"Types de colonnes dans customers_df: {customers_df[features].dtypes}")
                    st.stop()
                
                # Gérer les valeurs manquantes
                if handle_missing == "Supprimer les lignes":
                    model_data = model_data.dropna()
                elif handle_missing == "Remplacer par la médiane/mode":
                    for col in selected_numeric:
                        if col in model_data.columns and model_data[col].isnull().any():
                            model_data[col] = model_data[col].fillna(model_data[col].median())
                    
                    for col in selected_categorical:
                        if col in model_data.columns and model_data[col].isnull().any():
                            model_data[col] = model_data[col].fillna(model_data[col].mode()[0])
                
                # Encoder les variables catégorielles
                label_encoders = {}
                for col in selected_categorical:
                    if col in model_data.columns:
                        le = LabelEncoder()
                        model_data[col] = le.fit_transform(model_data[col].astype(str))
                        label_encoders[col] = le
                
                # Préparer les caractéristiques et la cible
                X = model_data[['recency', 'frequency', 'monetary_value'] + selected_numeric + selected_categorical]
                y = model_data[target_variable]
                
                # Mettre à l'échelle les variables numériques si demandé
                scaler = None
                if scale_features and len(selected_numeric) > 0:
                    scaler = StandardScaler()
                    X[selected_numeric] = scaler.fit_transform(X[selected_numeric])
                
                # Diviser les données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) > 1 else None
                )
                
                # Enregistrer les données préparées dans la session
                st.session_state.model_data = {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'feature_names': X.columns.tolist(),
                    'target_name': target_variable,
                    'label_encoders': label_encoders,
                    'scaler': scaler,
                    'preprocessing': {
                        'handle_missing': handle_missing,
                        'scale_features': scale_features,
                        'test_size': test_size,
                        'random_state': random_state
                    }
                }
                
                st.success(f"Données préparées avec succès pour la prédiction de {target_variable}!")
                st.write(f"- Nombre d'échantillons d'entraînement : {len(X_train)}")
                st.write(f"- Nombre d'échantillons de test : {len(X_test)}")
                st.write(f"- Nombre de caractéristiques : {len(X_train.columns)}")
                
                # Afficher un aperçu des données préparées
                st.subheader("Aperçu des données préparées")
                st.dataframe(pd.concat([X_train.head(), y_train.head()], axis=1))
                
            except Exception as e:
                st.error(f"Erreur lors de la préparation des données : {e}")

# Vérifier si les données sont préparées
if 'model_data' not in st.session_state:
    st.warning("Veuvez d'abord préparer les données dans la section ci-dessus.")
    st.stop()

# Section de modélisation
with st.expander("🤖 Entraînement du modèle", expanded=False):
    st.markdown("""
    ### 🎯 Entraînement du modèle
    
    Configurez et entraînez un modèle de machine learning pour effectuer des prédictions sur vos données.
    Comparez les performances de différents algorithmes et paramètres.
    """)
    
    # Indicateur visuel d'étape
    st.progress(0.5, text="Étape 2/4 - Entraînement du modèle")
    
    # Sélection du modèle
    model_type = st.selectbox(
        "Type de modèle",
        options=["Forêt aléatoire", "Gradient Boosting", "XGBoost"],
        index=0,
        help="Sélectionnez l'algorithme de machine learning à utiliser"
    )
    
    # Paramètres du modèle
    st.subheader("Paramètres du modèle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider(
            "Nombre d'arbres",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Nombre d'arbres dans la forêt ou d'estimateurs"
        )
        
        max_depth = st.slider(
            "Profondeur maximale",
            min_value=1,
            max_value=20,
            value=5,
            help="Profondeur maximale de chaque arbre"
        )
    
    with col2:
        min_samples_split = st.slider(
            "Nombre minimum d'échantillons pour diviser un nœud",
            min_value=2,
            max_value=20,
            value=2,
            help="Nombre minimum d'échantillons requis pour diviser un nœud interne"
        )
        
        min_samples_leaf = st.slider(
            "Nombre minimum d'échantillons par feuille",
            min_value=1,
            max_value=10,
            value=1,
            help="Nombre minimum d'échantillons requis pour être dans un nœud feuille"
        )
    
    # Options d'entraînement
    st.subheader("Options d'entraînement")
    
    use_cross_validation = st.checkbox(
        "Utiliser la validation croisée",
        value=True,
        help="Évalue le modèle avec une validation croisée sur l'ensemble d'entraînement"
    )
    
    cv_folds = st.slider(
        "Nombre de plis pour la validation croisée",
        min_value=3,
        max_value=10,
        value=5,
        disabled=not use_cross_validation
    )
    
    # Bouton pour lancer l'entraînement
    if st.button("Entraîner le modèle", type="primary"):
        with st.spinner("Entraînement du modèle en cours..."):
            try:
                # Récupérer les données préparées
                model_data = st.session_state.model_data
                X_train = model_data['X_train']
                X_test = model_data['X_test']
                y_train = model_data['y_train']
                y_test = model_data['y_test']
                
                # Initialiser le modèle sélectionné
                if model_type == "Forêt aléatoire":
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=model_data['preprocessing']['random_state'],
                        n_jobs=-1,
                        verbose=0
                    )
                elif model_type == "Gradient Boosting":
                    model = GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=model_data['preprocessing']['random_state'],
                        verbose=0
                    )
                else:  # XGBoost
                    try:
                        from xgboost import XGBClassifier  # lazy import
                        model = XGBClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_child_weight=min_samples_leaf,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=model_data['preprocessing']['random_state'],
                            n_jobs=-1,
                            verbosity=0
                        )
                    except Exception as e:
                        st.error("XGBoost n'est pas disponible dans cet environnement.")
                        st.info("Sélectionnez 'Forêt aléatoire' ou 'Gradient Boosting', ou ajoutez 'xgboost' aux requirements si nécessaire.")
                        st.stop()
                
                # Entraîner le modèle
                model.fit(X_train, y_train)
                
                # Faire des prédictions sur l'ensemble de test
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
                
                # Calculer les métriques d'évaluation
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Calculer l'AUC-ROC si possible (classification binaire ou multiclasse avec probabilités)
                if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                    auc_roc = roc_auc_score(y_test, y_pred_proba[:, 1])
                elif y_pred_proba is not None and len(np.unique(y_test)) > 2:
                    # Pour la classification multiclasse, on peut calculer l'AUC-ROC moyenné
                    try:
                        auc_roc = roc_auc_score(
                            pd.get_dummies(y_test), 
                            y_pred_proba,
                            multi_class='ovr',
                            average='weighted'
                        )
                    except:
                        auc_roc = None
                else:
                    auc_roc = None
                
                # Enregistrer le modèle et les résultats dans la session
                st.session_state.trained_model = {
                    'model': model,
                    'model_type': model_type,
                    'metrics': {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'auc_roc': auc_roc
                    },
                    'predictions': {
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba
                    },
                    'feature_importances': dict(zip(
                        model_data['feature_names'],
                        model.feature_importances_
                    )) if hasattr(model, 'feature_importances_') else None
                }
                
                st.success("Modèle entraîné avec succès !")
                
                # Afficher les métriques d'évaluation
                st.subheader("Performances du modèle")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Exactitude (Accuracy)", f"{accuracy:.3f}")
                
                with col2:
                    st.metric("Précision", f"{precision:.3f}")
                
                with col3:
                    st.metric("Rappel", f"{recall:.3f}")
                
                with col4:
                    st.metric("Score F1", f"{f1:.3f}")
                
                # Affichage amélioré des métriques
                st.markdown("### Détail des performances")
                
                # Création d'un tableau de métriques
                metrics_df = pd.DataFrame({
                    'Métrique': ['Précision', 'Rappel', 'F1-Score', 'Exactitude'],
                    'Valeur': [precision, recall, f1, accuracy],
                    'Description': [
                        'Capacité à ne pas classer à tort un client comme positif',
                        'Capacité à trouver tous les clients positifs',
                        'Moyenne harmonique entre précision et rappel',
                        'Pourcentage de prédictions correctes'
                    ]
                })
                
                # Afficher le tableau des métriques
                st.dataframe(
                    metrics_df,
                    column_config={
                        'Métrique': st.column_config.TextColumn('Métrique'),
                        'Valeur': st.column_config.ProgressColumn(
                            'Valeur',
                            format='%.3f',
                            min_value=0,
                            max_value=1
                        ),
                        'Description': st.column_config.TextColumn('Description')
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Affichage de la courbe ROC si les probabilités sont disponibles
                if hasattr(model, 'predict_proba'):
                    try:
                        # Obtenir les probabilités de prédiction
                        y_pred_proba = model.predict_proba(X_test)
                        classes_ = getattr(model, 'classes_', np.unique(y_test))
                        n_classes = len(classes_)
                        
                        st.markdown("### Courbe ROC")
                        
                        if n_classes == 2:
                            # Cas binaire
                            # Identifier l'index de la classe positive (on prend la 2e classe par convention)
                            pos_index = 1 if y_pred_proba.shape[1] > 1 else 0
                            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, pos_index], pos_label=classes_[pos_index])
                            roc_auc = auc(fpr, tpr)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.3f})', line=dict(color='#3498db', width=2)))
                            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aléatoire', line=dict(color='#95a5a6', width=1, dash='dash')))
                            fig.update_layout(title='Courbe ROC', xaxis_title='Taux de faux positifs', yaxis_title='Taux de vrais positifs', showlegend=True, height=500)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Cas multi-classes: One-vs-Rest
                            y_test_bin = label_binarize(y_test, classes=classes_)
                            fig = go.Figure()
                            for i, cls in enumerate(classes_):
                                try:
                                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                                    roc_auc = auc(fpr, tpr)
                                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Classe {cls} (AUC = {roc_auc:.3f})'))
                                except Exception:
                                    continue
                            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aléatoire', line=dict(color='#95a5a6', width=1, dash='dash')))
                            fig.update_layout(title='Courbes ROC (One-vs-Rest)', xaxis_title='Taux de faux positifs', yaxis_title='Taux de vrais positifs', showlegend=True, height=500)
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Impossible d'afficher la courbe ROC : {str(e)}")
                
                # Affichage de l'importance des caractéristiques
                st.markdown("### Importance des caractéristiques")
                
                try:
                    # Récupérer l'importance des caractéristiques
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        feature_importance = pd.DataFrame({
                            'Caractéristique': X_train.columns,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False)
                        
                        # Afficher le graphique d'importance
                        fig = px.bar(
                            feature_importance.head(10),
                            x='Importance',
                            y='Caractéristique',
                            orientation='h',
                            title='Top 10 des caractéristiques les plus importantes',
                            labels={'Importance': 'Importance', 'Caractéristique': 'Caractéristique'},
                            color='Importance',
                            color_continuous_scale='Blues'
                        )
                        
                        fig.update_layout(
                            height=400,
                            showlegend=False,
                            xaxis_title='Importance',
                            yaxis_title='Caractéristique',
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Afficher la matrice de confusion
                    st.markdown("### Matrice de confusion")
                    
                    classes_ = np.unique(y_test)
                    cm = confusion_matrix(y_test, y_pred, labels=classes_)
                    fig = px.imshow(
                        cm,
                        labels=dict(x="Prédit", y="Réel", color="Nombre"),
                        x=[str(c) for c in classes_],
                        y=[str(c) for c in classes_],
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='Blues'
                    )
                    
                    fig.update_layout(
                        title='Matrice de confusion',
                        xaxis_title='Prédiction',
                        yaxis_title='Vérité terrain',
                        coloraxis_showscale=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interprétation des résultats
                    st.markdown("### Interprétation des résultats")
                    
                    st.markdown("""
                    **Comment interpréter ces résultats ?**
                    
                    - **Précision élevée** : Lorsque le modèle prédit une classe, il a une forte probabilité d'avoir raison.
                    - **Rappel élevé** : Le modèle identifie bien la plupart des cas positifs.
                    - **F1-Score** : Bon équilibre entre précision et rappel (1 = parfait).
                    - **AUC-ROC** : Capacité du modèle à distinguer les classes (1 = parfait).
                    
                    **Recommandations :**
                    - Si la précision est faible, le modèle fait trop de faux positifs.
                    - Si le rappel est faible, le modèle manque trop de vrais positifs.
                    - Si les deux sont bas, le modèle doit être amélioré ou les données mieux préparées.
                    """)
                    
                except Exception as e:
                    st.warning(f"Impossible d'afficher certaines visualisations : {str(e)}")
                
                # Afficher la matrice de confusion
                st.subheader("Matrice de confusion")
                
                # Créer la figure de la matrice de confusion avec l'utilitaire (corrige l'appel)
                classes_ = np.unique(y_test)
                cm = confusion_matrix(y_test, y_pred, labels=classes_)
                fig_cm = plot_confusion_matrix(
                    cm,
                    class_names=[str(c) for c in classes_],
                    title=f"Matrice de confusion - {model_type}"
                )
                
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Afficher le rapport de classification
                st.subheader("Rapport de classification")
                
                # Générer le rapport de classification
                report = classification_report(
                    y_test, 
                    y_pred, 
                    target_names=[f"Classe {label}" for label in np.unique(y_test)],
                    output_dict=True
                )
                
                # Convertir en DataFrame pour un affichage plus propre
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
                
                # Afficher l'importance des caractéristiques si disponible
                if st.session_state.trained_model['feature_importances'] is not None:
                    st.subheader("Importance des caractéristiques")
                    
                    # Créer un graphique d'importance des caractéristiques
                    fig_importance = plot_feature_importance(
                        st.session_state.trained_model['feature_importances'],
                        title=f"Importance des caractéristiques - {model_type}"
                    )
                    
                    st.plotly_chart(fig_importance, use_container_width=True)
                
            except Exception as e:
                st.error(f"Erreur lors de l'entraînement du modèle : {e}")

# Vérifier si un modèle est entraîné
if 'trained_model' not in st.session_state:
    st.warning("Veuvez d'abord entraîner un modèle dans la section ci-dessus.")
    st.stop()

# Section de prédiction
with st.expander(" Faire des prédictions", expanded=False):
    st.markdown("""
    ### Effectuer des prédictions
    
    Utilisez le modèle entraîné pour effectuer des prédictions sur de nouvelles données
    ou sur un échantillon de test pour évaluer ses performances.
    """)
    
    # Indicateur visuel d'étape
    st.progress(0.8, text="Étape 3/4 - Prédictions")
    
    # Récupérer le modèle et les données
    model_data = st.session_state.model_data
    trained_model = st.session_state.trained_model
    model = trained_model['model']
    
    # Options de prédiction
    prediction_type = st.radio(
        "Type de prédiction",
        ["Sur un échantillon de test", "Sur de nouvelles données"],
        horizontal=True,
        help="Choisissez de tester le modèle sur des données de test ou de faire des prédictions sur de nouvelles entrées"
    )
    
    # Ajout d'un séparateur visuel
    st.markdown("---")
    
    # Aide contextuelle
    with st.expander(" Comment utiliser cette section", expanded=False):
        st.markdown("""
        **Prédictions sur échantillon de test**
        - Le modèle est évalué sur des données qu'il n'a jamais vues pendant l'entraînement
        - Permet d'estimer les performances réelles du modèle
        
        **Prédictions sur nouvelles données**
        - Saisissez manuellement les valeurs pour chaque caractéristique
        - Obtenez une prédiction instantanée
        - Parfait pour tester des scénarios spécifiques
        """)
    
    if prediction_type == "Sur un échantillon de test":
        # Récupérer les données de test
        X_test = model_data['X_test']
        y_test = model_data['y_test']
        
        # Nombre d'échantillons à afficher
        max_samples = min(50, len(X_test))
        min_samples = min(5, max_samples)  # S'assurer que min_samples <= max_samples
        
        if max_samples > 0:
            sample_size = st.slider(
                "Nombre d'échantillons à afficher",
                min_value=1,
                max_value=max(1, max_samples),
                value=min(10, max_samples),
                step=1
            )
        else:
            st.warning("Aucun échantillon disponible pour l'affichage.")
            st.stop()
        
        # Faire des prédictions
        with st.spinner("Prédiction en cours..."):
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculer les métriques
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Afficher les résultats
        st.subheader(" Résultats sur l'échantillon de test")
        
        # Afficher les métriques dans des colonnes
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Exactitude", f"{accuracy:.2%}")
        with col2:
            st.metric("Précision", f"{precision:.2%}")
        with col3:
            st.metric("Rappel", f"{recall:.2%}")
        with col4:
            st.metric("Score F1", f"{f1:.2%}")
        
        # Afficher un échantillon des prédictions
        st.markdown(f"### Aperçu des prédictions ({sample_size} échantillons)")
        
        # Sélection aléatoire d'échantillons
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        
        # Créer un DataFrame avec les résultats
        results = []
        for i in sample_indices:
            true_label = y_test.iloc[i]
            pred_label = y_pred[i]
            result = {
                'ID': i,
                'Vérité terrain': str(true_label),
                'Prédiction': str(pred_label),
                'Statut': ' Correct' if true_label == pred_label else ' Erreur'
            }
            if y_pred_proba is not None:
                # Probabilité de la classe prédite (ou max proba en multiclasses)
                proba = float(np.max(y_pred_proba[i]))
                result['Confiance'] = f"{proba:.1%}"
                result['_Jauge'] = proba
            results.append(result)
        
        # Créer un DataFrame avec les résultats
        results_df = pd.DataFrame(results)
        
        # Afficher le tableau des résultats avec mise en forme conditionnelle
        st.dataframe(
            results_df,
            column_config={
                'ID': 'ID',
                'Vérité terrain': st.column_config.TextColumn('Vérité terrain'),
                'Prédiction': st.column_config.TextColumn('Prédiction'),
                'Statut': st.column_config.TextColumn('Statut'),
                'Confiance': st.column_config.ProgressColumn(
                    'Confiance',
                    format='%.1f%%',
                    min_value=0,
                    max_value=1,
                    help="Niveau de confiance de la prédiction"
                ) if 'Confiance' in results_df.columns else None,
                '_Jauge': None  # Colonne masquée pour la jauge
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Afficher la distribution des prédictions
        st.markdown("### Distribution des prédictions")
        
        # Calculer les comptes de prédictions avec étiquettes dynamiques
        classes_pred = np.unique(y_pred)
        pred_counts = pd.Series(y_pred).value_counts().reindex(classes_pred, fill_value=0)
        labels = [str(c) for c in classes_pred]
        
        # Créer un graphique à barres
        fig = px.bar(
            x=labels,
            y=pred_counts.values,
            labels={'x': 'Classe prédite', 'y': "Nombre d'échantillons"},
            title="Distribution des prédictions sur l'ensemble de test",
            color=labels,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig.update_layout(
            showlegend=False,
            xaxis_title='Classe prédite',
            yaxis_title="Nombre d'échantillons",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Formulaire pour la saisie manuelle des caractéristiques
        st.subheader("Saisie des caractéristiques")
        
        # Créer un formulaire pour chaque caractéristique
        input_data = {}
        
        # Ajouter les champs pour les caractéristiques RFM
        input_data['recency'] = st.number_input(
            "Récence (jours)", 
            min_value=0, 
            max_value=365*5, 
            value=30,
            help="Nombre de jours depuis le dernier achat"
        )
        
        input_data['frequency'] = st.number_input(
            "Fréquence", 
            min_value=1, 
            value=5,
            help="Nombre total d'achats"
        )

        # Définir une valeur par défaut pour monetary_value
        default_monetary = 100.0  # Valeur par défaut si rfm_data n'est pas disponible
        if 'rfm_data' in st.session_state and 'monetary_value' in st.session_state.rfm_data:
            default_monetary = float(st.session_state.rfm_data['monetary_value'].median())
        
        input_data['monetary_value'] = st.number_input(
            "Valeur monétaire (€)",
            min_value=0.0,
            value=default_monetary,
            help="Valeur monétaire moyenne des achats"
        )

        # Ajouter les champs pour les autres caractéristiques numériques
        for col in model_data['X_train'].columns:
            if col not in ['recency', 'frequency', 'monetary_value'] and model_data['X_train'][col].dtype in ['int64', 'float64']:
                min_val = float(model_data['X_train'][col].min())
                max_val = float(model_data['X_train'][col].max())
                mean_val = float(model_data['X_train'][col].mean())
                
                input_data[col] = st.number_input(
                    f"{col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100 if (max_val - min_val) > 0 else 0.1
                )
        
        # Bouton pour effectuer la prédiction
        if st.button("Prédire", type="primary"):
            try:
                # Récupérer les noms des caractéristiques dans le même ordre que lors de l'entraînement
                feature_names = model_data.get('feature_names', model_data['X_train'].columns.tolist())
                
                # Créer un DataFrame vide avec les colonnes dans le bon ordre
                input_df = pd.DataFrame(columns=feature_names)
                
                # Remplir avec les valeurs du formulaire
                for col in feature_names:
                    if col in input_data:
                        # Convertir explicitement le type de données pour correspondre à l'entraînement
                        if col in model_data['X_train'].columns:
                            dtype = model_data['X_train'][col].dtype
                            if dtype == 'int64':
                                input_df[col] = [int(input_data[col])]
                            elif dtype == 'float64':
                                input_df[col] = [float(input_data[col])]
                            else:
                                input_df[col] = [input_data[col]]
                    else:
                        # Si une caractéristique est manquante, utiliser la médiane (pour les numériques) ou le mode (pour les catégorielles)
                        if col in model_data['X_train'].columns:
                            if model_data['X_train'][col].dtype in ['int64', 'float64']:
                                input_df[col] = [float(model_data['X_train'][col].median())]
                            else:
                                input_df[col] = [model_data['X_train'][col].mode()[0]]
                
                # Vérifier que toutes les colonnes nécessaires sont présentes
                missing_cols = set(feature_names) - set(input_df.columns)
                if missing_cols:
                    st.error(f"Colonnes manquantes dans les données d'entrée : {', '.join(missing_cols)}")
                    st.error("Veuvez réinitialiser l'application et vérifier les données d'entraînement.")
                    st.stop()
                
                # Vérifier que les types de données correspondent
                for col in input_df.columns:
                    expected_dtype = model_data['X_train'][col].dtype
                    if input_df[col].dtype != expected_dtype:
                        try:
                            if expected_dtype == 'int64':
                                input_df[col] = input_df[col].astype('int64')
                            elif expected_dtype == 'float64':
                                input_df[col] = input_df[col].astype('float64')
                        except Exception as e:
                            st.error(f"Impossible de convertir la colonne {col} en {expected_dtype}: {e}")
                            st.stop()
                
                # Afficher un aperçu des données avant prédiction (pour le débogage)
                st.write("### Données d'entrée pour la prédiction")
                st.dataframe(input_df[feature_names].head())
                
                # Mettre à l'échelle les caractéristiques si nécessaire
                if model_data['scaler'] is not None:
                    numeric_cols = [col for col in feature_names 
                                  if model_data['X_train'][col].dtype in ['int64', 'float64']]
                    
                    # Vérifier que toutes les colonnes numériques sont présentes
                    missing_numeric = [col for col in numeric_cols if col not in input_df.columns]
                    if missing_numeric:
                        st.error(f"Colonnes numériques manquantes pour la mise à l'échelle : {', '.join(missing_numeric)}")
                        st.stop()
                    
                    # Appliquer la mise à l'échelle
                    input_df[numeric_cols] = model_data['scaler'].transform(input_df[numeric_cols])
                
                # Vérifier une dernière fois que toutes les colonnes attendues sont présentes
                missing_final = set(feature_names) - set(input_df.columns)
                if missing_final:
                    st.error(f"Colonnes manquantes après préparation : {', '.join(missing_final)}")
                    st.stop()
                
                # Réorganiser les colonnes pour correspondre exactement à l'ordre d'entraînement
                input_df = input_df[feature_names]
                
                # Faire la prédiction
                prediction = model.predict(input_df)
                prediction_proba = model.predict_proba(input_df) if hasattr(model, "predict_proba") else None
                
            except Exception as e:
                st.error(f"Erreur lors de la préparation des données pour la prédiction : {e}")
                st.error("Détails de l'erreur :")
                st.error(traceback.format_exc())
                st.error("\nVeuvez vérifier que les données d'entrée correspondent au format attendu par le modèle.")
                st.stop()
            
            # Enregistrer l'exemple de prédiction
            example = input_df.copy()
            example['target_predicted'] = prediction[0]
            if prediction_proba is not None:
                for i, proba in enumerate(prediction_proba[0]):
                    example[f'prob_class_{i}'] = proba
            
            st.session_state.prediction_example = example
    
    # Afficher les résultats de la prédiction si disponible
    if 'prediction_example' in st.session_state:
        example = st.session_state.prediction_example
        
        st.subheader("Résultat de la prédiction")
        
        # Afficher la prédiction
        if 'target_actual' in example.columns:
            st.write(f"**Valeur réelle** : {example['target_actual'].iloc[0]}")
        
        st.write(f"**Prédiction** : {example['target_predicted'].iloc[0]}")
        
        # Afficher les probabilités si disponibles
        prob_cols = [col for col in example.columns if col.startswith('prob_class_')]
        if prob_cols:
            st.subheader("Probabilités par classe")
            
            # Créer un graphique à barres des probabilités
            prob_values = [example[col].iloc[0] for col in prob_cols]
            prob_labels = [f"Classe {i}" for i in range(len(prob_cols))]
            
            fig_proba = px.bar(
                x=prob_labels,
                y=prob_values,
                labels={'x': 'Classe', 'y': 'Probabilité'},
                title="Distribution des probabilités de prédiction",
                text=[f"{p*100:.1f}%" for p in prob_values]
            )
            
            fig_proba.update_traces(
                marker_color='#636EFA',
                textposition='outside',
                textfont_size=12
            )
            
            fig_proba.update_layout(
                yaxis=dict(range=[0, 1.1]),
                height=400
            )
            
            st.plotly_chart(fig_proba, use_container_width=True)
        
        # Afficher les caractéristiques d'entrée
        st.subheader("Caractéristiques d'entrée")
        
        # Filtrer les colonnes à afficher
        display_cols = [col for col in example.columns if not col.startswith(('prob_class_', 'target_'))]
        
        # Afficher les caractéristiques dans un tableau
        st.dataframe(
            example[display_cols].T.rename(columns={0: 'Valeur'}),
            use_container_width=True
        )

# Section d'exportation du modèle
with st.expander("💾 Exporter le modèle", expanded=False):
    st.markdown("""
    ### 📤 Exportation du modèle
    
    Exportez votre modèle entraîné pour une utilisation ultérieure ou pour le déploiement en production.
    """)
    
    # Indicateur visuel d'étape
    st.progress(1.0, text="Étape 4/4 - Exportation")
    
    # Options d'exportation
    export_format = st.selectbox(
        "Format d'exportation",
        ["Joblib (.pkl)", "ONNX", "PMML"],
        index=0
    )
    
    # Bouton pour exporter le modèle
    if st.button("Exporter le modèle", type="primary"):
        with st.spinner("Exportation en cours..."):
            try:
                # Créer un fichier temporaire
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                    # Sauvegarder le modèle et les métadonnées
                    model_data = {
                        'model': st.session_state.trained_model['model'],
                        'model_type': st.session_state.trained_model['model_type'],
                        'metrics': st.session_state.trained_model['metrics'],
                        'feature_names': st.session_state.model_data['feature_names'],
                        'target_name': st.session_state.model_data['target_name'],
                        'preprocessing': st.session_state.model_data['preprocessing']
                    }
                    
                    joblib.dump(model_data, tmp_file.name)
                    
                    # Lire le fichier pour le téléchargement
                    with open(tmp_file.name, 'rb') as f:
                        bytes_data = f.read()
                    
                    # Proposer le téléchargement
                    st.download_button(
                        label="Télécharger le modèle",
                        data=bytes_data,
                        file_name=f"modele_{st.session_state.model_data['target_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                        mime="application/octet-stream"
                    )
                
                # Supprimer le fichier temporaire
                os.unlink(tmp_file.name)
                
                st.success("Modèle exporté avec succès !")
                
            except Exception as e:
                st.error(f"Erreur lors de l'exportation du modèle : {e}")

# Navigation entre les pages
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("← Précédent", use_container_width=True):
        st.switch_page("pages/4_📈_Analyse_des_performances.py")

with col3:
    if st.button("Suivant →", type="primary", use_container_width=True):
        st.switch_page("pages/6_🎯_Stratégie_marketing.py")

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
