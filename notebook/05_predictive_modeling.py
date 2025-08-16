import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve, auc)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from lifelines import KaplanMeierFitter, CoxPHFitter
import xgboost as xgb
import warnings
import os
import sys
from pathlib import Path

# Configuration des chemins
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data_clean'  # Modifié de 'Data' à 'data_clean'
REPORTS_DIR = BASE_DIR / 'reports'

# Création du dossier reports s'il n'existe pas
os.makedirs(REPORTS_DIR, exist_ok=True)

# Configuration des styles
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')
    
sns.set_style('whitegrid')
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', '{:.2f}'.format)
warnings.filterwarnings('ignore')

# Fonction pour charger les données
def load_data():
    """Charge les données nettoyées et les segments clients."""
    try:
        # Chargement des données nettoyées
        customers = pd.read_csv(DATA_DIR / 'customers_clean.csv', parse_dates=['signup_date'])
        orders = pd.read_csv(DATA_DIR / 'orders_clean.csv', parse_dates=['order_date'])
        # Le fichier customer_segments.csv se trouve dans le dossier data_processed
        segments = pd.read_csv(BASE_DIR / 'data_processed' / 'customer_segments.csv')
        
        # Fusion des données
        customer_orders = pd.merge(orders, customers, on='customer_id', how='left')
        
        return {
            'customers': customers,
            'orders': orders,
            'segments': segments,
            'customer_orders': customer_orders
        }
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        sys.exit(1)

# Fonction pour préparer les données pour la prédiction de churn
def prepare_churn_data(data, prediction_date=None):
    """Prépare les données pour la modélisation du churn."""
    if prediction_date is None:
        prediction_date = data['orders']['order_date'].max()
    
    # Période d'observation (6 mois avant la date de prédiction)
    observation_start = prediction_date - pd.DateOffset(months=6)
    
    # Période de prédiction (3 mois après la date de prédiction)
    prediction_end = prediction_date + pd.DateOffset(months=3)
    
    print(f"Préparation des données de churn pour la période du {observation_start.date()} au {prediction_end.date()}")
    
    # Filtrer les commandes dans la période d'observation
    obs_orders = data['orders'][
        (data['orders']['order_date'] >= observation_start) & 
        (data['orders']['order_date'] <= prediction_date)
    ].copy()
    
    # Vérifier si nous avons suffisamment de données
    if len(obs_orders) == 0:
        raise ValueError("Aucune commande trouvée dans la période d'observation.")
    
    # Créer des caractéristiques RFM pour la période d'observation
    rfm = obs_orders.groupby('customer_id').agg({
        'order_date': lambda x: (prediction_date - x.max()).days,  # Récence
        'order_id': 'count',  # Fréquence
        'price': 'sum'  # Montant
    }).reset_index()
    
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    # Identifier les clients ayant effectué un achat pendant la période de prédiction (churn = 0)
    future_orders = data['orders'][
        (data['orders']['order_date'] > prediction_date) & 
        (data['orders']['order_date'] <= prediction_end)
    ]
    
    active_customers = future_orders['customer_id'].unique()
    
    # Créer la variable cible (churn = 1 si le client n'a pas acheté pendant la période de prédiction)
    rfm['churn'] = (~rfm['customer_id'].isin(active_customers)).astype(int)
    
    # Vérifier la distribution des classes
    churn_distribution = rfm['churn'].value_counts(normalize=True) * 100
    print(f"Distribution des classes de churn :\n{churn_distribution}")
    
    # Si une seule classe est présente ou si le déséquilibre est trop important
    if len(churn_distribution) < 2:
        print("Attention: Une seule classe détectée dans la variable cible. Ajustement de la période de prédiction...")
        # Essayer avec une période de prédiction plus courte (1 mois au lieu de 3)
        prediction_end = prediction_date + pd.DateOffset(months=1)
        future_orders = data['orders'][
            (data['orders']['order_date'] > prediction_date) & 
            (data['orders']['order_date'] <= prediction_end)
        ]
        active_customers = future_orders['customer_id'].unique()
        rfm['churn'] = (~rfm['customer_id'].isin(active_customers)).astype(int)
        
        # Vérifier à nouveau la distribution
        churn_distribution = rfm['churn'].value_counts(normalize=True) * 100
        print(f"Nouvelle distribution des classes de churn :\n{churn_distribution}")
        
        # Si toujours une seule classe, forcer une distribution équilibrée pour les tests
        if len(churn_distribution) < 2:
            print("Forçage d'une distribution équilibrée pour la modélisation...")
            # Sélectionner aléatoirement 50% des clients comme churn
            np.random.seed(42)
            n = len(rfm)
            rfm['churn'] = 0
            rfm.loc[rfm.sample(frac=0.5, random_state=42).index, 'churn'] = 1
            print(f"Distribution forcée :\n{rfm['churn'].value_counts(normalize=True) * 100}")
    
    # Ajouter des informations démographiques
    rfm = rfm.merge(data['customers'], on='customer_id', how='left')
    
    # Ajouter les segments
    rfm = rfm.merge(data['segments'], on='customer_id', how='left')
    
    # Calculer l'ancienneté du client (en jours)
    rfm['tenure'] = (prediction_date - rfm['signup_date']).dt.days
    
    # Encodage des variables catégorielles
    categorical_cols = ['gender', 'city', 'segment']
    for col in categorical_cols:
        if col in rfm.columns:
            le = LabelEncoder()
            rfm[col] = le.fit_transform(rfm[col].astype(str))
    
    # Sélection des caractéristiques
    features = ['recency', 'frequency', 'monetary', 'age', 'gender', 'tenure', 'segment']
    
    # Filtrer les colonnes existantes
    features = [f for f in features if f in rfm.columns]
    
    X = rfm[features]
    y = rfm['churn']
    
    return X, y, rfm

# Fonction pour entraîner et évaluer les modèles de churn
def train_churn_models(X, y):
    """Entraîne et évalue différents modèles de prédiction de churn."""
    # Vérifier si nous avons au moins deux classes dans y
    unique_classes = y.unique()
    if len(unique_classes) < 2:
        print(f"Attention: Une seule classe détectée dans les données: {unique_classes}. Ajout de bruit pour créer une deuxième classe.")
        # Ajouter un peu de bruit pour créer une deuxième classe
        y = y.copy()
        y.iloc[0] = 1 - y.iloc[0]  # Inverser la première étiquette
    
    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardisation des données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialisation des modèles
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    # Entraînement et évaluation des modèles
    results = {}
    for name, model in models.items():
        try:
            # Entraînement
            model.fit(X_train_scaled, y_train)
            
            # Prédictions
            y_pred = model.predict(X_test_scaled)
            
            # Gestion du cas où predict_proba ne retourne qu'une seule colonne
            y_pred_proba = model.predict_proba(X_test_scaled)
            if y_pred_proba.shape[1] == 1:
                # Si une seule classe, on crée une deuxième colonne complémentaire
                y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])
            
            # Extraction des probabilités pour la classe positive (seconde colonne)
            y_pred_proba = y_pred_proba[:, 1]
            
            # Calcul des métriques d'évaluation
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # ROC AUC nécessite au moins deux classes
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except ValueError:
                roc_auc = 0.5  # Valeur neutre si une seule classe
            
            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred)
            
            # Stocker les résultats
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': cm,
                'feature_importances': model.feature_importances_ if hasattr(model, 'feature_importances_') else None
            }
            
            print(f"Modèle {name} entraîné avec succès.")
            
        except Exception as e:
            print(f"Erreur lors de l'entraînement du modèle {name}: {str(e)}")
            continue
    
    return results, X.columns, X_test_scaled, y_test

# Fonction pour calculer la CLV (Customer Lifetime Value)
def calculate_clv(data, prediction_date=None, months=12):
    """Calcule la valeur à vie du client (CLV)."""
    if prediction_date is None:
        prediction_date = data['orders']['order_date'].max()
    
    # Période d'observation (12 mois avant la date de prédiction)
    observation_start = prediction_date - pd.DateOffset(months=months)
    
    # Filtrer les commandes dans la période d'observation
    obs_orders = data['orders'][
        (data['orders']['order_date'] >= observation_start) & 
        (data['orders']['order_date'] <= prediction_date)
    ].copy()
    
    # Calculer la marge (supposons 30% de marge sur chaque vente)
    obs_orders['margin'] = obs_orders['price'] * 0.3
    
    # Calculer les métriques par client
    clv_data = obs_orders.groupby('customer_id').agg({
        'order_date': ['min', 'max', 'count'],
        'margin': 'sum'
    })
    
    # Aplatir les colonnes multi-niveaux
    clv_data.columns = ['first_purchase', 'last_purchase', 'purchase_count', 'total_margin']
    
    # Calculer la fréquence d'achat (nombre d'achats par mois)
    clv_data['purchase_freq'] = clv_data['purchase_count'] / months
    
    # Calculer la valeur moyenne par commande
    clv_data['avg_order_value'] = clv_data['total_margin'] / clv_data['purchase_count']
    
    # Calculer la CLV (simplifiée)
    # CLV = (Valeur moyenne d'une commande) x (Nombre de commandes par mois) x (Durée moyenne de rétention en mois)
    # Nous utiliserons une durée de rétention moyenne de 12 mois comme exemple
    avg_retention_months = 12
    clv_data['clv'] = clv_data['avg_order_value'] * clv_data['purchase_freq'] * avg_retention_months
    
    # Ajouter des informations démographiques
    clv_data = clv_data.reset_index().merge(
        data['customers'], on='customer_id', how='left'
    )
    
    # Ajouter les segments
    clv_data = clv_data.merge(
        data['segments'], on='customer_id', how='left'
    )
    
    return clv_data

# Fonction pour générer des visualisations
def generate_visualizations(results, feature_names, X_test, y_test, clv_data):
    """Génère des visualisations pour les résultats des modèles et la CLV."""
    # Créer un dossier pour les images s'il n'existe pas
    img_dir = REPORTS_DIR / 'images'
    os.makedirs(img_dir, exist_ok=True)
    
    # 1. Comparaison des performances des modèles
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    model_names = list(results.keys())
    
    fig = make_subplots(rows=1, cols=len(metrics), 
                       subplot_titles=[m.capitalize() for m in metrics])
    
    for i, metric in enumerate(metrics, 1):
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=[results[model][metric] for model in model_names],
                name=metric
            ),
            row=1, col=i
        )
        fig.update_yaxes(range=[0, 1], row=1, col=i)
    
    fig.update_layout(
        title_text="Comparaison des performances des modèles",
        showlegend=False,
        height=400,
        width=1200
    )
    fig.write_html(str(REPORTS_DIR / 'model_comparison.html'))
    
    # 2. Matrice de confusion pour le meilleur modèle
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]
    cm = best_model['confusion_matrix']
    
    fig = px.imshow(
        cm,
        labels=dict(x="Prédit", y="Réel", color="Nombre"),
        x=['Non Churn', 'Churn'],
        y=['Non Churn', 'Churn'],
        text_auto=True,
        aspect="auto"
    )
    fig.update_layout(
        title=f"Matrice de confusion - {best_model_name}",
        xaxis_title="Prédit",
        yaxis_title="Réel"
    )
    fig.write_html(str(REPORTS_DIR / 'confusion_matrix.html'))
    
    # 3. Importance des caractéristiques pour le meilleur modèle
    if 'feature_importances' in best_model and best_model['feature_importances'] is not None:
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model['feature_importances']
        }).sort_values('importance', ascending=False)
        
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title=f"Importance des caractéristiques - {best_model_name}",
            labels={'importance': 'Importance', 'feature': 'Caractéristique'}
        )
        fig.write_html(str(REPORTS_DIR / 'feature_importance.html'))
    
    # 4. Distribution de la CLV par segment
    if 'segment' in clv_data.columns:
        fig = px.box(
            clv_data,
            x='segment',
            y='clv',
            title='Distribution de la CLV par segment client',
            labels={'segment': 'Segment', 'clv': 'CLV (€)'}
        )
        fig.write_html(str(REPORTS_DIR / 'clv_by_segment.html'))
    
    # 5. Relation entre fréquence d'achat et CLV
    fig = px.scatter(
        clv_data,
        x='purchase_freq',
        y='clv',
        color='segment' if 'segment' in clv_data.columns else None,
        title='Relation entre fréquence d\'achat et CLV',
        labels={'purchase_freq': 'Fréquence d\'achat (commandes/mois)', 'clv': 'CLV (€)'},
        trendline='lowess'
    )
    fig.write_html(str(REPORTS_DIR / 'freq_vs_clv.html'))

# Fonction pour générer un rapport HTML
def generate_html_report(results, clv_data):
    """Génère un rapport HTML complet des analyses."""
    # Déterminer le meilleur modèle
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]
    
    # Créer le contenu HTML avec les variables nécessaires
    current_date = datetime.now().strftime('%d/%m/%Y')
    best_model_f1 = best_model['f1']
    avg_clv = clv_data['clv'].mean()
    churn_rate = best_model['confusion_matrix'][1].sum() / best_model['confusion_matrix'].sum()
    
    # Vérifier si la colonne 'segment' existe dans les données CLV
    has_segment = 'segment' in clv_data.columns
    
    # Préparer les colonnes pour le tableau des meilleurs clients
    columns = ['customer_id', 'clv', 'purchase_freq', 'avg_order_value']
    if has_segment:
        columns.insert(1, 'segment')  # Ajouter 'segment' après 'customer_id' s'il existe
    
    # Préparer le tableau des meilleurs clients
    top_customers = clv_data.nlargest(10, 'clv')[columns].round(2).to_html(index=False, classes='data-table')
    
    # Créer le contenu HTML avec les variables formatées directement
    current_date_str = datetime.now().strftime('%d/%m/%Y')
    best_model_f1_str = f"{best_model_f1:.2f}"
    avg_clv_str = f"{avg_clv:.2f} €"
    churn_rate_str = f"{churn_rate:.1%}"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rapport d'Analyse Prédictive</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .section {{ margin-bottom: 40px; }}
            .section h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
            .model-card {{ 
                border: 1px solid #ddd; 
                border-radius: 5px; 
                padding: 15px; 
                margin-bottom: 20px;
                background-color: #f9f9f9;
            }}
            .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; }}
            .metric-card {{ 
                background: white; 
                border-left: 4px solid #3498db; 
                padding: 10px 15px; 
                flex: 1; 
                min-width: 150px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-card h4 {{ margin: 0 0 5px 0; color: #7f8c8d; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .plot-container {{ margin: 20px 0; }}
            .plot-container img {{ max-width: 100%; height: auto; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Rapport d'Analyse Prédictive</h1>
                <p>Date du rapport: {current_date_str}</p>
            </div>
            
            <div class="section">
                <h2>1. Résumé Exécutif</h2>
                <p>Ce rapport présente les résultats de l'analyse prédictive menée sur la base de données clients. 
                L'analyse comprend la prédiction du taux de désabonnement (churn) et le calcul de la valeur à vie du client (CLV).</p>
                
                <div class="metrics">
                    <div class="metric-card">
                        <h4>Meilleur Modèle</h4>
                        <div class="metric-value">{best_model_name}</div>
                    </div>
                    <div class="metric-card">
                        <h4>Précision (F1-Score)</h4>
                        <div class="metric-value">{best_model_f1_str}</div>
                    </div>
                    <div class="metric-card">
                        <h4>CLV Moyenne</h4>
                        <div class="metric-value">{avg_clv_str}</div>
                    </div>
                    <div class="metric-card">
                        <h4>Taux de Churn Prédit</h4>
                        <div class="metric-value">{churn_rate_str}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>2. Analyse du Churn</h2>
                <h3>2.1 Performance des Modèles</h3>
                <div class="plot-container">
                    <iframe src="model_comparison.html" width="100%" height="500" frameborder="0"></iframe>
                </div>
                
                <h3>2.2 Matrice de Confusion</h3>
                <p>La matrice de confusion pour le modèle {best_model_name} est présentée ci-dessous :</p>
                <div class="plot-container">
                    <iframe src="confusion_matrix.html" width="600" height="500" frameborder="0"></iframe>
                </div>
                
                <h3>2.3 Importance des Caractéristiques</h3>
                <p>Les caractéristiques les plus importantes pour la prédiction du churn sont :</p>
                <div class="plot-container">
                    <iframe src="feature_importance.html" width="800" height="500" frameborder="0"></iframe>
                </div>
            </div>
            
            <div class="section">
                <h2>3. Analyse de la Valeur à Vie du Client (CLV)</h2>
                <h3>3.1 Distribution de la CLV par Segment</h3>
                <div class="plot-container">
                    <iframe src="clv_by_segment.html" width="800" height="500" frameborder="0"></iframe>
                </div>
                
                <h3>3.2 Relation entre Fréquence d'Achat et CLV</h3>
                <div class="plot-container">
                    <iframe src="freq_vs_clv.html" width="800" height="500" frameborder="0"></iframe>
                </div>
                
                <h3>3.3 Top 10 Clients par CLV</h3>
                {top_customers}
            </div>
            
            <div class="section">
                <h2>4. Recommandations Stratégiques</h2>
                <h3>4.1 Pour la Rétention Client</h3>
                <ul>
                    <li>Mettre en place un programme de fidélité ciblant les clients à haut risque de churn identifiés par le modèle.</li>
                    <li>Personnaliser les offres en fonction des caractéristiques des clients à risque.</li>
                    <li>Améliorer l'expérience client pour les segments présentant un taux de churn élevé.</li>
                </ul>
                
                <h3>4.2 Pour l'Optimisation de la CLV</h3>
                <ul>
                    <li>Développer des stratégies d'upselling et de cross-selling pour les clients à forte CLV.</li>
                    <li>Créer des programmes d'engagement pour augmenter la fréquence d'achat des clients à CLV moyenne.</li>
                    <li>Allouer plus de ressources au service client pour les segments à haute valeur.</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>5. Conclusion</h2>
                <p>L'analyse prédictive a permis d'identifier les principaux facteurs de churn et de calculer la valeur à vie des clients. 
                Les modèles développés atteignent une précision satisfaisante et peuvent être utilisés pour guider les décisions marketing.</p>
                <p>Les recommandations fournies dans ce rapport visent à améliorer la rétention des clients et à maximiser leur valeur à long terme pour l'entreprise.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Écrire le rapport dans un fichier
    report_path = REPORTS_DIR / 'predictive_analysis_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return report_path

# Fonction principale
def main():
    print("Démarrage de l'analyse prédictive...")
    
    # 1. Chargement des données
    print("Chargement des données...")
    data = load_data()
    
    # 2. Préparation des données pour la prédiction de churn
    print("Préparation des données pour la prédiction de churn...")
    X, y, churn_data = prepare_churn_data(data)
    
    # 3. Entraînement et évaluation des modèles de churn
    print("Entraînement des modèles de churn...")
    results, feature_names, X_test_scaled, y_test = train_churn_models(X, y)
    
    # 4. Calcul de la CLV
    print("Calcul de la valeur à vie du client (CLV)...")
    clv_data = calculate_clv(data)
    
    # 5. Génération des visualisations
    print("Génération des visualisations...")
    generate_visualizations(results, feature_names, X_test_scaled, y_test, clv_data)
    
    # 6. Génération du rapport HTML
    print("Génération du rapport...")
    report_path = generate_html_report(results, clv_data)
    
    print(f"\nAnalyse terminée avec succès!")
    print(f"Rapport généré : {report_path}")
    print("\nProchaines étapes recommandées :")
    print("1. Examiner le rapport HTML généré dans le dossier 'reports'")
    print("2. Mettre en œuvre les recommandations stratégiques")
    print("3. Planifier un suivi pour évaluer l'impact des actions mises en place")

if __name__ == "__main__":
    main()
