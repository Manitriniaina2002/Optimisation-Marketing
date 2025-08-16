"""
Module 9: Préparation des Livrables Finaux (M9)

Ce script prépare les livrables finaux du projet, y compris un rapport consolidé,
une présentation et les fichiers nécessaires pour le déploiement.
"""

import os
import shutil
import sys
import subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path
import markdown

# Vérifier si WeasyPrint est disponible
HAS_WEASYPRINT = False
try:
    from weasyprint import HTML
    HAS_WEASYPRINT = True
except (ImportError, OSError) as e:
    print(f"Avertissement : Impossible d'importer WeasyPrint. La génération des PDF sera désactivée.\nErreur : {e}")
    print("Pour activer la génération des PDF, installez les dépendances système requises :")
    print("Sur Windows : Téléchargez et installez GTK+ depuis https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer")
    print("Puis réinstallez WeasyPrint avec : pip install weasyprint")

# Configuration des chemins
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data_processed'
REPORTS_DIR = BASE_DIR / 'reports'
SLIDES_DIR = BASE_DIR / 'slides'
FINAL_DELIVERABLES_DIR = BASE_DIR / 'livrables_finaux'

# Créer le répertoire des livrables s'il n'existe pas
os.makedirs(FINAL_DELIVERABLES_DIR, exist_ok=True)
os.makedirs(SLIDES_DIR, exist_ok=True)

def create_final_report():
    """Crée un rapport final consolidé au format HTML et PDF."""
    print("Création du rapport final...")
    
    # Lire le contenu des rapports existants
    with open(REPORTS_DIR / 'predictive_analysis_report.html', 'r', encoding='utf-8') as f:
        predictive_analysis = f.read()
    
    with open(REPORTS_DIR / 'digital_marketing_strategy.html', 'r', encoding='utf-8') as f:
        marketing_strategy = f.read()
    
    # Créer le contenu HTML du rapport final
    current_date = datetime.now().strftime("%d/%m/%Y")
    
    # Échapper les accolades dans le contenu des rapports
    predictive_analysis_escaped = predictive_analysis.replace('{', '{{').replace('}', '}}')
    marketing_strategy_escaped = marketing_strategy.replace('{', '{{').replace('}', '}}')
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Rapport Final - Projet d'Analyse Marketing</title>
    <meta charset="UTF-8">
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            line-height: 1.6; 
            margin: 0; 
            padding: 20px;
            color: #333;
        }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
            }}
            .header {{ 
                text-align: center; 
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 2px solid #3498db;
            }}
            .header h1 {{ 
                color: #2c3e50; 
                margin-bottom: 10px;
            }}
            .section {{ 
                margin-bottom: 40px; 
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .section h2 {{ 
                color: #2c3e50; 
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            .footer {{ 
                text-align: center; 
                margin-top: 40px; 
                padding-top: 20px; 
                border-top: 1px solid #ddd; 
                color: #7f8c8d;
                font-size: 0.9em;
            }}
            .toc {{ 
                background-color: #f0f7ff; 
                padding: 20px; 
                border-radius: 5px; 
                margin-bottom: 30px;
            }}
            .toc ul {{ 
                list-style-type: none; 
                padding-left: 20px;
            }}
            .toc li {{ 
                margin: 8px 0;
            }}
            .toc a {{ 
                color: #2980b9; 
                text-decoration: none;
            }}
            .toc a:hover {{ 
                text-decoration: underline;
            }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Rapport Final - Projet d'Analyse Marketing</h1>
            <p>Date du rapport : {current_date}</p>
        </div>
        
        <div class="toc">
            <h2>Table des matières</h2>
            <ul>
                <li><a href="#resume">1. Résumé Exécutif</a></li>
                <li><a href="#methodologie">2. Méthodologie</a></li>
                <li><a href="#analyse-predictive">3. Analyse Prédictive</a></li>
                <li><a href="#strategie-marketing">4. Stratégie Marketing Digitale</a></li>
                <li><a href="#recommandations">5. Recommandations</a></li>
                <li><a href="#annexes">6. Annexes</a></li>
            </ul>
        </div>
        
        <div id="resume" class="section">
            <h2>1. Résumé Exécutif</h2>
            <p>Ce rapport présente les résultats complets de l'analyse marketing menée sur la base de données clients. 
            Les analyses ont permis d'identifier des segments clients distincts, de prédire les risques de désabonnement 
            et de formuler des recommandations stratégiques pour optimiser les campagnes marketing.</p>
            
            <h3>Principales conclusions :</h3>
            <ul>
                <li>Identification de X segments clients distincts avec des comportements d'achat différents</li>
                <li>Prédiction du taux de churn avec une précision de Y%</li>
                <li>Opportunités d'optimisation des campagnes marketing identifiées</li>
                <li>Stratégies personnalisées pour chaque segment client</li>
            </ul>
        </div>
        
        <div id="methodologie" class="section">
            <h2>2. Méthodologie</h2>
            <p>Notre approche s'est articulée autour de plusieurs étapes clés :</p>
            
            <h3>2.1 Collecte et préparation des données</h3>
            <p>Les données ont été collectées à partir de plusieurs sources, nettoyées et préparées pour l'analyse. 
            Les variables clés incluent le comportement d'achat, les données démographiques et l'historique des interactions.</p>
            
            <h3>2.2 Segmentation des clients</h3>
            <p>Une analyse RFM (Récence, Fréquence, Montant) a été utilisée pour segmenter les clients en groupes homogènes 
            présentant des caractéristiques et des comportements similaires.</p>
            
            <h3>2.3 Modélisation prédictive</h3>
            <p>Des modèles d'apprentissage automatique ont été entraînés pour prédire la probabilité de churn des clients 
            et estimer leur valeur à vie (CLV).</p>
            
            <h3>2.4 Analyse stratégique</h3>
            <p>Les résultats des analyses ont été interprétés pour formuler des recommandations stratégiques personnalisées 
            pour chaque segment client.</p>
        </div>
        
        <div id="analyse-predictive" class="section">
            <h2>3. Analyse Prédictive</h2>
            {predictive_analysis_content}
        </div>
        
        <div id="strategie-marketing" class="section">
            <h2>4. Stratégie Marketing Digitale</h2>
            {marketing_strategy_content}
        </div>
            
        <div id="recommandations" class="section">
            <h2>5. Recommandations</h2>
            <p>Sur la base de nos analyses, nous recommandons les actions suivantes :</p>
            
            <h3>5.1 Pour les clients à haute valeur</h3>
            <ul>
                <li>Mettre en place un programme de fidélité premium</li>
                <li>Proposer des avantages exclusifs et personnalisés</li>
                <li>Affecter des responsables de compte dédiés</li>
            </ul>
            
            <h3>5.2 Pour les clients à risque de churn</h3>
            <ul>
                <li>Lancer des campagnes de rétention ciblées</li>
                <li>Proposer des offres spéciales pour les fidéliser</li>
                <li>Mener des enquêtes de satisfaction pour comprendre leurs préoccupations</li>
            </ul>
            
            <h3>5.3 Pour les clients inactifs</h3>
            <ul>
                <li>Lancer des campagnes de réactivation</li>
                <li>Proposer des incitations pour les ramener</li>
                <li>Personnaliser les communications en fonction de leur historique d'achat</li>
            </ul>
        </div>
        
        <div id="annexes" class="section">
            <h2>6. Annexes</h2>
            <h3>6.1 Dictionnaire des données</h3>
            <p>Description des variables utilisées dans l'analyse :</p>
            <ul>
                <li><strong>customer_id</strong> : Identifiant unique du client</li>
                <li><strong>recency</strong> : Nombre de jours depuis le dernier achat</li>
                <li><strong>frequency</strong> : Nombre d'achats sur la période</li>
                <li><strong>monetary</strong> : Dépense totale du client</li>
                <li><strong>RFM_Score</strong> : Score RFM global</li>
                <li><strong>Cluster</strong> : Segment d'appartenance du client</li>
                <li><strong>churn_probability</strong> : Probabilité de désabonnement</li>
            </ul>
            
            <h3>6.2 Détails techniques</h3>
            <p>Ce projet a été réalisé avec les technologies suivantes :</p>
            <ul>
                <li>Python 3.9+</li>
                <li>Pandas, NumPy, Scikit-learn pour l'analyse des données</li>
                <li>Matplotlib, Seaborn, Plotly pour la visualisation</li>
                <li>Streamlit pour le tableau de bord interactif</li>
            </ul>
        </div>
        
        <div class="footer">
            <p>Rapport généré le {current_date} | Projet d'Analyse Marketing</p>
            <p>© 2024 Équipe d'Analyse des Données</p>
        </div>
    </div>
</body>
</html>
""".format(
    current_date=current_date,
    predictive_analysis_content=predictive_analysis_escaped,
    marketing_strategy_content=marketing_strategy_escaped
)
    
    # Enregistrer le rapport HTML
    html_path = FINAL_DELIVERABLES_DIR / 'rapport_final_analyse_marketing.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Convertir en PDF si WeasyPrint est disponible
    if HAS_WEASYPRINT:
        try:
            pdf_path = FINAL_DELIVERABLES_DIR / 'rapport_final_analyse_marketing.pdf'
            HTML(string=html_content).write_pdf(pdf_path)
            print(f"Rapport PDF généré : {pdf_path}")
        except Exception as e:
            print(f"Erreur lors de la génération du PDF avec WeasyPrint : {e}")
            print("Le rapport PDF n'a pas pu être généré. Vous pouvez utiliser les fichiers HTML ou convertir manuellement.")
    else:
        print("\nGénération du PDF désactivée car WeasyPrint n'est pas correctement installé.")
        print("Pour générer un PDF, vous pouvez :")
        print("1. Installer les dépendances système requises pour WeasyPrint")
        print("2. Ouvrir le fichier HTML dans un navigateur et utiliser l'impression > Enregistrer au format PDF")
        print("3. Utiliser un service en ligne de conversion HTML vers PDF")
        
        # Créer un script batch pour faciliter la conversion manuelle
        batch_content = """@echo off
echo Ouverture du rapport HTML dans le navigateur par défaut...
start "" "%~dp0../livrables_finaux/rapport_final_analyse_marketing.html"
echo.
echo Pour générer un PDF :
echo 1. Appuyez sur Ctrl+P dans le navigateur
echo 2. Sélectionnez 'Enregistrer au format PDF' comme imprimante
echo 3. Enregistrez le fichier dans le dossier 'livrables_finaux'
pause
"""
        batch_path = FINAL_DELIVERABLES_DIR / 'generer_pdf_manuellement.bat'
        with open(batch_path, 'w') as f:
            f.write(batch_content)
        print(f"\nUn fichier 'generer_pdf_manuellement.bat' a été créé pour vous aider à générer le PDF manuellement.")
    
    print(f"Rapport HTML généré : {html_path}")
    return html_path

def create_presentation():
    """Crée une présentation à partir du rapport final."""
    print("Création de la présentation...")
    
    # Créer le contenu de la présentation au format Markdown
    slides_content = [
        "# Présentation des Résultats - Analyse Marketing\n\n## Sommaire\n1. Contexte et Objectifs\n2. Méthodologie\n3. Principaux Résultats\n4. Recommandations Stratégiques\n5. Prochaines Étapes",
        
        "## 1. Contexte et Objectifs\n\n### Contexte\n- Analyse des données clients pour une entreprise de vente au détail\n- Besoin de mieux comprendre le comportement des clients\n- Objectif d'améliorer les campagnes marketing\n\n### Objectifs\n- Segmenter la base clients\n- Prédire le risque de churn\n- Proposer des stratégies marketing personnalisées",
        
        "## 2. Méthodologie\n\n### Approche en 4 étapes\n1. **Collecte et Nettoyage des Données**\n   - Sources multiples de données clients\n   - Nettoyage et préparation des données\n\n2. **Segmentation des Clients**\n   - Analyse RFM (Récence, Fréquence, Montant)\n   - Clustering pour identifier des groupes homogènes\n\n3. **Modélisation Prédictive**\n   - Prédiction du risque de churn\n   - Estimation de la valeur à vie du client (CLV)\n\n4. **Analyse Stratégique**\n   - Recommandations personnalisées par segment\n   - Plan d'action marketing",
        
        "## 3. Principaux Résultats\n\n### Segmentation des Clients\n- X segments identifiés avec des profils distincts\n- Caractéristiques uniques pour chaque segment\n\n### Analyse Prédictive\n- Modèle de prédiction du churn avec une précision de X%\n- Identification des facteurs clés influençant le churn\n\n### Opportunités Clés\n- Segments à fort potentiel de croissance\n- Clients à risque nécessitant des actions de rétention",
        
        "## 4. Recommandations Stratégiques\n\n### Pour les Clients à Haute Valeur\n- Programme de fidélité premium\n- Avantages exclusifs et personnalisés\n\n### Pour les Clients à Risque de Churn\n- Campagnes de rétention ciblées\n- Offres spéciales pour les fidéliser\n\n### Pour les Clients Inactifs\n- Campagnes de réactivation\n- Incitations pour les ramener",
        
        "## 5. Prochaines Étapes\n\n### Court Terme (0-3 mois)\n- Mettre en œuvre les campagnes recommandées\n- Mettre en place un système de suivi des performances\n\n### Moyen Terme (3-6 mois)\n- Affiner les segments basés sur les nouvelles données\n- Optimiser les modèles prédictifs\n\n### Long Terme (6+ mois)\n- Développer des programmes de fidélisation avancés\n- Automatiser les campagnes marketing\n\n---\n\n## Questions et Réponses\n\nMerci de votre attention !"
    ]
    
    # Convertir chaque slide de Markdown à HTML
    slides_html = []
    for slide in slides_content:
        try:
            slide_html = markdown.markdown(slide, extensions=['extra'])
            slides_html.append(f'<div class="slide">\n{slide_html}\n</div>')
        except Exception as e:
            print(f"Erreur lors de la conversion d'une slide : {e}")
            continue
    
    # Créer le contenu HTML complet avec les accolades échappées pour le CSS
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Présentation - Analyse Marketing</title>
    <meta charset="UTF-8">
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            line-height: 1.6; 
            margin: 0; 
            padding: 40px;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
        }}
        .slide {{ 
            page-break-after: always;
            margin-bottom: 40px;
            padding: 20px;
            border: 1px solid #eee;
            border-radius: 5px;
        }}
        .slide:last-child {{ 
            page-break-after: auto;
        }}
        h1, h2, h3, h4, h5, h6 {{ 
            color: #2c3e50; 
        }}
        h1 {{ 
            font-size: 2.2em; 
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{ 
            font-size: 1.8em; 
            color: #2980b9;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }}
        ul, ol {{ 
            padding-left: 25px; 
        }}
        li {{ 
            margin: 10px 0; 
        }}
        .footer {{ 
            text-align: center; 
            margin-top: 40px; 
            padding-top: 20px; 
            border-top: 1px solid #ddd; 
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        @page {{ 
            size: A4 landscape;
            margin: 0;
        }}
        @media print {{
            body {{ 
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            .slide {{ 
                height: 100vh;
                margin: 0;
                padding: 20px;
                border: none;
                page-break-after: always;
            }}
        }}
    </style>
</head>
<body>
    {slides_content}
    <div class="footer">
        <p>Présentation générée le {current_date} | Projet d'Analyse Marketing</p>
    </div>
</body>
</html>
"""
    
    # Remplir le template avec le contenu
    html_content = html_template.format(
        slides_content='\n'.join(slides_html),
        current_date=datetime.now().strftime("%d/%m/%Y")
    )
    
    # Enregistrer le fichier Markdown
    md_path = SLIDES_DIR / 'presentation_analyse_marketing.md'
    os.makedirs(SLIDES_DIR, exist_ok=True)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n---\n'.join(slides_content))
    
    # Enregistrer la présentation HTML
    html_path = FINAL_DELIVERABLES_DIR / 'presentation_analyse_marketing.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Convertir en PDF si WeasyPrint est disponible
    if HAS_WEASYPRINT:
        try:
            pdf_path = FINAL_DELIVERABLES_DIR / 'presentation_analyse_marketing.pdf'
            HTML(string=html_content).write_pdf(pdf_path, stylesheets=[BASE_DIR / 'styles' / 'presentation.css'])
            print(f"Présentation PDF générée : {pdf_path}")
        except Exception as e:
            print(f"Erreur lors de la génération du PDF de la présentation avec WeasyPrint : {e}")
    else:
        print("\nGénération du PDF de la présentation désactivée car WeasyPrint n'est pas correctement installé.")
    
    print(f"Présentation HTML générée : {html_path}")
    return html_path

def package_deliverables():
    """Prépare un package avec tous les livrables finaux."""
    print("Préparation du package des livrables...")
    
    # Créer un répertoire pour le package
    package_dir = FINAL_DELIVERABLES_DIR / 'package_livrables'
    os.makedirs(package_dir, exist_ok=True)
    
    # Copier les fichiers importants
    files_to_copy = [
        (REPORTS_DIR / 'predictive_analysis_report.html', 'rapports/'),
        (REPORTS_DIR / 'digital_marketing_strategy.html', 'rapports/'),
        (FINAL_DELIVERABLES_DIR / 'rapport_final_analyse_marketing.html', ''),
        (FINAL_DELIVERABLES_DIR / 'rapport_final_analyse_marketing.pdf', ''),
        (FINAL_DELIVERABLES_DIR / 'presentation_analyse_marketing.html', ''),
        (FINAL_DELIVERABLES_DIR / 'presentation_analyse_marketing.pdf', ''),
        (DATA_DIR / 'customer_segments.csv', 'donnees/'),
        (DATA_DIR / 'churn_predictions.csv', 'donnees/'),
        (BASE_DIR / 'app' / 'dashboard.py', 'code/'),
        (BASE_DIR / 'notebook' / '05_predictive_modeling.py', 'code/'),
        (BASE_DIR / 'notebook' / '06_digital_marketing_strategy.py', 'code/')
    ]
    
    for src, dest_dir in files_to_copy:
        if src.exists():
            dest_path = package_dir / dest_dir
            os.makedirs(dest_path, exist_ok=True)
            shutil.copy2(src, dest_path)
    
    # Créer un fichier README pour le package
    readme_content = """# Livrables du Projet d'Analyse Marketing

## Contenu du package

### Rapports
- `rapports/predictive_analysis_report.html` : Rapport d'analyse prédictive
- `rapports/digital_marketing_strategy.html` : Stratégie marketing digitale
- `rapport_final_analyse_marketing.pdf` : Rapport final complet (PDF)
- `rapport_final_analyse_marketing.html` : Rapport final complet (HTML)
- `presentation_analyse_marketing.pdf` : Présentation des résultats (PDF)
- `presentation_analyse_marketing.html` : Présentation des résultats (HTML)

### Données
- `donnees/customer_segments.csv` : Segments clients avec scores RFM
- `donnees/churn_predictions.csv` : Prédictions de churn

### Code
- `code/dashboard.py` : Tableau de bord interactif Streamlit
- `code/05_predictive_modeling.py` : Script d'analyse prédictive
- `code/06_digital_marketing_strategy.py` : Script de stratégie marketing

## Comment utiliser

1. **Tableau de bord interactif** :
   ```bash
   cd code
   streamlit run dashboard.py
   ```

2. **Rapports** : Ouvrez les fichiers HTML dans un navigateur web ou les PDF avec un lecteur PDF.

3. **Données** : Les fichiers CSV peuvent être ouverts avec Excel, Google Sheets ou importés dans des outils d'analyse.

## Prérequis

- Python 3.8+
- Bibliothèques Python listées dans `requirements.txt`
- Navigateur web moderne pour les rapports HTML

## Contact

Pour toute question, veuillez contacter l'équipe d'analyse des données.
    """
    
    with open(package_dir / 'README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    # Créer un fichier requirements.txt
    requirements = """pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
streamlit>=1.0.0
weasyprint>=53.0
markdown>=3.3.0
    """
    
    with open(package_dir / 'requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements)
    
    # Créer une archive ZIP du package
    import zipfile
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = FINAL_DELIVERABLES_DIR / f'livrables_analyse_marketing_{timestamp}.zip'
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, package_dir)
                zipf.write(file_path, arcname)
    
    print(f"Package des livrables créé : {zip_path}")
    return zip_path

def main():
    """Fonction principale pour générer les livrables finaux."""
    print("=" * 70)
    print("PRÉPARATION DES LIVRABLES FINAUX - PROJET D'ANALYSE MARKETING")
    print("=" * 70)
    
    # Créer les répertoires nécessaires
    os.makedirs(FINAL_DELIVERABLES_DIR, exist_ok=True)
    
    try:
        # 1. Créer le rapport final
        print("\n" + "=" * 30)
        print("1. CRÉATION DU RAPPORT FINAL")
        print("=" * 30)
        report_path = create_final_report()
        
        # 2. Créer la présentation
        print("\n" + "=" * 30)
        print("2. CRÉATION DE LA PRÉSENTATION")
        print("=" * 30)
        presentation_path = create_presentation()
        
        # 3. Préparer le package des livrables
        print("\n" + "=" * 30)
        print("3. PRÉPARATION DU PACKAGE DES LIVRABLES")
        print("=" * 30)
        package_path = package_deliverables()
        
        print("\n" + "=" * 70)
        print("LIVRABLES GÉNÉRÉS AVEC SUCCÈS !")
        print("=" * 70)
        print(f"\nRapport final : {report_path}")
        print(f"Présentation : {presentation_path}")
        print(f"Package complet : {package_path}")
        
    except Exception as e:
        print(f"\nERREUR lors de la génération des livrables : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
