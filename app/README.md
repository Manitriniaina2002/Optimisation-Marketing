# Application d'Analyse Marketing

Cette application web interactive permet d'analyser les données marketing, de segmenter la clientèle et de générer des recommandations stratégiques.

## Fonctionnalités

- Importation de documents explicatifs (Word, PowerPoint, PDF)
- Chargement des fichiers de données (CSV)
- Exploration interactive des données
- Segmentation client avancée
- Analyse des performances marketing
- Génération de stratégies personnalisées
- Tableaux de bord interactifs

## Prérequis

- Python 3.8+
- pip (gestionnaire de paquets Python)

## Installation

1. Clonez ce dépôt :
   ```bash
   git clone [URL_DU_REPO]
   cd projet-IA/app
   ```

2. Créez un environnement virtuel (recommandé) :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows : .\venv\Scripts\activate
   ```

3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

1. Lancez l'application :
   ```bash
   streamlit run app.py
   ```

2. Ouvrez votre navigateur à l'adresse indiquée (généralement http://localhost:8501)

3. Suivez les étapes dans l'interface :
   - Importez vos documents explicatifs
   - Chargez les fichiers de données requis
   - Explorez les visualisations
   - Analysez les segments clients
   - Générez des rapports stratégiques

## Structure du projet

```
app/
├── pages/           # Modules séparés pour chaque étape
├── utils/           # Fonctions utilitaires
├── assets/          # Images et fichiers statiques
├── app.py           # Point d'entrée principal
└── requirements.txt # Dépendances Python
```

## Contribution

1. Forkez le projet
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Poussez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## Licence

Distribué sous la licence MIT. Voir `LICENSE` pour plus d'informations.

## Contact

[Votre nom] - [votre.email@exemple.com]

Lien du projet : [https://github.com/votrenom/projet-ia](https://github.com/votrenom/projet-ia)
