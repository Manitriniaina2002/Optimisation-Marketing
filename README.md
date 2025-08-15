# Optimisation Marketing - TeeTech Design

Ce projet fait partie du module M2 (Exploration des données) du cours d'Optimisation Marketing. L'objectif est de nettoyer, visualiser et résumer les données clients, produits et ventes de TeeTech Design, une entreprise de t-shirts à thème technologique.

## Structure du projet

```
Optimisation-Marketing/
├── data/                    # Dossiers pour les données brutes
│   ├── raw/                 # Données brutes initiales
│   └── processed/           # Données nettoyées (à ignorer par Git)
├── notebooks/               # Notebooks Jupyter pour l'analyse
├── src/                     # Code source Python
│   ├── data/                # Scripts de traitement des données
│   ├── visualization/       # Scripts de visualisation
│   └── utils/               # Utilitaires
├── reports/                 # Rapports et présentations
├── .gitignore               # Fichiers à ignorer par Git
└── README.md                # Ce fichier
```

## Installation

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/Manitriniaina2002/Optimisation-Marketing.git
   cd Optimisation-Marketing
   ```

2. **Créer un environnement virtuel** (recommandé) :
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Sur Windows
   ```

3. **Installer les dépendances** :
   ```bash
   pip install pandas numpy matplotlib seaborn
   ```

## Utilisation

1. **Générer les données d'exemple** :
   ```bash
   python TeeTech_Data.py
   ```

2. **Explorer les données** :
   ```bash
   python m2_data_exploration.py
   ```

## Structure des données

### Données clients (`customers_data.csv`)
- `customer_id` : Identifiant unique du client
- `age` : Âge du client
- `city` : Ville de résidence
- `type` : Type de client (Étudiant, Développeur, etc.)
- `total_spent` : Montant total dépensé (en Ariary)

### Données produits (`products_data.csv`)
- `product_id` : Identifiant unique du produit
- `design_name` : Nom du design du t-shirt
- `category` : Catégorie du produit (IA, Code, Geek, etc.)
- `price` : Prix unitaire (en Ariary)

### Données de ventes (`sales_data.csv`)
- `order_id` : Identifiant de commande
- `customer_id` : Référence au client
- `product_id` : Référence au produit
- `order_date` : Date de la commande
- `quantity` : Quantité commandée

## Auteurs

- [Votre nom]

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
Les instructions d'installation seront ajoutées ici.
