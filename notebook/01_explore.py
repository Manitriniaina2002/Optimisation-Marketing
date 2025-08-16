# ========================
# Module M2 - Exploration & Nettoyage
# TeeTech Design
# ========================

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io

# Configurer l'encodage de la console pour Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 📂 1. Chargement des fichiers
p = Path("../Data")  # Dossier où sont stockés les CSV

# Chargement avec le bon séparateur (;)
customers = pd.read_csv(p / "customers.csv", sep=";", encoding="utf-8")
orders = pd.read_csv(p / "orders.csv", sep=";", encoding="utf-8")
tshirts = pd.read_csv(p / "tshirts.csv", sep=";", encoding="utf-8")
marketing = pd.read_csv(p / "marketing.csv", sep=";", encoding="utf-8")

# ========================
# 2. Exploration initiale
# ========================

def explore_df(name, df):
    print(f"\n=== {name.upper()} ===")
    print(df.head(), "\n")
    print(df.info(), "\n")
    print("Valeurs manquantes :\n", df.isna().sum(), "\n")
    print("Doublons :", df.duplicated().sum(), "\n")
    print(df.describe(include='all'))

explore_df("Customers", customers)
explore_df("Orders", orders)
explore_df("Tshirts", tshirts)
explore_df("Marketing", marketing)

# ========================
# 3. Nettoyage
# ========================

# 🔹 Conversion des dates
date_cols = {
    "customers": ["signup_date"],
    "orders": ["order_date"],
    "marketing": ["date"]
}

for col in date_cols["customers"]:
    customers[col] = pd.to_datetime(customers[col], errors="coerce", dayfirst=True)
for col in date_cols["orders"]:
    orders[col] = pd.to_datetime(orders[col], errors="coerce", dayfirst=True)
for col in date_cols["marketing"]:
    marketing[col] = pd.to_datetime(marketing[col], errors="coerce", dayfirst=True)

# 🔹 Suppression des doublons
customers.drop_duplicates(inplace=True)
orders.drop_duplicates(inplace=True)
tshirts.drop_duplicates(inplace=True)
marketing.drop_duplicates(inplace=True)

# 🔹 Gestion des valeurs manquantes
if "age" in customers.columns:
    customers["age"] = customers["age"].fillna(customers["age"].median())

# Ne pas remplir les prix manquants ici, on le fera après la jointure avec tshirts
if "revenue" in marketing.columns:
    marketing["revenue"] = marketing["revenue"].fillna(0)

# 🔹 Filtrage des valeurs aberrantes
print("\n=== DEBUG: Avant filtrage des commandes ===")
print("Nombre de commandes avant filtrage:", len(orders))
print("Valeurs uniques dans 'price':", orders['price'].unique())

if "age" in customers.columns:
    customers = customers[(customers["age"] >= 15) & (customers["age"] <= 100)]

# Ne pas filtrer les prix négatifs car ils sont tous NaN pour l'instant
# On s'occupera des prix après la jointure avec tshirts

# ========================
# 4. Enrichissement
# ========================

# Jointure pour avoir les prix de base des t-shirts
print("\n=== DEBUG: Avant jointure avec tshirts ===")
print("Nombre de commandes:", len(orders))
print("Colonnes de orders:", orders.columns.tolist())
print("Colonnes de tshirts:", tshirts.columns.tolist())

orders_with_prices = orders.merge(tshirts[['tshirt_id', 'base_price']], on='tshirt_id', how='left')

print("\n=== DEBUG: Après jointure avec tshirts ===")
print("Nombre de commandes après jointure:", len(orders_with_prices))
print("Valeurs manquantes dans base_price:", orders_with_prices['base_price'].isna().sum())

# Utiliser le prix de base si le prix de la commande est manquant
if 'price' not in orders_with_prices.columns or orders_with_prices['price'].isna().all():
    orders_with_prices['price'] = orders_with_prices['base_price']

# Ajout du montant total de commande
orders_with_prices["amount"] = orders_with_prices["quantity"] * orders_with_prices["price"]

print("\n=== DEBUG: Avant jointure finale ===")
print("Nombre de commandes avec prix:", len(orders_with_prices))
print("Colonnes avant jointure finale:", orders_with_prices.columns.tolist())

# Jointure pour avoir infos clients + t-shirts dans un seul tableau de ventes
orders_full = (orders_with_prices
    .merge(customers, on="customer_id", how="left")
    .merge(tshirts, on="tshirt_id", how="left", suffixes=('', '_y')))

print("\n=== DEBUG: Après jointure finale ===")
print("Nombre de commandes après jointure finale:", len(orders_full))
print("Colonnes après jointure finale:", orders_full.columns.tolist())
print("\nAperçu des données avant sauvegarde:")
print(orders_full.head())

# ========================
# 5. Sauvegarde des fichiers nettoyés
# ========================

output_dir = Path("../data_clean")
output_dir.mkdir(exist_ok=True)

customers.to_csv(output_dir / "customers_clean.csv", index=False, encoding="utf-8")
orders_full.to_csv(output_dir / "orders_clean.csv", index=False, encoding="utf-8")
marketing.to_csv(output_dir / "marketing_clean.csv", index=False, encoding="utf-8")
tshirts.to_csv(output_dir / "tshirts_clean.csv", index=False, encoding="utf-8")

print("\n✅ Nettoyage terminé. Fichiers sauvegardés dans data_clean/")
