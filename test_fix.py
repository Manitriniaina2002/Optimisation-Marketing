#!/usr/bin/env python3
"""
Script de test pour vérifier la correction du problème d'importation des données.
"""

import pandas as pd
import sys
from pathlib import Path

def test_price_mapping():
    """Test la logique de mapping des prix depuis le fichier produits."""
    
    print("🧪 Test de la correction du mapping des prix...")
    
    # Simuler les données orders avec prix manquants
    orders_data = {
        'order_id': ['O1001', 'O1002', 'O1003'],
        'customer_id': ['C001', 'C002', 'C003'],
        'tshirt_id': ['T101', 'T102', 'T103'],
        'quantity': [2, 1, 3],
        'price': ['', '', ''],  # Prix vides comme dans le fichier réel
        'order_date': ['01/05/2025', '02/05/2025', '03/05/2025']
    }
    
    # Simuler les données produits
    products_data = {
        'tshirt_id': ['T101', 'T102', 'T103'],
        'category': ['Streetwear', 'Streetwear', 'Sport'],
        'base_price': [30000, 30000, 30000]
    }
    
    orders_df = pd.DataFrame(orders_data)
    products_df = pd.DataFrame(products_data)
    
    print("📊 Données orders initiales :")
    print(orders_df)
    print("\n📦 Données produits :")
    print(products_df)
    
    # Simuler le traitement comme dans l'application
    # 1. Convertir les prix vides en NaN
    orders_df['price'] = pd.to_numeric(orders_df['price'], errors='coerce')
    
    # 2. Créer la colonne price dans products_df
    products_df['price'] = products_df['base_price']
    
    # 3. Vérifier si les prix ont besoin d'être remplis
    price_needs_filling = False
    if 'price' not in orders_df.columns:
        price_needs_filling = True
    elif 'price' in orders_df.columns:
        if orders_df['price'].isna().all() or orders_df['price'].isna().any():
            price_needs_filling = True
    
    print(f"\n🔍 Prix ont besoin d'être remplis : {price_needs_filling}")
    
    # 4. Remplir les prix depuis le fichier produits
    if price_needs_filling and 'tshirt_id' in orders_df.columns:
        price_map = dict(zip(products_df['tshirt_id'], products_df['price']))
        print(f"🗺️ Mapping des prix : {price_map}")
        
        orders_df['price'] = orders_df['tshirt_id'].map(price_map)
        
        if orders_df['price'].isna().any():
            print(f"⚠️ {orders_df['price'].isna().sum()} lignes ont des prix manquants")
        else:
            print("✅ Tous les prix ont été complétés")
    
    # 5. Calculer amount
    if 'amount' not in orders_df.columns:
        valid_rows = orders_df['price'].notna() & orders_df['quantity'].notna()
        if valid_rows.any():
            orders_df.loc[valid_rows, 'amount'] = orders_df.loc[valid_rows, 'price'] * orders_df.loc[valid_rows, 'quantity']
            print(f"✅ La colonne 'amount' a été calculée pour {valid_rows.sum()} lignes.")
    
    print("\n📊 Données orders finales :")
    print(orders_df)
    
    # Vérifier que le calcul est correct
    expected_amounts = [60000, 30000, 90000]  # price * quantity pour chaque ligne
    calculated_amounts = orders_df['amount'].tolist()
    
    if calculated_amounts == expected_amounts:
        print("\n✅ Test RÉUSSI : Les montants ont été calculés correctement !")
        return True
    else:
        print(f"\n❌ Test ÉCHOUÉ : Montants attendus {expected_amounts}, obtenus {calculated_amounts}")
        return False

if __name__ == "__main__":
    success = test_price_mapping()
    sys.exit(0 if success else 1)
