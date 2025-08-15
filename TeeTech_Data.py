import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Générer customers_data
np.random.seed(42)
customers_data = pd.DataFrame({
    'customer_id': range(1, 101),
    'age': np.random.randint(18, 50, 100),
    'city': random.choices(['Antananarivo', 'Toamasina', 'Fianarantsoa', 'Mahajanga', 'Toliara'], k=100),
    'type': random.choices(['Étudiant', 'Développeur', 'Startup', 'Passionné tech'], k=100),
    'total_spent': np.random.uniform(10000, 200000, 100).round(2)
})

# Générer products_data
products_data = pd.DataFrame({
    'product_id': range(1, 21),
    'design_name': [f'T-shirt Design {i}' for i in range(1, 21)],
    'category': random.choices(['IA', 'Code', 'Geek', 'Tech Pop'], k=20),
    'price': np.random.uniform(20000, 50000, 20).round(2)
})

# Générer sales_data
start_date = datetime(2025, 1, 1)
sales_data = pd.DataFrame({
    'order_id': range(1, 201),
    'customer_id': random.choices(customers_data['customer_id'], k=200),
    'product_id': random.choices(products_data['product_id'], k=200),
    'order_date': [start_date + timedelta(days=random.randint(0, 180)) for _ in range(200)],
    'quantity': np.random.randint(1, 5, 200),
})

# Ajouter des valeurs manquantes et incohérences pour tester le nettoyage
customers_data.loc[5:10, 'age'] = np.nan
customers_data.loc[15, 'total_spent'] = -5000
sales_data.loc[20:25, 'quantity'] = 0

# Sauvegarder en CSV
customers_data.to_csv('customers_data.csv', index=False)
products_data.to_csv('products_data.csv', index=False)
sales_data.to_csv('sales_data.csv', index=False)