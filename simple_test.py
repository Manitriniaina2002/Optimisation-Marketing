import pandas as pd

# Test simple de la logique
orders_data = {
    'order_id': ['O1001', 'O1002'],
    'tshirt_id': ['T101', 'T102'],
    'quantity': [2, 1],
    'price': ['', '']  # Prix vides
}

products_data = {
    'tshirt_id': ['T101', 'T102'],
    'base_price': [30000, 30000]
}

orders_df = pd.DataFrame(orders_data)
products_df = pd.DataFrame(products_data)

print("Données initiales:")
print(orders_df)

# Logique de correction
orders_df['price'] = pd.to_numeric(orders_df['price'], errors='coerce')
products_df['price'] = products_df['base_price']

if orders_df['price'].isna().any():
    price_map = dict(zip(products_df['tshirt_id'], products_df['price']))
    orders_df['price'] = orders_df['tshirt_id'].map(price_map)
    print("Prix mappés depuis produits")

valid_rows = orders_df['price'].notna() & orders_df['quantity'].notna()
orders_df.loc[valid_rows, 'amount'] = orders_df.loc[valid_rows, 'price'] * orders_df.loc[valid_rows, 'quantity']

print("Données finales:")
print(orders_df)
print("Test réussi !" if not orders_df['amount'].isna().any() else "Test échoué")
