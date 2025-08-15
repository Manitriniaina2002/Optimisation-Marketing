import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_theme(style="whitegrid")

def load_data():
    """Load and return the datasets."""
    customers = pd.read_csv('customers_data.csv')
    products = pd.read_csv('products_data.csv')
    sales = pd.read_csv('sales_data.csv', parse_dates=['order_date'])
    return customers, products, sales

def clean_data(customers, products, sales):
    """Clean the datasets and return cleaned versions."""
    # Clean customers data
    customers_clean = customers.copy()
    customers_clean['age'].fillna(customers_clean['age'].median(), inplace=True)
    customers_clean['age'] = customers_clean['age'].astype(int)
    customers_clean['total_spent'] = customers_clean['total_spent'].clip(lower=0)
    
    # Clean products data
    products_clean = products.drop_duplicates()
    
    # Clean sales data
    sales_clean = sales[sales['quantity'] > 0].copy()
    sales_clean = sales_clean.merge(products_clean[['product_id', 'price']], on='product_id', how='left')
    sales_clean['total_amount'] = sales_clean['quantity'] * sales_clean['price']
    
    return customers_clean, products_clean, sales_clean

def explore_data(customers, products, sales):
    """Generate exploratory data analysis visualizations and statistics."""
    # 1. Customer Demographics
    plt.figure(figsize=(10, 6))
    sns.histplot(customers['age'], bins=15, color='#1f77b4')
    plt.title('Répartition des âges des clients')
    plt.xlabel('Âge')
    plt.ylabel('Nombre de clients')
    plt.savefig('age_distribution.png', bbox_inches='tight')
    plt.close()
    
    # 2. Sales by Product Category
    sales_by_category = sales.merge(products[['product_id', 'category']], on='product_id')
    sales_by_category = sales_by_category.groupby('category')['total_amount'].sum().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='category', y='total_amount', data=sales_by_category, palette='Set2')
    plt.title('Chiffre d\'affaires par catégorie de produit')
    plt.xlabel('Catégorie')
    plt.ylabel('CA total (Ar)')
    plt.savefig('sales_by_category.png', bbox_inches='tight')
    plt.close()
    
    # 3. Monthly Sales Trend
    sales['month'] = sales['order_date'].dt.to_period('M')
    monthly_sales = sales.groupby('month')['total_amount'].sum().reset_index()
    monthly_sales['month'] = monthly_sales['month'].astype(str)
    
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_sales['month'], monthly_sales['total_amount'], marker='o', color='#ff7f0e')
    plt.title('Évolution du CA mensuel')
    plt.xlabel('Mois')
    plt.ylabel('CA total (Ar)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('monthly_sales.png', bbox_inches='tight')
    plt.close()

def generate_summary(customers, products, sales):
    """Generate and print summary statistics."""
    print("=== RÉSUMÉ STATISTIQUE ===")
    
    # Customer Summary
    print("\n=== CLIENTS ===")
    print(f"Nombre total de clients: {len(customers)}")
    print(f"Âge moyen des clients: {customers['age'].mean():.1f} ans")
    print(f"Dépense moyenne par client: {customers['total_spent'].mean():,.2f} Ar")
    
    # Products Summary
    print("\n=== PRODUITS ===")
    print(f"Nombre total de produits: {len(products)}")
    print(f"Prix moyen: {products['price'].mean():,.2f} Ar")
    print("\nRépartition par catégorie:")
    print(products['category'].value_counts())
    
    # Sales Summary
    print("\n=== VENTES ===")
    print(f"Nombre total de ventes: {len(sales)}")
    print(f"CA total: {sales['total_amount'].sum():,.2f} Ar")
    print(f"Quantité moyenne par commande: {sales['quantity'].mean():.1f}")
    print(f"Panier moyen: {sales['total_amount'].mean():,.2f} Ar")
    
    # Save summary to file
    with open('data_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=== RAPPORT D'ANALYSE ===\n\n")
        f.write("1. STATISTIQUES DES CLIENTS\n")
        f.write(f"- Nombre total de clients: {len(customers)}\n")
        f.write(f"- Âge moyen: {customers['age'].mean():.1f} ans\n")
        f.write(f"- Dépense moyenne: {customers['total_spent'].mean():,.2f} Ar\n\n")
        
        f.write("2. STATISTIQUES DES PRODUITS\n")
        f.write(f"- Nombre total de produits: {len(products)}\n")
        f.write(f"- Prix moyen: {products['price'].mean():,.2f} Ar\n")
        f.write("- Répartition par catégorie:\n")
        f.write(products['category'].value_counts().to_string() + "\n\n")
        
        f.write("3. STATISTIQUES DES VENTES\n")
        f.write(f"- Nombre total de ventes: {len(sales)}\n")
        f.write(f"- CA total: {sales['total_amount'].sum():,.2f} Ar\n")
        f.write(f"- Panier moyen: {sales['total_amount'].mean():,.2f} Ar\n")
        f.write(f"- Quantité moyenne par commande: {sales['quantity'].mean():.1f}\n")

import os

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    """Main function to run the data exploration."""
    # Create output directory if it doesn't exist
    output_dir = 'output'
    ensure_dir(output_dir)
    
    print("Chargement des données...")
    customers, products, sales = load_data()
    
    print("Nettoyage des données...")
    customers_clean, products_clean, sales_clean = clean_data(customers, products, sales)
    
    print("Génération des visualisations...")
    explore_data(customers_clean, products_clean, sales_clean)
    
    print("Génération du rapport...")
    generate_summary(customers_clean, products_clean, sales_clean)
    
    # Save cleaned data
    print("\nSauvegarde des données nettoyées...")
    customers_clean.to_csv(os.path.join(output_dir, 'customers_clean.csv'), index=False)
    products_clean.to_csv(os.path.join(output_dir, 'products_clean.csv'), index=False)
    sales_clean.to_csv(os.path.join(output_dir, 'sales_clean.csv'), index=False)
    
    print(f"\nAnalyse terminée! Les résultats ont été enregistrés dans le dossier '{output_dir}'.")
    print("Fichiers générés:")
    print(f"- {os.path.join(output_dir, 'customers_clean.csv')}")
    print(f"- {os.path.join(output_dir, 'products_clean.csv')}")
    print(f"- {os.path.join(output_dir, 'sales_clean.csv')}")
    print("Visualisations:")
    print(f"- {os.path.join(os.getcwd(), 'age_distribution.png')}")
    print(f"- {os.path.join(os.getcwd(), 'sales_by_category.png')}")
    print(f"- {os.path.join(os.getcwd(), 'monthly_sales.png')}")
    print(f"Rapport: {os.path.join(os.getcwd(), 'data_summary.txt')}")

if __name__ == "__main__":
    main()
