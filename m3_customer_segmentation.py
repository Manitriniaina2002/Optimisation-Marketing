import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pkg_resources
import traceback
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

class StringIOLogHandler(logging.Handler):
    """A custom logging handler that captures log messages in a StringIO buffer."""
    def __init__(self):
        super().__init__()
        self.buffer = []
        
    def emit(self, record):
        log_entry = self.format(record)
        self.buffer.append(log_entry)
        # Also print to console
        print(log_entry)
        
    def get_logs(self):
        return "\n".join(self.buffer)

def setup_logging():
    """Set up logging configuration with both console and in-memory handlers."""
    # Create logger
    logger = logging.getLogger('segmentation')
    logger.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Create in-memory handler
    mem_handler = StringIOLogHandler()
    mem_handler.setLevel(logging.DEBUG)
    mem_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(mem_handler)
    
    return logger, mem_handler

def check_data_files():
    """Check if required data files exist and are accessible."""
    required_files = [
        'output/customers_clean.csv',
        'output/products_clean.csv',
        'output/sales_clean.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        logger.error(f"Missing required data files: {', '.join(missing_files)}")
        logger.info(f"Current directory: {os.getcwd()}")
        logger.info(f"Files in output directory: {os.listdir('output') if os.path.exists('output') else 'No output directory'}")
        return False
    return True

def check_environment():
    """Check if required data files exist and are accessible."""
    required_files = ['customers_data.csv', 'products_data.csv', 'sales_data.csv']
    all_files_exist = True
    
    for file in required_files:
        file_path = os.path.abspath(file)
        exists = os.path.isfile(file)
        logger.info(f"Checking {file}: {'Found' if exists else 'Missing'}")
        if exists:
            logger.info(f"  Path: {file_path}")
            logger.info(f"  Size: {os.path.getsize(file)} bytes")
        else:
            logger.error(f"  Required file not found: {file}")
            all_files_exist = False
    
    if not all_files_exist:
        logger.error("\nCurrent directory contents:")
        for item in os.listdir('.'):
            item_path = os.path.join('.', item)
            logger.info(f"- {item} (file: {os.path.isfile(item_path)}, dir: {os.path.isdir(item_path)})")
    
    return all_files_exist

def load_and_prepare_data():
    """Load and prepare data for clustering."""
    logger.info("\n" + "="*30 + " LOADING DATA " + "="*30)
    try:
        logger.info("Loading customers data...")
        customers = pd.read_csv('customers_data.csv')
        logger.info(f"Loaded {len(customers)} customers")
        
        logger.info("Loading products data...")
        products = pd.read_csv('products_data.csv')
        logger.info(f"Loaded {len(products)} products")
        
        logger.info("Loading sales data...")
        sales = pd.read_csv('sales_data.csv')
        logger.info(f"Loaded {len(sales)} sales records")
        
        if customers is None or products is None or sales is None:
            logger.error("Failed to load required data. Exiting.")
            return
            
        logger.info("\nData loaded successfully:")
        logger.info(f"- Customers: {len(customers)} records")
        logger.info(f"- Products: {len(products)} records")
        logger.info(f"- Sales: {len(sales)} records")
        
        # Log sample data
        logger.info("\nSample customer data:" + str(customers.head(1).to_dict(orient='records')))
        logger.info("Sample product data:" + str(products.head(1).to_dict(orient='records')))
        logger.info("Sample sales data:" + str(sales.head(1).to_dict(orient='records')))
        
    except Exception as e:
        logger.exception("Error loading data:")
        return
    
    # Clean data (as in M2)
    customers['age'].fillna(customers['age'].median(), inplace=True)
    customers['total_spent'] = customers['total_spent'].clip(lower=0)
    sales = sales[sales['quantity'] > 0]
    sales['order_date'] = pd.to_datetime(sales['order_date'])
    
    # Merge sales with products to get categories
    sales = sales.merge(products[['product_id', 'category', 'price']], on='product_id', how='left')
    sales['total_amount'] = sales['quantity'] * sales['price']
    
    return customers, sales

def create_customer_features(customers, sales):
    """Create features for customer segmentation."""
    logger.info("\n" + "="*30 + " CREATING FEATURES " + "="*30)
    try:
        logger.info("Creating customer features...")
        # Basic customer features
        customer_features = customers[['customer_id', 'age', 'total_spent']].copy()
        
        # Purchase frequency (number of orders per customer)
        purchase_freq = sales.groupby('customer_id')['order_id'].count().reset_index(name='purchase_count')
        
        # Average order value
        avg_order_value = sales.groupby('customer_id')['total_amount'].mean().reset_index(name='avg_order_value')
        
        # Days since last purchase
        last_purchase = sales.groupby('customer_id')['order_date'].max().reset_index()
        last_purchase['days_since_last_purchase'] = (pd.Timestamp.now() - last_purchase['order_date']).dt.days
        
        # Favorite category (most purchased category)
        category_pref = sales.groupby(['customer_id', 'category'])['quantity'].sum().reset_index()
        category_pref = category_pref.loc[category_pref.groupby('customer_id')['quantity'].idxmax()]
        category_pref = category_pref[['customer_id', 'category']].rename(columns={'category': 'favorite_category'})
        
        # Merge all features
        customer_features = customer_features.merge(purchase_freq, on='customer_id', how='left')
        customer_features = customer_features.merge(avg_order_value, on='customer_id', how='left')
        customer_features = customer_features.merge(last_purchase[['customer_id', 'days_since_last_purchase']], on='customer_id', how='left')
        customer_features = customer_features.merge(category_pref, on='customer_id', how='left')
        
        # Fill missing values
        customer_features['purchase_count'].fillna(0, inplace=True)
        customer_features['avg_order_value'].fillna(0, inplace=True)
        customer_features['days_since_last_purchase'].fillna(365, inplace=True)  # If never purchased, set to 1 year
        customer_features['favorite_category'].fillna('None', inplace=True)
        
        if customer_features is None or customer_features.empty:
            logger.error("Failed to create customer features or features are empty. Exiting.")
            return
            
        logger.info("\nFeature creation completed successfully:")
        logger.info(f"Number of customers: {len(customer_features)}")
        logger.info("Feature columns: " + ", ".join(customer_features.columns))
        logger.info("\nSample features:" + str(customer_features.head(2).to_dict(orient='records')))
        
    except Exception as e:
        logger.exception("Error creating features:")
        return
    
    return customer_features

def prepare_for_clustering(customer_features):
    """Prepare data for clustering by encoding and scaling."""
    logger.info("\n" + "="*30 + " PREPARING FOR CLUSTERING " + "="*30)
    try:
        logger.info("Preparing for clustering...")
        # One-hot encode favorite category
        features_encoded = pd.get_dummies(customer_features, columns=['favorite_category'], prefix='cat')
        
        # Select features for clustering
        feature_columns = ['age', 'total_spent', 'purchase_count', 'avg_order_value', 'days_since_last_purchase']
        feature_columns += [col for col in features_encoded.columns if col.startswith('cat_')]
        
        X = features_encoded[feature_columns]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        logger.info(f"Features shape: {X.shape}")
        logger.debug(f"Sample features: {X[:2]}")
        
    except Exception as e:
        logger.exception("Error preparing for clustering:")
        return
    
    return X_scaled, features_encoded, feature_columns

def find_optimal_clusters(X_scaled, max_clusters=10):
    """Find the optimal number of clusters using the elbow method and silhouette score."""
    logger.info("\n" + "="*30 + " FINDING OPTIMAL CLUSTERS " + "="*30)
    try:
        logger.info("Determining optimal number of clusters...")
        # Calculate inertia for different numbers of clusters
        inertia = []
        silhouette_scores = []
        K = range(2, max_clusters + 1)
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)
            
            # Calculate silhouette score
            if k > 1:  # Silhouette score requires at least 2 clusters
                score = silhouette_score(X_scaled, kmeans.labels_)
                silhouette_scores.append(score)
        
        # Plot elbow curve and silhouette scores
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Elbow method
        plt.subplot(1, 2, 1)
        plt.plot(K, inertia, 'bo-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        
        # Plot 2: Silhouette scores
        plt.subplot(1, 2, 2)
        plt.plot(K, silhouette_scores, 'ro-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis')
        
        plt.tight_layout()
        plt.savefig('output/clustering_metrics.png')
        plt.close()
        
        # Find the optimal number of clusters (max silhouette score)
        optimal_k = K[np.argmax(silhouette_scores)]
        logger.info(f"\nOptimal number of clusters: {optimal_k} (based on silhouette score)")
        
        if optimal_k is None:
            logger.error("Failed to determine optimal number of clusters. Exiting.")
            return
            
    except Exception as e:
        logger.exception("Error finding optimal clusters:")
        return
    
    return optimal_k

def perform_clustering(X_scaled, n_clusters, features_encoded):
    """Perform K-means clustering and add cluster labels to the data."""
    logger.info("\n" + "="*30 + " PERFORMING CLUSTERING " + "="*30)
    try:
        logger.info(f"Performing K-means clustering with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to the features DataFrame
        features_encoded['cluster'] = cluster_labels
        
        logger.info(f"\nClustering completed. Cluster sizes:\n{features_encoded['cluster'].value_counts().sort_index()}")
        
        # Log cluster distribution
        cluster_counts = features_encoded['cluster'].value_counts().sort_index()
        logger.info("\nCluster distribution:")
        for cluster, count in cluster_counts.items():
            logger.info(f"- Cluster {cluster}: {count} customers ({count/len(features_encoded):.1%})")
            
        if features_encoded is None or kmeans is None:
            logger.error("Clustering failed. Exiting.")
            return
            
    except Exception as e:
        logger.exception("Error during clustering:")
        return
    
    return features_encoded, kmeans

def visualize_clusters(features_encoded, kmeans):
    """Visualize clusters using PCA for dimensionality reduction."""
    logger.info("\n" + "="*30 + " VISUALIZING CLUSTERS " + "="*30)
    try:
        logger.info("Visualizing clusters...")
        # Get the features used for clustering
        X = features_encoded.drop(['customer_id', 'cluster'], axis=1, errors='ignore')
        
        # Apply PCA for 2D visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        
        # Create a DataFrame with PCA results
        pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
        pca_df['cluster'] = features_encoded['cluster']
        
        # Plot clusters
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=pca_df, palette='viridis', s=100, alpha=0.7)
        plt.title('Visualisation des clusters de clients (PCA)')
        plt.xlabel(f'Composante Principale 1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'Composante Principale 2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.legend(title='Cluster')
        plt.grid(True, alpha=0.3)
        plt.savefig('output/customer_clusters_pca.png')
        plt.close()
        
        logger.info("Cluster visualization generated successfully")
        
    except Exception as e:
        logger.exception("Error visualizing clusters:")
        return

def analyze_clusters(features_encoded):
    """Analyze and describe each cluster."""
    logger.info("\n" + "="*30 + " ANALYZING CLUSTERS " + "="*30)
    try:
        logger.info("Analyzing clusters...")
        # Group by cluster and calculate statistics
        cluster_analysis = features_encoded.groupby('cluster').agg({
            'age': ['mean', 'std', 'count'],
            'total_spent': ['mean', 'sum'],
            'purchase_count': ['mean', 'sum'],
            'avg_order_value': 'mean',
            'days_since_last_purchase': 'mean'
        }).round(2)
        
        # Add percentage of total customers
        cluster_analysis[('percentage', '')] = (cluster_analysis[('age', 'count')] / len(features_encoded) * 100).round(1).astype(str) + '%'
        
        # Save cluster analysis to CSV
        analysis_path = os.path.join('output', 'cluster_analysis.csv')
        cluster_analysis.to_csv(analysis_path, index=False)
        
        logger.info(f"Saved cluster analysis to {analysis_path}")
        
    except Exception as e:
        logger.exception("Error analyzing clusters:")
        return

def main():
    # Set up logging with both console and in-memory handlers
    logger, mem_handler = setup_logging()
    
    # Store logs in a list for later reference
    logs = []
    
    def log_info(msg):
        logger.info(msg)
        logs.append(f"INFO: {msg}")
        
    def log_error(msg):
        logger.error(msg)
        logs.append(f"ERROR: {msg}")
        
    def log_debug(msg):
        logger.debug(msg)
        logs.append(f"DEBUG: {msg}")
    
    log_info("="*70)
    log_info("STARTING CUSTOMER SEGMENTATION ANALYSIS")
    log_info("="*70)
    
    # Log system information
    log_info(f"Python executable: {sys.executable}")
    log_info(f"Working directory: {os.getcwd()}")
    log_info(f"Command line: {' '.join(sys.argv)}")
    
    # Log environment variables that might affect execution
    log_info("\nEnvironment variables:")
    for var in ['PATH', 'PYTHONPATH', 'PYTHONHOME']:
        log_info(f"  {var}: {os.environ.get(var, 'Not set')}")
    
    # Log environment information
    log_info(f"Working directory: {os.getcwd()}")
    log_info(f"Python executable: {sys.executable}")
    log_info(f"Python version: {sys.version}")
    
    # Log installed packages
    try:
        import pkg_resources
        logger.info("\nInstalled packages:")
        for pkg in sorted(pkg_resources.working_set, key=lambda x: x.key):
            logger.info(f"  {pkg.key}=={pkg.version}")
    except Exception as e:
        logger.warning(f"Could not list installed packages: {e}")
    
    # Create output directory if it doesn't exist
    output_dir = 'output'
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"\nOutput directory: {os.path.abspath(output_dir)}")
        logger.info(f"Output directory exists: {os.path.exists(output_dir)}")
        logger.info(f"Output directory is writable: {os.access(output_dir, os.W_OK)}")
        logger.info(f"Contents of output directory: {os.listdir(output_dir)}")
    except Exception as e:
        logger.error(f"Error creating/accessing output directory: {e}")
        return
    
    # Check for required data files
    logger.info("\nChecking for required data files...")
    if not check_environment():
        logger.error("Exiting due to missing data files.")
        # Print all logs before exiting
        print("\n" + "="*70)
        print("FULL EXECUTION LOG:")
        print("="*70)
        print("\n".join(logs))
        return
    
    print("\nChargement et préparation des données...")
    customers, sales = load_and_prepare_data()
    
    print("Création des caractéristiques clients...")
    customer_features = create_customer_features(customers, sales)
    
    print("Préparation pour le clustering...")
    X_scaled, features_encoded, feature_columns = prepare_for_clustering(customer_features)
    
    print("Recherche du nombre optimal de clusters...")
    optimal_k = find_optimal_clusters(X_scaled, max_clusters=6)
    print(f"Nombre optimal de clusters suggéré : {optimal_k}")
    
    print(f"Exécution du clustering avec k={optimal_k}...")
    features_with_clusters, kmeans = perform_clustering(X_scaled, optimal_k, features_encoded)
    
    print("Visualisation des clusters...")
    visualize_clusters(features_with_clusters, kmeans)
    
    print("Analyse des clusters...")
    analyze_clusters(features_with_clusters)
    
    # Save final results
    logger.info("Saving results...")
    try:
        segments_path = os.path.join(output_dir, 'customer_segments.csv')
        features_with_clusters.to_csv(segments_path, index=False)
        logger.info(f"Saved customer segments to {segments_path}")
        
        logger.info("="*50)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("="*50)
        logger.info("Generated files in 'output' directory:")
        for f in os.listdir(output_dir):
            logger.info(f"- {f}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        logger.error(traceback.format_exc())
        return

    # Print all logs at the end
    print("\n" + "="*70)
    print("FULL EXECUTION LOG:")
    print("="*70)
    print(mem_handler.get_logs())

if __name__ == "__main__":
    import os
    import sys
    import logging
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    try:
        main()
    except Exception as e:
        print(f"\nERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
