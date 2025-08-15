import logging, sys, os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('m3_segmentation.log')])
logger = logging.getLogger(__name__)


def build_features(customers, sales):
    # Exemple: RFM + AOV + diversité catégories
    rfm = sales.groupby('customer_id').agg(
        total_spent=('total_amount','sum'),
        frequency=('order_id','nunique'),
        aov=('total_amount','mean')
    ).reset_index()
    return customers.merge(rfm, on='customer_id', how='left')


def segment(df):
    features = ['age','total_spent','frequency','aov']
    X = df[features].fillna(0)
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = km.fit_predict(Xs)
    score = silhouette_score(Xs, labels)
    logger.info(f'Silhouette: {score:.3f}')
    df_out = df[['customer_id']].copy()
    df_out['cluster_id'] = labels
    return df_out


def main():
    os.makedirs('output', exist_ok=True)
    customers = pd.read_csv('output/customers_clean.csv')
    sales = pd.read_csv('output/sales_clean.csv')
    feats = build_features(customers, sales)
    seg = segment(feats)
    seg.to_csv('output/customer_segments.csv', index=False)


if __name__ == '__main__':
    main()
