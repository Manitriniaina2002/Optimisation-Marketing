import logging, sys, os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('m6_predictive.log')])
logger = logging.getLogger(__name__)


def build_churn_dataset(customers, sales, horizon_days=90):
    # TODO: labellisation churn selon logique métier
    agg = sales.groupby('customer_id').agg(total_spent=('total_amount','sum'), orders=('order_id','nunique')).reset_index()
    df = customers.merge(agg, on='customer_id', how='left').fillna(0)
    df['churn'] = (df['orders'] == 0).astype(int)
    return df


def train_churn(df):
    X = df[['age','total_spent','orders']].fillna(0)
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)
    logger.info(f'AUC: {auc:.3f}')
    return clf


def main():
    os.makedirs('output', exist_ok=True)
    customers = pd.read_csv('output/customers_clean.csv')
    sales = pd.read_csv('output/sales_clean.csv')
    df = build_churn_dataset(customers, sales)
    _ = train_churn(df)


if __name__ == '__main__':
    main()
