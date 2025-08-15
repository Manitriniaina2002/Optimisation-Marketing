import logging, sys, pandas as pd, os, glob, io, csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('m2_data_preparation.log')])
logger = logging.getLogger(__name__)


def detect_file(patterns, candidates):
    for pat in patterns:
        for c in candidates:
            name = os.path.basename(c).lower()
            if pat in name:
                return c
    return None


def read_csv_robust(path: str, sample_size: int = 4096) -> pd.DataFrame:
    """Read CSV with delimiter sniffing and encoding fallback.
    Tries utf-8-sig, utf-8, latin-1, cp1252. Uses csv.Sniffer to detect delimiter.
    Falls back to pandas' engine with sep=None (python engine) if needed.
    """
    if os.path.getsize(path) == 0:
        # Empty file -> return empty DataFrame
        logging.warning(f"Empty file: {path}")
        return pd.DataFrame()

    # Read a sample for sniffing
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
    delimiters_to_try = [',', ';', '\t', '|']
    sniffed_delim = None
    last_error = None

    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc, errors='replace') as f:
                sample = f.read(sample_size)
                try:
                    sniff = csv.Sniffer().sniff(sample, delimiters=delimiters_to_try)
                    sniffed_delim = sniff.delimiter
                except Exception:
                    sniffed_delim = None
            # First attempt: use sniffed delimiter if available
            if sniffed_delim:
                try:
                    return pd.read_csv(path, encoding=enc, sep=sniffed_delim)
                except Exception as e:
                    last_error = e
            # Second attempt: let pandas infer delimiter (python engine)
            try:
                return pd.read_csv(path, encoding=enc, sep=None, engine='python')
            except Exception as e:
                last_error = e
        except Exception as e:
            last_error = e

    # As a last resort, try without encoding (system default)
    try:
        return pd.read_csv(path, sep=None, engine='python')
    except Exception as e:
        last_error = e
        logging.error(f"read_csv_robust failed for {path}: {e}")
        raise last_error


def load_raw():
    # Detect CSVs in data/
    candidates = glob.glob(os.path.join('data', '*.csv'))
    logger.info(f"Found CSVs in data/: {[os.path.basename(c) for c in candidates]}")
    files = {
        'customers': detect_file(['customer', 'customers'], candidates),
        'products': detect_file(['product', 'products', 'tshirt', 'tshirts', 't-shirts', 'tee'], candidates),
        'sales': detect_file(['sale', 'sales', 'order', 'orders', 'transactions'], candidates),
        'marketing': detect_file(['marketing', 'ads', 'facebook'], candidates),
    }
    out = {}
    for k, p in files.items():
        if p and os.path.exists(p):
            try:
                out[k] = read_csv_robust(p)
                logger.info(f"Loaded {k} from {p}: {out[k].shape}")
            except Exception as e:
                logger.error(f"Failed to load {k} from {p}: {e}")
        else:
            logger.warning(f"No detected file for {k}")
    return out


def clean_and_save(dfs):
    os.makedirs('output', exist_ok=True)
    # Helpers
    def normalize(df):
        df = df.copy()
        df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
        return df

    def coerce_numeric(df, cols):
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df

    def coerce_date(df, cols):
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors='coerce')
        return df

    # Standardize customers
    if 'customers' in dfs:
        df = normalize(dfs['customers'])
        mapping = {
            'customer_id': ['customer_id', 'id', 'client_id', 'cust_id'],
            'age': ['age'],
            'city': ['city', 'ville'],
            'signup_date': ['signup_date', 'registration_date', 'created_at'],
            'client_type': ['client_type', 'type', 'segment']
        }
        std = {}
        for target, alts in mapping.items():
            for a in alts:
                if a in df.columns:
                    std[target] = df[a]
                    break
        # Fallback to original normalized df if no mapped columns were found
        if len(std) == 0:
            logger.warning("Customers: aucun champ standard détecté. Sauvegarde fallback avec colonnes d'origine.")
            cust = df
        else:
            cust = pd.DataFrame(std)
        cust = coerce_numeric(cust, ['age'])
        cust = coerce_date(cust, ['signup_date'])
        cust.to_csv('output/customers_clean.csv', index=False)
        logger.info(f"Saved output/customers_clean.csv (rows={len(cust)}, cols={len(cust.columns)})")

    # Standardize products (tshirts)
    if 'products' in dfs:
        df = normalize(dfs['products'])
        mapping = {
            'product_id': ['product_id', 'id', 'sku'],
            'name': ['name', 'product_name', 'designation'],
            'category': ['category', 'categorie', 'type'],
            'price': ['price', 'unit_price', 'prix']
        }
        std = {}
        for target, alts in mapping.items():
            for a in alts:
                if a in df.columns:
                    std[target] = df[a]
                    break
        if len(std) == 0:
            logger.warning("Products: aucun champ standard détecté. Sauvegarde fallback avec colonnes d'origine.")
            prod = df
        else:
            prod = pd.DataFrame(std)
        prod = coerce_numeric(prod, ['price'])
        prod.to_csv('output/products_clean.csv', index=False)
        logger.info(f"Saved output/products_clean.csv (rows={len(prod)}, cols={len(prod.columns)})")

    # Standardize sales (orders)
    if 'sales' in dfs:
        df = normalize(dfs['sales'])
        mapping = {
            'order_id': ['order_id', 'id', 'sale_id', 'transaction_id'],
            'customer_id': ['customer_id', 'client_id', 'cust_id'],
            'product_id': ['product_id', 'sku', 'item_id'],
            'order_date': ['order_date', 'date', 'created_at'],
            'quantity': ['quantity', 'qty', 'qte'],
            'unit_price': ['unit_price', 'price', 'prix'],
            'total_amount': ['total_amount', 'amount', 'total', 'montant']
        }
        std = {}
        for target, alts in mapping.items():
            for a in alts:
                if a in df.columns:
                    std[target] = df[a]
                    break
        if len(std) == 0:
            logger.warning("Sales: aucun champ standard détecté. Sauvegarde fallback avec colonnes d'origine.")
            sales = df
        else:
            sales = pd.DataFrame(std)
        sales = coerce_date(sales, ['order_date'])
        sales = coerce_numeric(sales, ['quantity', 'unit_price', 'total_amount'])
        # Compute total_amount if missing
        if 'total_amount' not in sales.columns or sales['total_amount'].isna().all():
            if 'quantity' in sales.columns and 'unit_price' in sales.columns:
                sales['total_amount'] = sales['quantity'] * sales['unit_price']
        sales.to_csv('output/sales_clean.csv', index=False)
        logger.info(f"Saved output/sales_clean.csv (rows={len(sales)}, cols={len(sales.columns)})")

    # Standardize marketing
    if 'marketing' in dfs:
        df = normalize(dfs['marketing'])
        mapping = {
            'date': ['date', 'day', 'jour'],
            'channel': ['channel', 'canal', 'platform', 'plateforme'],
            'impressions': ['impressions', 'impr'],
            'clicks': ['clicks', 'clics'],
            'conversions': ['conversions', 'purchases', 'achats'],
            'cost': ['cost', 'spend', 'cout'],
            'revenue': ['revenue', 'revenu', 'ca']
        }
        std = {}
        for target, alts in mapping.items():
            for a in alts:
                if a in df.columns:
                    std[target] = df[a]
                    break
        if len(std) == 0:
            logger.warning("Marketing: aucun champ standard détecté. Sauvegarde fallback avec colonnes d'origine.")
            mkt = df
        else:
            mkt = pd.DataFrame(std)
        mkt = coerce_date(mkt, ['date'])
        mkt = coerce_numeric(mkt, ['impressions', 'clicks', 'conversions', 'cost', 'revenue'])
        # Default channel to Facebook if absent
        if 'channel' not in mkt.columns:
            mkt['channel'] = 'Facebook'
        mkt.to_csv('output/marketing_clean.csv', index=False)
        logger.info(f"Saved output/marketing_clean.csv (rows={len(mkt)}, cols={len(mkt.columns)})")


def main():
    logger.info('M2 – Exploration & Nettoyage')
    dfs = load_raw()
    clean_and_save(dfs)


if __name__ == '__main__':
    main()
