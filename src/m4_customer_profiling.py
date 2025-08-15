import os
import sys
import logging
import traceback
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('m4_profiling.log')
    ]
)
logger = logging.getLogger(__name__)

# Set style for plots
sns.set_theme(style="whitegrid")

def check_data_files():
    """Check if required data files exist."""
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

def load_data():
    """Load cleaned data from M2."""
    try:
        logger.info("Loading customers data...")
        customers = pd.read_csv('output/customers_clean.csv')
        logger.info(f"Loaded {len(customers)} customers")
        
        logger.info("Loading products data...")
        products = pd.read_csv('output/products_clean.csv')
        logger.info(f"Loaded {len(products)} products")
        
        logger.info("Loading sales data...")
        sales = pd.read_csv('output/sales_clean.csv')
        logger.info(f"Loaded {len(sales)} sales records")
        
        return customers, products, sales
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None

def standardize_products(products: pd.DataFrame) -> pd.DataFrame:
    """Ensure products has 'product_id' and 'category' columns.
    Try common alternatives and create sensible fallbacks."""
    df = products.copy()
    cols = {c: c.lower() for c in df.columns}
    df.columns = [c.lower() for c in df.columns]

    # Find an ID-like column
    id_alts = ['product_id', 'sku', 'id', 'code', 'productcode', 'product id', 'product']
    id_col = next((c for c in id_alts if c in df.columns), None)
    if id_col is None:
        # Create a synthetic id if none exists
        df['product_id'] = [f"P{i+1}" for i in range(len(df))]
        logger.warning("M4: No product_id-like column found in products; created synthetic 'product_id'.")
    else:
        df['product_id'] = df[id_col]

    # Find a category-like column
    cat_alts = ['category', 'categorie', 'type', 'product_category']
    cat_col = next((c for c in cat_alts if c in df.columns), None)
    if cat_col is None:
        # Try using name/description as a proxy; else Unknown
        name_alts = ['name', 'product_name', 'designation', 'label']
        name_col = next((c for c in name_alts if c in df.columns), None)
        if name_col is not None:
            df['category'] = df[name_col]
            logger.warning("M4: No 'category' found; using product name as category proxy.")
        else:
            df['category'] = 'Unknown'
            logger.warning("M4: No 'category' or name-like column found; set category to 'Unknown'.")
    else:
        df['category'] = df[cat_col]

    # Keep only needed columns to avoid surprises
    return df[['product_id', 'category']].copy()

def standardize_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Ensure sales has required columns and consistent dtypes for merge/statistics."""
    df = sales.copy()
    df.columns = [c.lower() for c in df.columns]

    # product_id for join
    pid_alts = ['product_id', 'sku', 'item_id', 'product', 'product code']
    pid = next((c for c in pid_alts if c in df.columns), None)
    if pid is None:
        logger.warning("M4: No product reference in sales; creating placeholder 'product_id' = 'UNKNOWN'.")
        df['product_id'] = 'UNKNOWN'
    else:
        df['product_id'] = df[pid]

    # customer_id
    cid_alts = ['customer_id', 'client_id', 'cust_id']
    cid = next((c for c in cid_alts if c in df.columns), None)
    if cid is None:
        # Create synthetic if missing
        df['customer_id'] = [f"C{i+1}" for i in range(len(df))]
        logger.warning("M4: No customer_id in sales; created synthetic 'customer_id'.")
    else:
        df['customer_id'] = df[cid]

    # order_id
    oid_alts = ['order_id', 'id', 'sale_id', 'transaction_id']
    oid = next((c for c in oid_alts if c in df.columns), None)
    if oid is None:
        df['order_id'] = range(1, len(df) + 1)
        logger.warning("M4: No order_id in sales; generated sequential 'order_id'.")
    else:
        df['order_id'] = df[oid]

    # order_date
    date_alts = ['order_date', 'date', 'created_at']
    did = next((c for c in date_alts if c in df.columns), None)
    if did is None:
        df['order_date'] = pd.NaT
    else:
        df['order_date'] = pd.to_datetime(df[did], errors='coerce')

    # amounts
    if 'total_amount' not in df.columns:
        # Try compute from quantity * unit_price
        qty = next((c for c in ['quantity', 'qty', 'qte'] if c in df.columns), None)
        up = next((c for c in ['unit_price', 'price', 'prix'] if c in df.columns), None)
        if qty and up:
            df['total_amount'] = pd.to_numeric(df[qty], errors='coerce') * pd.to_numeric(df[up], errors='coerce')
        else:
            df['total_amount'] = pd.to_numeric(df.get('total_amount', 0), errors='coerce')

    # Ensure types as strings for merge keys
    df['product_id'] = df['product_id'].astype(str)
    df['customer_id'] = df['customer_id'].astype(str)
    return df

def analyze_segments(customers, sales, products):
    """Analyze customer segments and create personas."""
    # Safe getter for mode values
    def safe_mode(series, fallback='Unknown'):
        try:
            if series.empty:
                return fallback
            m = series.mode(dropna=True)
            return m.iloc[0] if len(m) > 0 else fallback
        except Exception:
            return fallback
    
    # Normalize products and sales schemas for robust merging
    products_std = standardize_products(products)
    sales_std = standardize_sales(sales)

    # Align dtype for join
    products_std['product_id'] = products_std['product_id'].astype(str)
    # Merge sales with products to get category information
    sales_std = sales_std.merge(products_std[['product_id', 'category']], on='product_id', how='left')
    
    # Calculate purchase frequency and recency
    customer_stats = sales_std.groupby('customer_id').agg(
        total_spent=('total_amount', 'sum'),
        purchase_count=('order_id', 'nunique'),
        avg_order_value=('total_amount', 'mean'),
        last_purchase=('order_date', 'max'),
        favorite_category=('category', safe_mode)  # Use the safe_mode function here
    ).reset_index()
    
    # Merge with customer data
    # Only include columns that actually exist in customers
    cust_cols = ['customer_id', 'age', 'city']
    if 'client_type' in customers.columns:
        cust_cols.append('client_type')
    customer_stats = customer_stats.merge(
        customers[cust_cols],
        on='customer_id',
        how='left'
    )
    # Ensure client_type exists for downstream logic
    if 'client_type' not in customer_stats.columns:
        customer_stats['client_type'] = 'Unknown'
    
    # Assign segments based on RFM (Recency, Frequency, Monetary)
    # This is a simplified version - in a real scenario, you might use the clusters from M3
    # Make robust in case of non-unique bin edges or limited unique values
    labels = ['Low Value', 'Medium Value', 'High Value']
    try:
        # Try qcut first
        customer_stats['segment'] = pd.qcut(
            customer_stats['total_spent'],
            q=3,
            labels=labels
        )
    except Exception:
        # Fallback: compute quantiles manually and use cut with unique bin edges
        qs = customer_stats['total_spent'].quantile([0.0, 1/3, 2/3, 1.0]).values
        # Ensure strictly increasing bin edges
        bins = []
        for v in qs:
            if not bins or v > bins[-1]:
                bins.append(v)
        # If after deduplication we have fewer than 2 bins, place everyone in a single segment
        if len(bins) <= 2:
            customer_stats['segment'] = pd.Series(['Medium Value'] * len(customer_stats), index=customer_stats.index, dtype='category')
        else:
            # Number of labels must be len(bins)-1
            use_labels = labels[:len(bins)-1]
            # If only two bins -> one label, map to Medium; if three bins -> Low/High; if four -> Low/Medium/High
            if len(use_labels) == 1:
                use_labels = ['Medium Value']
            elif len(use_labels) == 2:
                use_labels = ['Low Value', 'High Value']
            customer_stats['segment'] = pd.cut(customer_stats['total_spent'], bins=bins, labels=use_labels, include_lowest=True)
    
    return customer_stats

def create_personas(customer_stats):
    """Create detailed customer personas based on segments."""
    personas = {}
    
    # Drop NaN segments if any
    valid_segments = [s for s in customer_stats['segment'].dropna().unique()]
    for segment in valid_segments:
        segment_data = customer_stats[customer_stats['segment'] == segment]
        
        # Safe getters for mode values
        def safe_mode(series, fallback='Unknown'):
            try:
                m = series.mode(dropna=True)
                return m.iloc[0] if len(m) else fallback
            except Exception:
                return fallback
        
        persona = {
            'segment': segment,
            'count': len(segment_data),
            'avg_age': segment_data['age'].mean() if 'age' in segment_data.columns else float('nan'),
            'avg_spend': segment_data['total_spent'].mean(),
            'avg_orders': segment_data['purchase_count'].mean(),
            'top_city': safe_mode(segment_data['city']) if 'city' in segment_data.columns else 'Unknown',
            'top_category': safe_mode(segment_data['favorite_category']) if 'favorite_category' in segment_data.columns else 'Unknown',
            'common_client_type': safe_mode(segment_data['client_type']) if 'client_type' in segment_data.columns else 'Unknown'
        }
        
        personas[segment] = persona
    
    return personas

def visualize_personas(personas, output_dir='output'):
    """Generate visualizations for each persona."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert personas to DataFrame for easier plotting
    df = pd.DataFrame.from_dict(personas, orient='index')
    
    # Plot 1: Segment Distribution
    plt.figure(figsize=(10, 6))
    df['count'].plot(kind='bar', color='skyblue')
    plt.title('Distribution of Customer Segments')
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'segment_distribution.png'))
    plt.close()
    
    # Plot 2: Average Spend by Segment
    plt.figure(figsize=(10, 6))
    df['avg_spend'].plot(kind='bar', color='lightgreen')
    plt.title('Average Spend by Customer Segment')
    plt.ylabel('Average Spend (Ar)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_spend_by_segment.png'))
    plt.close()

def generate_report(personas, output_file='output/customer_personas_report.txt'):
    """Generate a text report of customer personas."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("CUSTOMER PERSONAS REPORT\n")
        f.write("Generated on: {}\n\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        total = sum(p['count'] for p in personas.values()) or 1
        for segment, persona in personas.items():
            f.write("="*50 + "\n")
            f.write(f"PERSONA: {segment} CUSTOMERS\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Profile:\n")
            f.write(f"- Segment Size: {persona['count']} customers ({persona['count']/total:.1%} of total)\n")
            f.write(f"- Average Age: {persona['avg_age']:.1f} years\n")
            f.write(f"- Average Total Spend: {persona['avg_spend']:,.0f} Ar\n")
            f.write(f"- Average Number of Orders: {persona['avg_orders']:.1f}\n")
            f.write(f"- Most Common City: {persona['top_city']}\n")
            f.write(f"- Favorite Category: {persona['top_category']}\n")
            f.write(f"- Client Type: {persona['common_client_type']}\n\n")
            
            f.write("Marketing Recommendations:\n")
            if segment == 'High Value':
                f.write("- Offer exclusive products or early access to new collections\n")
                f.write("- Create a loyalty program with premium benefits\n")
                f.write("- Personalized recommendations based on past purchases\n")
            elif segment == 'Medium Value':
                f.write("- Bundle products they frequently purchase together\n")
                f.write("- Limited-time offers to encourage more frequent purchases\n")
                f.write("- Email campaigns featuring their favorite categories\n")
            else:  # Low Value
                f.write("- Welcome discounts for first-time buyers\n")
                f.write("- Educational content about your products\n")
                f.write("- Entry-level products at competitive prices\n")
            
            f.write("\n")

def main():
    logger.info("="*50)
    logger.info("STARTING M4: CUSTOMER PROFILING AND PERSONA CREATION")
    logger.info("="*50)
    
    # Check data files first
    logger.info("Checking for required data files...")
    if not check_data_files():
        logger.error("Exiting due to missing data files.")
        return
    
    # Load data
    logger.info("Loading data...")
    customers, products, sales = load_data()
    if customers is None or products is None or sales is None:
        logger.error("Failed to load one or more data files.")
        return
        
    logger.info("Data loaded successfully.")
    logger.info(f"Customers shape: {customers.shape}")
    logger.info(f"Products shape: {products.shape}")
    logger.info(f"Sales shape: {sales.shape}")
    
    try:
        # Analyze segments
        logger.info("Analyzing customer segments...")
        customer_stats = analyze_segments(customers, sales, products)
        logger.info(f"Analyzed {len(customer_stats)} customer records")
        
        # Create personas
        logger.info("Creating customer personas...")
        personas = create_personas(customer_stats)
        logger.info(f"Created {len(personas)} customer personas")
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        visualize_personas(personas)
        
        # Generate report
        logger.info("Generating report...")
        generate_report(personas)
        
        logger.info("\nANALYSIS COMPLETE!")
        logger.info("Generated files:")
        logger.info("- output/segment_distribution.png")
        logger.info("- output/avg_spend_by_segment.png")
        logger.info("- output/customer_personas_report.txt")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return

if __name__ == "__main__":
    main()
