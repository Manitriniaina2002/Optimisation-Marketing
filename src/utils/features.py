import pandas as pd


def compute_rfm(sales: pd.DataFrame) -> pd.DataFrame:
    rfm = sales.groupby('customer_id').agg(
        monetary=('total_amount','sum'),
        frequency=('order_id','nunique')
    ).reset_index()
    return rfm
