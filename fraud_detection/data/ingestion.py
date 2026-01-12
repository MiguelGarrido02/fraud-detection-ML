import pandas as pd

def build_base_dataset(paths_cfg: dict) -> pd.DataFrame:
    """Build the base dataset by merging customers and transactions raw data"""
    customers_df = load_costumers_data(paths_cfg["raw_customers_data_path"])
    transactions_df = load_transactions_data(paths_cfg["raw_transactions_data_path"])
    
    base_df = transactions_df.merge(
        customers_df,
        how='left',
        on='customer_id',
        validate='many_to_one'
    )
    
    return base_df

def load_costumers_data(path: str) -> pd.DataFrame:
    """Load customers data from a CSV file."""
    return pd.read_csv(path)

def load_transactions_data(path: str) -> pd.DataFrame:
    """Load transactions data from a CSV file."""
    return pd.read_csv(path)


