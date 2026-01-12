import pandas as pd

def preprocess_base_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the base dataset"""
    df = parse_datetime_columns(df)
    df = cast_numeric_columns(df)
    df = cast_categorical_columns(df)
    df = handle_missing_values(df)
    return df

def parse_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse datetime columns in the dataset"""
    datetime_cols = ['transaction_timestamp', 'signup_date']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def cast_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast numeric columns to appropriate types"""
    numeric_cols = ['transaction_amount', 'age', 'income']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def cast_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical columns to 'category' dtype"""
    categorical_cols = ['region', 'merchant_category', 'transaction_status', 'channel', 'entry_mode', 'transaction_country', 'is_international']
    for col in categorical_cols:
        for cols in df.columns:
            if col in categorical_cols:
                df[col] = df[col].astype('category')
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values by removing rows with missing values"""
    df = df.dropna()
    return df



    