import pandas as pd

def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features derived from transaction_amount."""
    df = add_monthly_income(df)
    df = add_amount_vs_income_ratio(df)
    return df


def add_monthly_income(df:pd.DataFrame) -> pd.DataFrame:
    """Add monthly income feature derived from income column."""
    if 'income' in df.columns:
        df['monthly_income'] = df['income'] / 12
    return df

def add_amount_vs_income_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio of transaction amount to income."""
    """Acts as risk score feature already"""
    if 'transaction_amount' in df.columns and 'income' in df.columns:
        df['amount_vs_income_ratio'] = df['transaction_amount'] / df['monthly_income']
    return df