import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features derived from transaction_timestamp."""
    
    if 'transaction_timestamp' not in df.columns:
        return df  # fail-safe, no-op

    # Hour of day (0–23)
    df['hour_of_day'] = df['transaction_timestamp'].dt.hour

    # Day of week (0 = Monday, 6 = Sunday)
    df['day_of_week'] = df['transaction_timestamp'].dt.dayofweek

    # Weekend flag (Saturday=5, Sunday=6)
    df['is_weekend'] = df['day_of_week'].isin([5, 6])

    # Night-time transaction (22:00 → 06:00)
    df['is_night'] = df['hour_of_day'].apply(lambda x: x >= 22 or x < 6)

    return df
