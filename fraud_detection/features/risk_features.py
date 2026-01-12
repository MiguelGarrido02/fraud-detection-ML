import pandas as pd

def add_risk_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add risk-related features to the DataFrame."""
    df = add_category_risk(df, cfg["risk_features"])
    df = add_channel_risk(df, cfg["risk_features"])
    df = entry_mode_risk(df, cfg["risk_features"])
    df = add_country_risk(df, cfg["risk_features"])
    return df


def add_category_risk(df: pd.DataFrame, risk_map: dict) -> pd.DataFrame:
    """Add risk level based on merchant_category."""
    df["category_risk_score"] = df["merchant_category"].map(risk_map["category_risk"]).astype(float).fillna(risk_map.get("default_risk", 0))
    return df

def add_channel_risk(df: pd.DataFrame, risk_map: dict) -> pd.DataFrame:
    """Add risk level based on channel."""
    df["channel_risk_score"] = df["channel"].map(risk_map["channel_risk"]).astype(float).fillna(risk_map.get("default_risk", 0))
    return df

def entry_mode_risk(df: pd.DataFrame, risk_map: dict) -> pd.DataFrame:
    """Add risk level based on entry_point."""
    df["entry_mode_risk_score"] = df["entry_mode"].map(risk_map["entry_mode_risk"]).astype(float).fillna(risk_map.get("default_risk", 0))
    return df

def add_country_risk(df: pd.DataFrame, risk_map: dict) -> pd.DataFrame:
    """Add risk level based on transaction_country."""
    df["country_risk_score"] = df["transaction_country"].map(risk_map["country_risk"]).astype(float).fillna(risk_map.get("default_risk", 0))
    return df