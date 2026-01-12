import pandas as pd

def apply_fraud_rules(df: pd.DataFrame, rules_cfg: dict) -> pd.DataFrame:
    """Apply fraud detection rules to the DataFrame."""
    weights_cfg = rules_cfg["fraud_scoring"]["weights"]
    scoring_cfg = rules_cfg["fraud_scoring"]
    thresholds_cfg = rules_cfg["fraud_scoring"]["thresholds"]
    labeling_cfg = rules_cfg["fraud_scoring"]["labeling"]

    # Caulculate fraud score based on weighted features
    df["fraud_score"] = (

        df["amount_vs_income_ratio"].where(
            df["amount_vs_income_ratio"] >= thresholds_cfg["min_amount_vs_income_ratio"], 0.0)  * weights_cfg.get("amount_vs_income_ratio_weight", 1.0) +
        df["category_risk_score"].where(
            df["category_risk_score"] >= thresholds_cfg["min_category_risk_score"], 0.0) * weights_cfg.get("category_risk_score_weight", 1.0) +
        df["channel_risk_score"].where(
            df["channel_risk_score"] >= thresholds_cfg["min_channel_risk_score"], 0.0) * weights_cfg.get("channel_risk_score_weight", 1.0) +
        df["entry_mode_risk_score"].where(
            df["entry_mode_risk_score"] >= thresholds_cfg["min_entry_mode_risk_score"], 0.0) * weights_cfg.get("entry_mode_risk_score_weight", 1.0) +
        df["country_risk_score"].where(
            df["country_risk_score"] >= thresholds_cfg["min_country_risk_score"], 0.0) * weights_cfg.get("country_risk_score_weight", 1.0) +
        df["is_night"].astype(int) * weights_cfg.get("is_night_transaction_weight", 1.0) +
        df["is_international"].astype(int) * weights_cfg.get("is_international_weight", 1.0)
    )

    # Cap score limited to 2 decimals
    df["fraud_score"] = df["fraud_score"].round(2)

    # Label transactions based on fraud score thresholds
    if labeling_cfg["use_dynamic_threshold"]:
        # Dynamic threshold based on target_fraud_rate percentile
        dynamic_threshold = df["fraud_score"].quantile(1 - labeling_cfg["target_fraud_rate"]) # e.g., 0.95 for top 5%
        df["fraud_label"] = (df["fraud_score"] >= dynamic_threshold).astype(int)
    else:
        # Static threshold
        df["fraud_label"] = (df["fraud_score"] >= labeling_cfg["min_score_threshold"]).astype(int)

    return df

