#Main

from fraud_detection.utils.config_loader import load_config
from fraud_detection.data.ingestion import build_base_dataset
from fraud_detection.data.preprocessing import preprocess_base_dataset
from fraud_detection.features.time_features import add_time_features
from fraud_detection.features.amount_features import add_amount_features
from fraud_detection.features.risk_features import add_risk_features
from fraud_detection.labeling.fraud_rules_engine import apply_fraud_rules
import pandas as pd
import os

def build_dataset(paths_cfg: str, features_cfg: str, risks_cfg: str, fraud_rules: str) -> pd.DataFrame:
    """Build and preprocess the dataset based on the provided configuration."""
    
    # Load configuration
    paths_config = load_config(paths_cfg)
    paths_cfg = paths_config["paths"]
    # features_cfg = load_config(features_cfg)
    risks_cfg = load_config(risks_cfg)
    fraud_rules_cfg = load_config(fraud_rules)
    
    # Build base dataset
    base_df = build_base_dataset(paths_cfg)
    
    # Preprocess dataset
    preprocessed_df = preprocess_base_dataset(base_df)

    # Add features
    df = add_time_features(preprocessed_df)
    df = add_amount_features(df)
    df = add_risk_features(df, risks_cfg)

    # Apply fraud rules and labeling
    df = apply_fraud_rules(df, fraud_rules_cfg)

    # Save the final dataset as CSV
    output_path = paths_cfg["output_data_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    return df





if __name__ == "__main__":
    paths_config = r"fraud_detection\configs\paths.yaml"
    features_config = r"fraud_detection\configs\features.yaml"
    risks_config = r"fraud_detection\configs\risks.yaml"
    fraud_rules_cfg = r"fraud_detection\configs\fraud_rules.yaml"
    dataset = build_dataset(paths_config, features_config, risks_config, fraud_rules_cfg)
    print("Dataset built successfully with shape:", dataset.shape)
    print(dataset.head())