import os
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(message)s')


def build_features(
    clean_data_path: str = "data/processed/cleaned.parquet",
    features_path: str = "data/processed/features.parquet"
) -> pd.DataFrame:
    """
    Reads the cleaned dataset, encodes categories, creates new features,
    and dumps out a ready-to-model feature matrix.
    """
    os.makedirs(os.path.dirname(features_path), exist_ok=True)

    logging.info(f"Loading cleaned data from {clean_data_path}")
    df = pd.read_parquet(clean_data_path)

    # Quick sanity check
    assert 'Churn' in df.columns, "Churn column missing!"

    # Encode binary flags (Yes/No) and gender
    binaries = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    le = LabelEncoder()
    for col in binaries:
        df[col] = le.fit_transform(df[col])
    logging.info("Encoded binary columns")

    # One‑hot for multi‑class features
    multis = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]
    df = pd.get_dummies(df, columns=multis, drop_first=True)
    logging.info("One‑hot encoded multi‑class features")

    # Tenure buckets: quick grouping
    bins = [0, 12, 24, 48, 72]
    names = ['0-12m', '13-24m', '25-48m', '49-72m']
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=names)
    df = pd.get_dummies(df, columns=['tenure_group'], drop_first=True)
    logging.info("Created tenure buckets")

    # Feature: average spend per month (handles tenure=0)
    df['avg_monthly_charge'] = df['TotalCharges'] / df['tenure'].replace(0, 1)
    logging.info("Added avg_monthly_charge feature")

    # Write out features
    df.to_parquet(features_path, index=False)
    logging.info(f"Feature matrix saved to {features_path}")
    return df


if __name__ == '__main__':
    build_features()