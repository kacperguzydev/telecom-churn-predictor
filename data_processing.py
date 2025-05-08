import os
import pandas as pd
import logging

# Set up a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def load_and_clean_data(
    raw_csv_path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    cleaned_parquet_path: str = "data/processed/cleaned.parquet"
) -> pd.DataFrame:
    """
    Loads the raw churn CSV, cleans it (fixes TotalCharges, drops bad rows),
    and writes out a Parquet file for downstream steps.

    Returns:
        The cleaned DataFrame.
    """
    # Ensure the output folder exists
    os.makedirs(os.path.dirname(cleaned_parquet_path), exist_ok=True)

    logging.info(f"Reading raw data from {raw_csv_path}")
    try:
        df = pd.read_csv(raw_csv_path)
    except FileNotFoundError:
        logging.error(f"Cannot find raw data at {raw_csv_path}. Did you place the CSV there?")
        raise

    # Tackle messy TotalCharges entries
    before_rows = len(df)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    dropped = before_rows - len(df)
    logging.info(f"Dropped {dropped} rows due to invalid TotalCharges")

    # Drop the customer identifier; we don’t need it for modeling
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        logging.debug("Removed 'customerID' column")

    # Save cleaned data
    df.to_parquet(cleaned_parquet_path, index=False)
    logging.info(f"Cleaned data saved to {cleaned_parquet_path}")
    return df


if __name__ == '__main__':
    load_and_clean_data()