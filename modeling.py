import os
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

logging.basicConfig(level=logging.INFO, format='%(message)s')


def train_model(
    features_path: str = "data/processed/features.parquet",
    output_model_path: str = "models/best_model.pkl"
) -> None:
    """
    Runs a grid search on a RandomForest and saves the top model.
    """
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

    logging.info(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logging.info("Split data: train size = %d, test size = %d", len(X_train), len(X_test))

    param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    joblib.dump(best, output_model_path)
    logging.info(
        "Best model (AUC=%.3f) saved to %s",
        grid.best_score_, output_model_path
    )


if __name__ == '__main__':
    train_model()