import pandas as pd
import logging
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import joblib

logging.basicConfig(level=logging.INFO, format='%(message)s')


def evaluate_model(
    feature_path: str = "data/processed/features.parquet",
    model_path: str = "models/best_model.pkl",
    decision_threshold: float = 0.5
) -> None:
    """
    Loads the model and feature set, then prints key metrics.
    """
    logging.info(f"Loading features from {feature_path}")
    df = pd.read_parquet(feature_path)

    X = df.drop(columns=['Churn'])
    y = df['Churn']
    model = joblib.load(model_path)

    logging.info("Computing probabilities and metrics...")
    probs = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs)
    preds = (probs >= decision_threshold).astype(int)

    cm = confusion_matrix(y, preds)
    report = classification_report(y, preds)

    logging.info("ROC AUC: %.3f", auc)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)


if __name__ == '__main__':
    evaluate_model()