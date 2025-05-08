# Telecom Churn Predictor

A complete end-to-end machine learning pipeline for predicting customer churn in a telecommunications company. This project demonstrates data ingestion, cleaning, feature engineering, model training, evaluation, and a user-friendly Streamlit demo app.

---

## 🚀 Features

- **Data Preprocessing**: Clean raw CSV, fix data types, handle missing values  
- **Feature Engineering**: Label-encode binary fields, one-hot encode multi-class features, tenure buckets, derived spend metrics  
- **Model Training**: Logistic Regression baseline and Random Forest with hyperparameter tuning via `GridSearchCV`  
- **Evaluation**: ROC AUC, confusion matrix, precision/recall/F1 report on hold-out data  
- **Deployment**: Interactive Streamlit app for live churn-risk predictions  
- **Reproducibility**: Single-command pipeline orchestration via `main.py` and versioned dependencies in `requirements.txt`

---

## 🛠 Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/kacperguzydev/telecom-churn-predictor.git
   cd telecom-churn-predictor
Create & activate a virtual environment

python3 -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Prepare data folders

mkdir -p data/raw data/processed models outputs
Download the dataset
Place WA_Fn-UseC_-Telco-Customer-Churn.csv into data/raw/.

▶️ Usage
Run each stage individually, or all stages at once:

# 1. Clean & preprocess raw data
python main.py preprocess

# 2. Generate modeling features
python main.py features

# 3. Train & tune the model
python main.py train

# 4. Evaluate performance
python main.py evaluate

# 5. Run the full pipeline end-to-end
python main.py all
Launch the Streamlit demo

streamlit run streamlit_app.py
Open your browser at http://localhost:8501 to interact with the churn predictor.

## 📂 Project Structure



```plaintext
telecom-churn-predictor/
├── data/                  # Project data
│   ├── raw/               # Original CSV files
│   └── processed/         # Cleaned & feature datasets
├── models/                # Saved ML model artifacts
├── outputs/               # Reports & plots from evaluation
├── data_processing.py     # Load & clean raw data
├── feature_engineering.py # Generate modeling features
├── modeling.py            # Train & serialize the best model
├── evaluation.py          # Compute & display metrics
├── main.py                # CLI pipeline orchestrator
├── streamlit_app.py       # Streamlit front-end app
├── utils.py               # Shared helper functions
└── requirements.txt       # Python dependencies

📈 Results
After running the full pipeline, the Random Forest model achieved:

Cross-validated AUC: ~0.842

Hold-out ROC AUC: ~0.93

Classification report (on hold-out data):

Accuracy: ~85%

Precision (churn): 0.79, Recall (churn): 0.61, F1-score (churn): 0.69
