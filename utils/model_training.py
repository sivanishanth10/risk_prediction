# utils/model_training.py
import os
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

MODEL_PATHS = {
    "log_reg": "models/logistic_regression.pkl",
    "catboost": "models/catboost_model.cbm",
    "scaler": "models/scaler.pkl"
}

def train_and_save_models():
    """Train and save all models"""
    print("🔄 Loading dataset...")
    df = pd.read_csv("data/cs-training.csv")
    df.dropna(inplace=True)

    # Drop the index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
        print("✅ Dropped index column 'Unnamed: 0'")

    X = df.drop("SeriousDlqin2yrs", axis=1)
    y = df["SeriousDlqin2yrs"]

    # Handle missing values
    X = X.fillna(0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("✅ Training Logistic Regression...")
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    pickle.dump(log_reg, open(MODEL_PATHS["log_reg"], "wb"))
    print(f"💾 Saved Logistic Regression model → {MODEL_PATHS['log_reg']}")

    print("✅ Training CatBoost Classifier...")
    cat = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        loss_function='Logloss',
        verbose=False,
        random_seed=42
    )
    cat.fit(X_train, y_train)
    cat.save_model(MODEL_PATHS["catboost"])
    print(f"💾 Saved CatBoost model → {MODEL_PATHS['catboost']}")

    # Save scaler
    pickle.dump(scaler, open(MODEL_PATHS["scaler"], "wb"))
    print(f"💾 Saved Scaler → {MODEL_PATHS['scaler']}")

    print("🎉 Model training complete!")

def ensure_models_exist():
    """Check if models are saved, if not, train and save them."""
    if not os.path.exists("models"):
        os.makedirs("models")
        print("📁 Created models directory")

    if not all(os.path.exists(path) for path in MODEL_PATHS.values()):
        print("🔄 Training models because they don't exist yet...")
        train_and_save_models()
    else:
        print("✅ Models already exist.")

if __name__ == "__main__":
    ensure_models_exist()
