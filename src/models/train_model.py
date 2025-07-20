import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import mlflow
import mlflow.sklearn
import os

def train(n_estimators=100, max_depth=8):
    df = pd.read_csv("data/processed/featured_telco.csv")
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        roc_auc = roc_auc_score(y_test, y_pred)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(model, "model")
        print(classification_report(y_test, y_pred))
        print("ROC AUC:", roc_auc)

    # Save model locally
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/rf_telco_churn.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=8)
    args = parser.parse_args()
    train(n_estimators=args.n_estimators, max_depth=args.max_depth)
