import pandas as pd
import os

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    # Remove customerID
    df = df.drop('customerID', axis=1)
    # Convert TotalCharges to numeric, coerce errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Fill missing TotalCharges with median
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    # Encode categorical variables
    cat_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

if __name__ == "__main__":
    input_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    output_path = os.path.join("data", "processed", "cleaned_telco.csv")
    df = load_data(input_path)
    df_clean = preprocess(df)
    df_clean.to_csv(output_path, index=False) 