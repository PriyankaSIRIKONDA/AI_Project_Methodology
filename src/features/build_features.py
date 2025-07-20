import pandas as pd
import numpy as np
import os

def add_tenure_group(df):
    bins = [0, 12, 24, 48, 60, np.inf]
    labels = ['0-12', '13-24', '25-48', '49-60', '61+']
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels)
    df = pd.get_dummies(df, columns=['tenure_group'], drop_first=True)
    return df

def add_services_count(df):
    # Only add if columns exist
    services = [col for col in df.columns if any(s in col for s in ['PhoneService_Yes', 'MultipleLines_Yes', 'InternetService_Fiber optic',
                'InternetService_DSL', 'OnlineSecurity_Yes', 'OnlineBackup_Yes',
                'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes'])]
    df['num_services'] = df[services].sum(axis=1)
    return df

if __name__ == "__main__":
    input_path = os.path.join("data", "processed", "cleaned_telco.csv")
    output_path = os.path.join("data", "processed", "featured_telco.csv")
    df = pd.read_csv(input_path)
    df = add_tenure_group(df)
    df = add_services_count(df)
    df.to_csv(output_path, index=False)
