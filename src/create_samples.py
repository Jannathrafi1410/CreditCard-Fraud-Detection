"""
Utility script to create representative sample datasets from raw creditcard.csv.
"""
import os
import pandas as pd

def generate_samples():
    raw_path = "data/raw/creditcard.csv"
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found.")
        return

    print("Loading raw dataset to create sample splits...")
    df = pd.read_csv(raw_path)

    fraud_df = df[df["Class"] == 1]
    non_fraud_df = df[df["Class"] == 0]

    # Create 500-record representative sample dataset (100 fraud, 400 legitimate)
    sample_df = pd.concat([
        fraud_df.sample(n=min(100, len(fraud_df)), random_state=42),
        non_fraud_df.sample(n=400, random_state=42)
    ]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    os.makedirs("data/sample", exist_ok=True)
    sample_path = "data/sample/sample_transactions.csv"
    sample_df.to_csv(sample_path, index=False)
    print(f"Generated {sample_path} with shape {sample_df.shape}. Fraud count: {(sample_df['Class'] == 1).sum()}")

    # Create 50-record unlabelled batch test set for dashboard batch upload testing
    batch_test = pd.concat([
        fraud_df.sample(n=15, random_state=101),
        non_fraud_df.sample(n=35, random_state=101)
    ]).sample(frac=1.0, random_state=101).reset_index(drop=True)

    batch_test_features = batch_test.drop(columns=["Class"])
    batch_path = "data/sample/test_batch_transactions.csv"
    batch_test_features.to_csv(batch_path, index=False)
    print(f"Generated {batch_path} with shape {batch_test_features.shape}")

if __name__ == "__main__":
    generate_samples()
