import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/shashidj/Predictive-Maintenance/engine_data.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")

print("\n=== Data Cleaning ===")

# Handle missing values
numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
print("Missing values handled")

# Define target variable
target_col = "Engine Condition"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset")

X = df.drop(columns=[target_col])
y = df[target_col]

print(f"\nFeature columns: {list(X.columns)}")
print(f"Target distribution:\n{y.value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
print("\nDatasets saved locally")

files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="shashidj/Predictive-Maintenance",
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} to Hugging Face Hub")

print("\n=== Data Preparation Complete ===")
print(f"Processed {len(df)} records")
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
