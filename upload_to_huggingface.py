"""
Upload engine_data.csv to Hugging Face dataset repository.
Run: python upload_to_huggingface.py
You will be prompted for your HF token (get it from https://huggingface.co/settings/tokens)
"""

from huggingface_hub import HfApi, login
from datasets import load_dataset
import os

# Configuration
HF_USERNAME = "shashidj"
REPO_NAME = "Predictive-Maintenance"
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "engine_data.csv")

# Login to Hugging Face
print("Login to Hugging Face (get token from https://huggingface.co/settings/tokens)")
login()

api = HfApi()

# Create dataset repo if it doesn't exist
repo_id = f"{HF_USERNAME}/{REPO_NAME}"
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
print(f"Dataset repo ready: https://huggingface.co/datasets/{repo_id}")

# Upload the CSV file
api.upload_file(
    path_or_fileobj=DATA_FILE,
    path_in_repo="data/engine_data.csv",
    repo_id=repo_id,
    repo_type="dataset",
)
print(f"Successfully uploaded engine_data.csv to {repo_id}")
print(f"View at: https://huggingface.co/datasets/{repo_id}")
