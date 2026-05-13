
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

def deploy_to_huggingface_space():
    SPACE_NAME = "Predictive-Maintenance"
    REPO_ID = f"shashidj/{SPACE_NAME}"

    print("🚀 Starting deployment to HuggingFace Spaces...")
    print(f"📦 Target Space: {REPO_ID}")

    api = HfApi(token=os.getenv("HF_TOKEN"))

    try:
        api.repo_info(repo_id=REPO_ID, repo_type="space")
        print(f"📍 Space '{REPO_ID}' already exists")
    except RepositoryNotFoundError:
        print(f"🆕 Creating new space '{REPO_ID}'...")
        create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="docker", private=False)
        print("✅ Space created successfully")

    deployment_dir = Path("predictive_maintenance_mlops/deployment")
    required_files = ["Dockerfile", "app.py", "requirements.txt"]

    print("\n🔍 Verifying deployment files...")
    for file_name in required_files:
        if (deployment_dir / file_name).exists():
            print(f"✅ {file_name} found")
        else:
            print(f"❌ {file_name} missing")
            return False

    readme_content = """---
title: Predictive Maintenance
emoji: 🔧
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
app_file: app.py
pinned: false
license: mit
---

# Predictive Maintenance — Engine Health Monitor 🔧

An intelligent ML-powered system that monitors engine sensor data and predicts failure risk in real time.

## Features
- **Real-time Predictions**: Instant engine health scoring
- **Interactive Interface**: User-friendly Streamlit web application
- **Multiple Algorithms**: Best model selected from 6 different algorithms
- **MLOps Integration**: Complete pipeline with experiment tracking

## Technical Stack
- **Backend**: Python, Scikit-learn, XGBoost
- **Frontend**: Streamlit
- **ML Tracking**: MLflow
- **Deployment**: Docker, HuggingFace Spaces
"""
    (deployment_dir / "README.md").write_text(readme_content)
    print("✅ README.md created")

    print("\n📤 Uploading files to HuggingFace Space...")
    api.upload_folder(
        folder_path=str(deployment_dir),
        repo_id=REPO_ID,
        repo_type="space",
        commit_message="Deploy Predictive Maintenance application"
    )
    print(f"\n🎉 Deployment completed!")
    print(f"🌐 https://huggingface.co/spaces/{REPO_ID}")
    return True

if __name__ == "__main__":
    if not os.getenv("HF_TOKEN"):
        print("❌ HF_TOKEN environment variable not set!")
    else:
        deploy_to_huggingface_space()
