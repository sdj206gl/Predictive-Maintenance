
import os
import subprocess

def run_command(command, check=False):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout.strip(), result.stderr.strip(), result.returncode

def initialize_git_repository():
    print("🔧 Initializing Git repository...")
    if not os.path.exists('.git'):
        run_command("git init")
        print("✅ Git repository initialized")
    else:
        print("ℹ️ Git repository already exists")
    run_command("git config --global user.name 'MLOps Pipeline' || true", check=False)
    run_command("git config --global user.email 'mlops@example.com' || true", check=False)
    return True

def create_gitignore():
    gitignore_content = """
__pycache__/
*.py[cod]
.Python
venv/
env/
pip-log.txt
*.csv
*.pkl
*.h5
predictive_maintenance_mlops/data/
mlruns/
.mlflow/
.ipynb_checkpoints/
*.ipynb
.vscode/
.env
.DS_Store
temp/
tmp/
"""
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content.strip())
    print("✅ .gitignore file created")

def create_directory_structure():
    print("📁 Creating project directory structure...")
    directories = [
        "predictive_maintenance_mlops/data",
        "predictive_maintenance_mlops/model_building",
        "predictive_maintenance_mlops/deployment",
        "predictive_maintenance_mlops/cicd",
        ".github/workflows",
        "reports", "docs", "tests"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Directory structure created")

def create_readme():
    readme_content = """# Predictive Maintenance MLOps Pipeline 🔧

An end-to-end MLOps pipeline for predicting engine failure from sensor data.

## Project Structure
```
Predictive-Maintenance/
├── .github/workflows/pipeline.yml
├── predictive_maintenance_mlops/
│   ├── data/
│   ├── model_building/
│   ├── deployment/
│   └── cicd/
└── reports/
```

## Quick Start
```bash
export HF_TOKEN="your_huggingface_token"
python predictive_maintenance_mlops/cicd/automate_workflow.py
```

## Links
- **App**: https://huggingface.co/spaces/shashidj/Predictive-Maintenance
- **Model**: https://huggingface.co/shashidj/Predictive-Maintenance-Model
- **Dataset**: https://huggingface.co/datasets/shashidj/Predictive-Maintenance
"""
    with open('README.md', 'w') as f:
        f.write(readme_content)
    print("✅ README.md created")

def main():
    print("=" * 60)
    print("🚀 PREDICTIVE MAINTENANCE - REPOSITORY SETUP")
    print("=" * 60)
    initialize_git_repository()
    create_directory_structure()
    create_gitignore()
    create_readme()
    print("\n✅ Repository setup completed!")
    print("📋 Next Steps:")
    print("   1. Set HF_TOKEN environment variable")
    print("   2. Run: git add . && git commit -m 'Initial commit'")
    print("   3. Run: git push -u origin main")
    print("=" * 60)

if __name__ == "__main__":
    main()
