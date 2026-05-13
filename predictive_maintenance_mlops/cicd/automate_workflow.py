
import os
import sys
import subprocess
from datetime import datetime

# Use python3 explicitly for macOS compatibility
PYTHON = sys.executable

def run_command(command, cwd=None):
    process = subprocess.Popen(
        command, shell=True, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True
    )
    for line in process.stdout:
        print(f"   {line.rstrip()}")
    process.wait()
    return process.returncode == 0

def check_prerequisites():
    print("🔍 Checking prerequisites...")
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN environment variable not set")
        return False
    print("✅ HF_TOKEN found")
    required_files = [
        ".github/workflows/pipeline.yml",
        "predictive_maintenance_mlops/model_building/data_register.py",
        "predictive_maintenance_mlops/model_building/prep.py",
        "predictive_maintenance_mlops/model_building/model_training.py",
        "predictive_maintenance_mlops/deployment/Dockerfile",
        "predictive_maintenance_mlops/deployment/app.py",
        "predictive_maintenance_mlops/deployment/requirements.txt",
        "predictive_maintenance_mlops/deployment/deploy_to_hf_space.py"
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"❌ Missing files: {missing}")
        return False
    print("✅ All required files present")
    return True

def setup_environment():
    print("🔧 Setting up environment...")
    for d in ["predictive_maintenance_mlops/data", "reports", ".github/workflows"]:
        os.makedirs(d, exist_ok=True)
    return True

def execute_local_pipeline():
    print("\n📊 Running Data Registration...")
    if not run_command(f'"{PYTHON}" predictive_maintenance_mlops/model_building/data_register.py'):
        return False
    print("\n🔄 Running Data Preparation...")
    if not run_command(f'"{PYTHON}" predictive_maintenance_mlops/model_building/prep.py'):
        return False
    print("\n💡 Model training delegated to GitHub Actions")
    return True

def initialize_repository():
    return run_command(f'"{PYTHON}" predictive_maintenance_mlops/cicd/setup_repository.py')

def push_to_github():
    return run_command(f'"{PYTHON}" predictive_maintenance_mlops/cicd/push_to_github.py')

def main():
    print("=" * 70)
    print("🚀 PREDICTIVE MAINTENANCE - COMPLETE MLOPS AUTOMATION")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if not check_prerequisites():
        return False
    if not setup_environment():
        return False
    if not execute_local_pipeline():
        return False
    if not initialize_repository():
        return False
    if not push_to_github():
        return False

    print("\n" + "=" * 70)
    print("🎉 MLOPS AUTOMATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("Monitor: https://github.com/sdj206gl/Predictive-Maintenance/actions")
    print("App:     https://huggingface.co/spaces/shashidj/Predictive-Maintenance")
    return True

if __name__ == "__main__":
    main()
