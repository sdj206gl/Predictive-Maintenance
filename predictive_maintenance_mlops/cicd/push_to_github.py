
import os
import subprocess
from datetime import datetime

def run_command(command, cwd=None):
    result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
    return result.stdout.strip(), result.stderr.strip(), result.returncode

def verify_repository():
    print("🔧 Verifying repository configuration...")
    stdout, stderr, code = run_command("git rev-parse --is-inside-work-tree")
    if code != 0:
        print("❌ Not inside a Git repository")
        return False
    stdout, stderr, code = run_command("git remote get-url origin")
    if code != 0:
        print("⚠️ No remote origin configured")
        return False
    print(f"✅ Remote origin: {stdout}")
    return True

def stage_and_commit(message=None):
    if not message:
        message = f"MLOps Pipeline Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    run_command("git add .")
    stdout, stderr, code = run_command(f'git commit -m "{message}"')
    if code != 0 and "nothing to commit" not in stderr:
        print(f"❌ Commit failed: {stderr}")
        return False
    print("✅ Commit created")
    return True

def push_to_remote(branch="main"):
    print(f"🚀 Pushing to {branch}...")
    stdout, stderr, code = run_command(f"git push origin {branch}")
    if code != 0:
        run_command(f"git pull --rebase origin {branch}")
        stdout, stderr, code = run_command(f"git push origin {branch}")
    if code != 0:
        print(f"❌ Push failed: {stderr}")
        return False
    print("✅ Pushed to remote")
    return True

def main():
    print("=" * 60)
    print("🚀 PUSH TO GITHUB")
    print("=" * 60)
    if not verify_repository():
        print("Run setup_repository.py first.")
        return False
    stage_and_commit()
    push_to_remote("main")
    print("\n🎉 Successfully pushed to GitHub!")
    print("GitHub Actions pipeline will be triggered automatically.")

if __name__ == "__main__":
    main()
