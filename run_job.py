import modal
import subprocess
import os

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
REPO_BASE = "https://github.com/MOSHIIUR/lpr-rsr-ext.git"
REPO_DIR = "lpr-rsr-ext"
BRANCH = "main"

# 2. Define a Volume to store logs and checkpoints
results_vol = modal.Volume.from_name("lpr-rsr-ext", create_if_missing=True)

# 3. Define the Environment
image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.9"
    )
    .apt_install(
        "git", "wget", "curl", "build-essential",
        "bash",
        "readline-common",
        "libreadline8",
        "ncurses-term",
        "less",
        # OpenCV dependencies
        "libgl1-mesa-glx",
        "libglib2.0-0",
    )
    .pip_install("uv", "wheel", "setuptools")
    .run_commands(
        # Install PyTorch with CUDA 11.8
        "uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu118",
        # Install TensorFlow and other dependencies
        "uv pip install --system tensorflow tf-keras",
        "uv pip install --system opencv-python scikit-image albumentations Pillow",
        "uv pip install --system numpy pandas tqdm matplotlib python-Levenshtein",
        "uv pip install --system wandb",
    )
)

# Define secrets
wandb_secret = modal.Secret.from_name("wandb-secret")
github_secret = modal.Secret.from_name("github-secret")

app = modal.App("lpr-rsr-ext")

@app.function(
    image=image,
    gpu="T4",
    cpu=2,
    memory=8192,
    secrets=[wandb_secret, github_secret],
    timeout=86400,  # 24 Hours
    volumes={"/root/experiments": results_vol}
)
def run_experiments():
    print("🚀 Container started.")

    VOLUME_ROOT = "/root/experiments"
    REPO_NAME = "lpr-rsr-ext"
    REPO_PATH = os.path.join(VOLUME_ROOT, REPO_NAME)

    # Construct repo URL with token from secret if available
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        repo_url = REPO_BASE.replace("https://", f"https://{github_token}@")
    else:
        repo_url = REPO_BASE
        print("⚠️  Warning: No GITHUB_TOKEN found. Using public access.")

    # 1. Clone or Update the Repo INSIDE the Volume
    if os.path.exists(REPO_PATH):
        print(f"🔄 Repo found at {REPO_PATH}. Pulling latest changes...")
        try:
            subprocess.run("git pull", cwd=REPO_PATH, shell=True, check=True,
                         capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Git pull failed: {e.stderr}")
            print("Attempting to reset and pull...")
            subprocess.run("git fetch origin", cwd=REPO_PATH, shell=True, check=False)
            subprocess.run(f"git reset --hard origin/{BRANCH}", cwd=REPO_PATH, shell=True, check=False)
    else:
        print(f"📂 Cloning repo into Volume at {REPO_PATH}...")
        subprocess.run(f"git clone -q -b {BRANCH} {repo_url} {REPO_PATH}", shell=True, check=True)

    print("🏃 Running Training...")
    subprocess.run(
        ["bash", "run.sh"],
        cwd=REPO_PATH,
        check=True,
    )

    # Commit Volume (Save everything to the cloud)
    print("💾 Committing volume...")
    results_vol.commit()
    print("🎉 Done. All files saved to volume.")


@app.local_entrypoint()
def main():
    run_experiments.remote()
