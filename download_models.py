import os
import pathlib
import requests
from tqdm import tqdm
import gdown
import subprocess

# Directory paths inside container
MODELS_DIR = pathlib.Path("/workspace/models")
LORA_DIR = pathlib.Path("/workspace/lora_models")

# Base model URLs (SDXL Turbo + AnimateDiff motion adapter + ControlNet)
SDXL_TURBO_URL = os.getenv("SDXL_TURBO_URL", "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sdxl_turbo.safetensors")
MOTION_ADAPTER_URL = os.getenv("MOTION_ADAPTER_URL", "https://huggingface.co/guoyww/animatediff-motion-adapter-sdxl-beta/resolve/main/diffusion_pytorch_model.safetensors")
CONTROLNET_URL = os.getenv("CONTROLNET_URL", "https://huggingface.co/diffusers/controlnet-openpose-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors")

# LoRA weight URLs – you can override via env vars for private links
TEMO_LORA_URL = os.getenv("TEMO_LORA_URL")  # required
FELFEL_LORA_URL = os.getenv("FELFEL_LORA_URL")  # required

# Google Drive URL for LoRA models
GOOGLE_DRIVE_LORA_URL = "https://drive.google.com/drive/folders/1k-LT9g4GjguFuxxTMjpMdf1CGYFoRpEJ?usp=sharing"

CONTROLNET_POSE_REPO = "lllyasviel/ControlNet-v1-1"  # Example repo for pose ControlNet
CONTROLNET_POSE_MODEL = "controlnet_pose.pth"  # Example filename
CONTROLNET_POSE_DIR = os.path.join("models", "controlnet")

DOWNLOADS = [
    # (url, destination path)
    (SDXL_TURBO_URL, MODELS_DIR / "sdxl-turbo" / "sdxl_turbo.safetensors"),
    (MOTION_ADAPTER_URL, MODELS_DIR / "animatediff" / "motion_adapter" / "diffusion_pytorch_model.safetensors"),
    (CONTROLNET_URL, MODELS_DIR / "controlnet-openpose-sdxl" / "diffusion_pytorch_model.safetensors"),
]

if TEMO_LORA_URL:
    DOWNLOADS.append((TEMO_LORA_URL, LORA_DIR / "temo_lora" / "deep_sdxl_turbo_lora_weights.pt"))
if FELFEL_LORA_URL:
    DOWNLOADS.append((FELFEL_LORA_URL, LORA_DIR / "felfel_lora" / "deep_sdxl_turbo_lora_weights.pt"))


def fetch(url: str, dest: pathlib.Path):
    """Stream-download a file with progress bar"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"✔ {dest} already exists, skipping download")
        return

    print(f"⬇ Downloading {url} → {dest}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    print(f"✅ Saved {dest}")


def download_from_google_drive(url, output_path):
    gdown.download_folder(url, output=output_path, quiet=False, use_cookies=False)


def download_controlnet_pose():
    os.makedirs(CONTROLNET_POSE_DIR, exist_ok=True)
    model_path = os.path.join(CONTROLNET_POSE_DIR, CONTROLNET_POSE_MODEL)
    if not os.path.exists(model_path):
        print(f"Downloading ControlNet pose model to {model_path}...")
        # Example using huggingface-cli; replace with actual download logic as needed
        subprocess.run([
            "huggingface-cli", "download", CONTROLNET_POSE_REPO, "--filename", CONTROLNET_POSE_MODEL, "--local-dir", CONTROLNET_POSE_DIR
        ], check=True)
    else:
        print(f"ControlNet pose model already exists at {model_path}")


def main():
    missing_private = [name for name, url in [("TEMO_LORA_URL", TEMO_LORA_URL), ("FELFEL_LORA_URL", FELFEL_LORA_URL)] if url is None]
    if missing_private:
        print("⚠ WARNING: Missing env vars for private LoRA URLs –" + ", ".join(missing_private))
    for url, dest in DOWNLOADS:
        if url is None:
            print(f"⚠ Skipping download for {dest.name} (no URL)")
            continue
        fetch(url, dest)

    # Always try to download LoRA models from Google Drive if folders don't exist
    if not os.path.exists('/workspace/lora_models/felfel_lora'):
        print("⬇ Downloading Felfel LoRA from Google Drive...")
        download_from_google_drive(GOOGLE_DRIVE_LORA_URL + '/felfel_lora', '/workspace/lora_models/felfel_lora')

    if not os.path.exists('/workspace/lora_models/temo_lora'):
        print("⬇ Downloading Temo LoRA from Google Drive...")
        download_from_google_drive(GOOGLE_DRIVE_LORA_URL + '/temo_lora', '/workspace/lora_models/temo_lora')

    download_controlnet_pose()


if __name__ == "__main__":
    main()

