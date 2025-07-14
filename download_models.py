import os
import pathlib
import requests
from tqdm import tqdm
import gdown
import subprocess

# Directory paths inside container
MODELS_DIR = pathlib.Path("/workspace/models")
LORA_DIR = pathlib.Path("/workspace/lora_models")

# HIGH QUALITY BASE MODEL URLs
SDXL_BASE_URL = os.getenv("SDXL_BASE_URL", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors")
SDXL_VAE_URL = os.getenv("SDXL_VAE_URL", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors")
MOTION_ADAPTER_URL = os.getenv("MOTION_ADAPTER_URL", "https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2/resolve/main/diffusion_pytorch_model.safetensors")
CONTROLNET_URL = os.getenv("CONTROLNET_URL", "https://huggingface.co/diffusers/controlnet-openpose-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors")

# LoRA weight URLs – you can override via env vars for private links
TEMO_LORA_URL = os.getenv("TEMO_LORA_URL")  # required
FELFEL_LORA_URL = os.getenv("FELFEL_LORA_URL")  # required

# Google Drive URL for LoRA models
GOOGLE_DRIVE_LORA_URL = "https://drive.google.com/drive/folders/1k-LT9g4GjguFuxxTMjpMdf1CGYFoRpEJ?usp=sharing"

DOWNLOADS = [
    # (url, destination path)
    (SDXL_BASE_URL, MODELS_DIR / "stable-diffusion-xl-base-1.0" / "sd_xl_base_1.0.safetensors"),
    (SDXL_VAE_URL, MODELS_DIR / "stable-diffusion-xl-base-1.0" / "sd_xl_base_1.0_0.9vae.safetensors"),
    (MOTION_ADAPTER_URL, MODELS_DIR / "animatediff" / "motion_adapter_v1_5_2" / "diffusion_pytorch_model.safetensors"),
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

    print(f"⬇ Downloading HIGH QUALITY model: {url} → {dest}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    print(f"✅ Downloaded HIGH QUALITY model: {dest}")


def download_from_google_drive(url, output_path):
    """Download LoRA models from Google Drive"""
    print(f"⬇ Downloading from Google Drive: {url}")
    gdown.download_folder(url, output=output_path, quiet=False, use_cookies=False)


def download_huggingface_model(repo_id, local_dir):
    """Download complete model from HuggingFace using git clone"""
    print(f"⬇ Downloading complete model: {repo_id}")
    try:
        subprocess.run([
            "git", "clone", f"https://huggingface.co/{repo_id}", str(local_dir)
        ], check=True)
        print(f"✅ Model downloaded: {local_dir}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download {repo_id}: {e}")


def main():
    print("🚀 Downloading HIGH QUALITY models for maximum animation quality...")
    
    missing_private = [name for name, url in [("TEMO_LORA_URL", TEMO_LORA_URL), ("FELFEL_LORA_URL", FELFEL_LORA_URL)] if url is None]
    if missing_private:
        print("⚠ WARNING: Missing env vars for private LoRA URLs –" + ", ".join(missing_private))
    
    # Download individual model files
    for url, dest in DOWNLOADS:
        if url is None:
            print(f"⚠ Skipping download for {dest.name} (no URL)")
            continue
        fetch(url, dest)
    
    # Download complete models from HuggingFace (better for quality)
    print("\n🔄 Downloading complete HIGH QUALITY models from HuggingFace...")
    
    # Full SDXL Base model
    sdxl_dir = MODELS_DIR / "stable-diffusion-xl-base-1.0"
    if not sdxl_dir.exists():
        download_huggingface_model("stabilityai/stable-diffusion-xl-base-1.0", sdxl_dir)
    
    # Stable motion adapter
    motion_dir = MODELS_DIR / "animatediff-motion-adapter-v1-5-2"
    if not motion_dir.exists():
        download_huggingface_model("guoyww/animatediff-motion-adapter-v1-5-2", motion_dir)
    
    # ControlNet for pose guidance
    controlnet_dir = MODELS_DIR / "controlnet-openpose-sdxl-1.0"
    if not controlnet_dir.exists():
        download_huggingface_model("diffusers/controlnet-openpose-sdxl-1.0", controlnet_dir)

    # Always try to download LoRA models from Google Drive if folders don't exist
    if not os.path.exists('/workspace/lora_models/felfel_lora'):
        print("⬇ Downloading Felfel LoRA from Google Drive...")
        download_from_google_drive(GOOGLE_DRIVE_LORA_URL + '/felfel_lora', '/workspace/lora_models/felfel_lora')

    if not os.path.exists('/workspace/lora_models/temo_lora'):
        print("⬇ Downloading Temo LoRA from Google Drive...")
        download_from_google_drive(GOOGLE_DRIVE_LORA_URL + '/temo_lora', '/workspace/lora_models/temo_lora')

    print("\n🎉 All HIGH QUALITY models downloaded successfully!")
    print("💎 Your system is now configured for MAXIMUM QUALITY animation generation!")


if __name__ == "__main__":
    main()

