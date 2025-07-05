import os
import pathlib
import requests
from tqdm import tqdm

# Directory paths inside container
MODELS_DIR = pathlib.Path("/workspace/models")
LORA_DIR = pathlib.Path("/workspace/lora_models")

# Base model URLs (SDXL Turbo + AnimateDiff motion adapter)
SDXL_TURBO_URL = os.getenv("SDXL_TURBO_URL", "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sdxl_turbo.safetensors")
MOTION_ADAPTER_URL = os.getenv("MOTION_ADAPTER_URL", "https://huggingface.co/animatediff/animatediff-motion-adapter-v1/resolve/main/diffusion_pytorch_model.safetensors")

# LoRA weight URLs – you can override via env vars for private links
TEMO_LORA_URL = os.getenv("TEMO_LORA_URL")  # required
FELFEL_LORA_URL = os.getenv("FELFEL_LORA_URL")  # required

DOWNLOADS = [
    # (url, destination path)
    (SDXL_TURBO_URL, MODELS_DIR / "sdxl-turbo" / "sdxl_turbo.safetensors"),
    (MOTION_ADAPTER_URL, MODELS_DIR / "animatediff" / "motion_adapter" / "diffusion_pytorch_model.safetensors"),
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


def main():
    missing_private = [name for name, url in [("TEMO_LORA_URL", TEMO_LORA_URL), ("FELFEL_LORA_URL", FELFEL_LORA_URL)] if url is None]
    if missing_private:
        print("⚠ WARNING: Missing env vars for private LoRA URLs –" + ", ".join(missing_private))
    for url, dest in DOWNLOADS:
        if url is None:
            print(f"⚠ Skipping download for {dest.name} (no URL)")
            continue
        fetch(url, dest)


if __name__ == "__main__":
    main()

