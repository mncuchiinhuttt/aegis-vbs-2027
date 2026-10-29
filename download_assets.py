#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import os
import sys
import subprocess
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT_DIR / "weights"
DATASETS_DIR = ROOT_DIR / "datasets"
DEPS_DIR = ROOT_DIR / ".download_assets_deps"

if DEPS_DIR.exists() and str(DEPS_DIR) not in sys.path:
    sys.path.insert(0, str(DEPS_DIR))

def ensure_package(package_name: str) -> None:
    if importlib.util.find_spec(package_name) is not None:
        return

    print(f"Installing {package_name}...")
    DEPS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--target",
            str(DEPS_DIR),
            package_name,
        ]
    )

    if str(DEPS_DIR) not in sys.path:
        sys.path.insert(0, str(DEPS_DIR))

    importlib.invalidate_caches()

    if importlib.util.find_spec(package_name) is None:
        raise ModuleNotFoundError(
            f"Unable to import {package_name} after installing it into {DEPS_DIR}"
        )

def get_hf_executable() -> str:
    exe = shutil.which("hf")
    if exe:
        return exe
    local_path = os.path.expanduser("~/.local/bin/hf")
    if os.path.exists(local_path):
        return local_path
    return "hf"

def download_model(repo_id: str, local_name: str, token: str = None, revision: str = None):
    dest_path = WEIGHTS_DIR / local_name
    print(f"\n--- Downloading model {repo_id} to {dest_path} using hf CLI ---")
    dest_path.mkdir(parents=True, exist_ok=True)

    hf_exe = get_hf_executable()
    cmd = [
        hf_exe, "download", repo_id,
        "--local-dir", str(dest_path),
        "--type", "model",
    ]
    if revision:
        cmd.extend(["--revision", revision])

    env = os.environ.copy()
    if token:
        env["HF_TOKEN"] = token

    subprocess.check_call(cmd, env=env)
    print(f"Download complete: {local_name}")

def download_and_unzip_m2d_clap():
    url = "https://github.com/nttcslab/m2d/releases/download/v0.5.0/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025.zip"
    dest_dir = WEIGHTS_DIR / "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025"
    zip_path = WEIGHTS_DIR / "m2d_clap_vit_base.zip"
    
    if (dest_dir / "checkpoint-30.pth").exists():
        print(f"\n--- M2D-CLAP checkpoint already exists in {dest_dir} ---")
        return
        
    print(f"\n--- Downloading M2D-CLAP model from {url} ---")
    ensure_package("requests")
    import requests
    import zipfile
    import shutil as sh
    
    # Download zip file
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                
    print(f"Extracting M2D-CLAP model...")
    # Extract to a temp directory first
    temp_extract_dir = WEIGHTS_DIR / "_temp_m2d_clap"
    temp_extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)
        
    # Find checkpoint-30.pth inside temp_extract_dir
    found_ckpt = None
    for p in temp_extract_dir.glob("**/checkpoint-30.pth"):
        found_ckpt = p
        break
        
    if found_ckpt:
        # Move it to weights/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth
        dest_dir.mkdir(parents=True, exist_ok=True)
        sh.move(str(found_ckpt), str(dest_dir / "checkpoint-30.pth"))
        print(f"Successfully placed checkpoint-30.pth in {dest_dir}")
    else:
        print("Warning: checkpoint-30.pth not found in zip archive.")
        
    # Clean up temp dir and zip file
    if temp_extract_dir.exists():
        sh.rmtree(temp_extract_dir)
    if zip_path.exists():
        zip_path.unlink()
        
    print("M2D-CLAP download and extraction complete.")

def download_m2d_runtime():
    """Acquire the unchanged NTT runtime from a pinned upstream commit."""
    dest = ROOT_DIR / "models" / "portable_m2d.py"
    expected_sha256 = "b64d94a18b558c0643b1bba578f215629e3dd00656284a8cbf992fbc2f563ee2"
    if dest.exists():
        content = dest.read_bytes()
    else:
        revision = "3d0c4de9447c404a8d3f9f37e04f53bc902e09b3"
        url = f"https://raw.githubusercontent.com/nttcslab/m2d/{revision}/examples/portable_m2d.py"
        print(f"\n--- Downloading NTT's evaluation-only M2D runtime from {url} ---")
        ensure_package("requests")
        import requests

        response = requests.get(url, timeout=60)
        response.raise_for_status()
        content = response.content
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise RuntimeError("M2D runtime differs from the pinned upstream source; it was not installed.")
    dest.write_bytes(content)
    print(f"Verified M2D portable runtime at {dest}")

def download_real_esrgan(local_name: str):
    dest_path = WEIGHTS_DIR / local_name
    if dest_path.exists():
        print(f"\n--- Real-ESRGAN weights already exist at {dest_path} ---")
        return

    url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{local_name}"
    print(f"\n--- Downloading Real-ESRGAN weights from {url} ---")
    ensure_package("requests")
    import requests

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Real-ESRGAN weights downloaded to {dest_path}")

def ensure_ffmpeg():
    """ffmpeg is required for audio extraction; this script never downloads binaries."""
    if shutil.which("ffmpeg") is not None or (ROOT_DIR / "bin" / "ffmpeg").exists():
        return
    print("\nWarning: ffmpeg was not found. Install it with your package manager "
          "(for example `brew install ffmpeg` or `apt install ffmpeg`) before running preprocessing.")


def load_env_values():
    env_vars = {}
    
    # Load from root .env if exists
    root_env = ROOT_DIR / ".env"
    if root_env.exists():
        with open(root_env, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    env_vars[k.strip()] = v.strip()
                    
    # Load from preprocessing/.env if exists (overriding defaults)
    pre_env = ROOT_DIR / "preprocessing" / ".env"
    if pre_env.exists():
        with open(pre_env, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    env_vars[k.strip()] = v.strip()
                    
    return env_vars

def main():
    parser = argparse.ArgumentParser(description="Download AEGIS model assets for full preprocessing.")
    parser.add_argument(
        "--accept-m2d-license", action="store_true",
        help="Confirm you have read and accepted NTT's evaluation-only M2D terms in NOTICE.md.",
    )
    args = parser.parse_args()
    if not args.accept_m2d_license:
        parser.error("Full preprocessing requires NTT M2D. Read NOTICE.md, then pass --accept-m2d-license.")
    print("=== AEGIS assets downloader ===")
    
    # Create directories
    WEIGHTS_DIR.mkdir(exist_ok=True)
    DATASETS_DIR.mkdir(exist_ok=True)
    
    # Load model configuration from .env files
    env_vars = load_env_values()
    qwen_vlm_id = env_vars.get("QWEN_VLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
    wemm_model_id = "tencent/WeMM-Embedding-4B"
    wemm_revision = (
        os.getenv("WEMM_MODEL_REVISION")
        or env_vars.get("WEMM_MODEL_REVISION")
        or "41a4fafdd26252ddff05c025413af7e61e26f08e"
    )
    asr_model_id = env_vars.get("ASR_MODEL_ID", "deepdml/faster-whisper-large-v3-turbo-ct2")
    yoloe_id = env_vars.get("YOLOE_MODEL_ID", "yoloe-26x-seg.pt")
    sam3_id = env_vars.get("SAM3_MODEL_ID", "facebook/sam3")
    fallback_vlm_id = env_vars.get("FALLBACK_VLM_MODEL_ID", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    real_esrgan_id = env_vars.get("REAL_ESRGAN_MODEL_ID", "RealESRGAN_x4plus.pth")
    vlm_option = env_vars.get("VLM_OPTION", "openai")
    hf_token = env_vars.get("HF_TOKEN")

    # Download weights
    # 1. Whisper large-v3-turbo (CTranslate2 format, for faster-whisper)
    download_model(asr_model_id, asr_model_id.split("/")[-1], token=hf_token)

    # 2. M2D-CLAP Environmental model (unzipped from URL)
    download_and_unzip_m2d_clap()
    download_m2d_runtime()

    # 3. YOLOE-26 zero-shot detector - fetched via the ultralytics asset
    # registry (GitHub releases), not the Hugging Face Hub like the other
    # models here, so it's downloaded directly to its expected weights/ path
    ensure_package("ultralytics")
    from ultralytics import YOLOE
    yoloe_path = WEIGHTS_DIR / yoloe_id
    if not yoloe_path.exists():
        print(f"\n--- Downloading YOLOE detector {yoloe_id} to {yoloe_path} ---")
        YOLOE(str(yoloe_path))
    else:
        print(f"\n--- YOLOE detector already exists at {yoloe_path} ---")

    # 4. Qwen VLM (if VLM_OPTION=local)
    if vlm_option == "local":
        download_model(qwen_vlm_id, qwen_vlm_id.split("/")[-1], token=hf_token)
    else:
        print(f"\nSkipping local Qwen VLM download (VLM_OPTION={vlm_option}).")
        print("To download the local VLM, set VLM_OPTION=local in preprocessing/.env")
        
    # 5. WeMM visual/text embedding. `hf download` includes the complete
    # pinned repository snapshot, including the model's custom implementation.
    download_model(
        wemm_model_id,
        "WeMM-Embedding-4B",
        token=hf_token,
        revision=wemm_revision,
    )

    # 6. SAM3 region-proposal pre-filter (gated - accept the license at
    # https://huggingface.co/facebook/sam3 and set HF_TOKEN before this runs)
    if not hf_token:
        print("\nWarning: HF_TOKEN not set - SAM3 (facebook/sam3) is a gated repo and this "
              "download will likely fail until you accept its license and set HF_TOKEN.")
    download_model(sam3_id, "sam3", token=hf_token)

    # 7. Fallback VLM for low-confidence OCR crops (SmolVLM2 by default)
    download_model(fallback_vlm_id, fallback_vlm_id.split("/")[-1], token=hf_token)

    # 8. Real-ESRGAN x4 weights - conditional Super-Resolution for small OCR crops
    download_real_esrgan(real_esrgan_id)

    # Ensure ffmpeg is installed
    ensure_ffmpeg()

    print("\n=== All requested models checked/downloaded! ===")
    print(f"Global weights stored in: {WEIGHTS_DIR}")

if __name__ == "__main__":
    main()
