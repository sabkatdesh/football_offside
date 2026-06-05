"""
download_models.py
──────────────────
Downloads both YOLO model weights from HuggingFace Hub into data/models/.
Run once before starting the server:

    python download_models.py
"""

import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        "repo_id": "Sabkat/football-pitch-detection",
        "filename": "football-pitch-detection.pt",
        "local_name": "football-pitch-detection.pt",
    },
    {
        "repo_id": "Sabkat/football-player-detection",
        "filename": "football-player-detection.pt",
        "local_name": "football-player-detection.pt",
    },
]


def download():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("❌  huggingface_hub not installed. Run:  pip install huggingface_hub")
        sys.exit(1)

    for m in MODELS:
        dest = MODELS_DIR / m["local_name"]
        if dest.exists():
            print(f"✅  {m['local_name']} already exists — skipping.")
            continue
        print(f"⬇️   Downloading {m['repo_id']} / {m['filename']} …")
        try:
            path = hf_hub_download(
                repo_id=m["repo_id"],
                filename=m["filename"],
                local_dir=str(MODELS_DIR),
            )
            # hf_hub_download may place the file in a subfolder — move if needed
            downloaded = Path(path)
            if downloaded != dest:
                downloaded.rename(dest)
            print(f"✅  Saved to {dest}")
        except Exception as e:
            print(f"❌  Failed to download {m['repo_id']}: {e}")
            sys.exit(1)

    print("\n✅  All models downloaded successfully.")
    print(f"    Location: {MODELS_DIR}\n")


if __name__ == "__main__":
    download()
