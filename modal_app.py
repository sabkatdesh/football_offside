"""
modal_app.py
────────────
Modal deployment for Sabkat's Football Offside Detection System.

Deploy:
    modal deploy modal_app.py
"""

import modal

# ── Image ──────────────────────────────────────────────────────────────────────
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime",
        add_python="3.10",
    )
    .apt_install(
        "git", "libgl1", "libglib2.0-0", "libsm6",
        "libxrender1", "libxext6", "ffmpeg",
    )
    .pip_install_from_requirements("requirements.txt")
    .add_local_file("download_models.py", remote_path="/app/download_models.py", copy=True)
    # Models baked into image at /app/data/models — volume NOT mounted here
    .run_commands(
        "mkdir -p /app/data/models",
        "cd /app && python download_models.py",
    )
    .add_local_dir("backend",  remote_path="/app/backend")
    .add_local_dir("frontend", remote_path="/app/frontend")
)

# ── Modal App ──────────────────────────────────────────────────────────────────
app = modal.App("offside-detection", image=image)

# Volume mounted at /vol — separate from /app/data where models live
data_volume = modal.Volume.from_name("offside-data", create_if_missing=True)


# ── ASGI endpoint ──────────────────────────────────────────────────────────────
@app.function(
    cpu=4,
    memory=8192,
    timeout=600,
    volumes={"/vol": data_volume},  # clean separate mount point
    gpu="T4",  # uncomment for GPU
)
@modal.asgi_app()
def fastapi_app():
    import sys
    import os
    sys.path.insert(0, "/app/backend")

    # Uploads and outputs go to the persistent volume at /vol
    os.environ["UPLOAD_DIR"]  = "/vol/uploads"
    os.environ["OUTPUT_DIR"]  = "/vol/outputs"
    # Models stay in the image
    os.environ["MODELS_DIR"]  = "/app/data/models"

    os.makedirs("/vol/uploads", exist_ok=True)
    os.makedirs("/vol/outputs", exist_ok=True)

    from main import app as _app
    return _app