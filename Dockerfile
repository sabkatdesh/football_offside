# ─────────────────────────────────────────────────────────────────────────────
# Offside Detection — Docker image
# Supports GPU (CUDA 12.1) and CPU fallback automatically.
# Build:   docker build -t offside-app .
# Run GPU: docker run --gpus all -p 8000:8000 offside-app
# Run CPU: docker run -p 8000:8000 offside-app
# ─────────────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY backend/  ./backend/
COPY frontend/ ./frontend/
COPY download_models.py .

# Create data directories
RUN mkdir -p data/models data/uploads data/outputs

# Download models from HuggingFace at build time
RUN python download_models.py

EXPOSE 7860

# Fix: run uvicorn from backend/ so 'app' module is found
WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]