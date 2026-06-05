---
title: Football Offside Detection
emoji: ⚽
colorFrom: blue
colorTo: pink
sdk: docker
pinned: false
---

# Sabkat's Football — Offside Detection System

AI-powered football offside detection from broadcast video clips.
Detects players, classifies teams, computes homography, and renders a frame-accurate offside verdict with a tactical radar overlay.

---

## Architecture

```
Frontend (HTML/CSS/JS)
      │  POST /api/upload
      │  GET  /api/status/{job_id}   ← polling
      │  GET  /api/result/{job_id}
      ▼
FastAPI backend
      │
      ├── YOLOv8l          — player / ball / GK / referee detection
      ├── YOLOv8l-pose     — 32-point pitch keypoint detection
      ├── SigLIP + UMAP + KMeans  — team classification
      ├── ViewTransformer  — frame ↔ pitch homography
      ├── ByteTrack        — multi-object tracking
      └── OffsideDetector  — 2nd-last-defender rule + radar overlay
```

---

## Quick Start

### Option A — Docker (recommended)

**GPU (NVIDIA):**
```bash
docker compose --profile gpu up --build
```

**CPU only:**
```bash
docker compose --profile cpu up --build
```

Models are downloaded from HuggingFace automatically during the build.
Open **http://localhost:8000** in your browser.

---

### Option B — Local (Python 3.10+)

**1. Clone and install:**
```bash
git clone <repo-url>
cd offside-detection
pip install -r requirements.txt
```

**2. Download models:**
```bash
python download_models.py
```
This downloads ~90 MB × 2 from:
- `Sabkat/football-pitch-detection`
- `Sabkat/football-player-detection`

Into `data/models/`.

**3. Run the server:**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000**

---

## Project Structure

```
project/
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── download_models.py          ← downloads models from HuggingFace
├── README.md
│
├── backend/
│   ├── main.py                 ← FastAPI app, CORS, static file serving
│   └── app/
│       ├── __init__.py
│       ├── api/
│       │   ├── __init__.py
│       │   └── routes.py       ← /upload /status /result /video /download
│       ├── services/
│       │   ├── __init__.py
│       │   ├── pipeline.py     ← full CV pipeline (detection → offside → render)
│       │   └── offside.py      ← offside logic (2nd-last defender rule)
│       ├── models/
│       │   ├── __init__.py
│       │   └── schemas.py      ← Pydantic response models
│       └── core/
│           ├── __init__.py
│           └── config.py       ← paths, thresholds, constants
│
├── frontend/
│   ├── index.html              ← single-page app (4 states)
│   ├── style.css               ← dark tactical design system
│   └── app.js                  ← upload, polling, result rendering
│
└── data/
    ├── models/                 ← .pt model files (auto-downloaded)
    ├── uploads/                ← temp input videos (auto-cleaned)
    └── outputs/                ← annotated output videos
```

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload video, returns `job_id` |
| `GET`  | `/api/status/{job_id}` | Poll progress (0–100) and status |
| `GET`  | `/api/result/{job_id}` | Get verdict + video URL |
| `GET`  | `/api/video/{job_id}` | Stream annotated video |
| `GET`  | `/api/download/{job_id}` | Download annotated video |

**Verdict values:** `OFFSIDE` | `ONSIDE` | `NO_FOOTBALL_DETECTED`

---

## Supported Input

Broadcast-style wide-angle football footage from professional matches (Bundesliga, EPL, UCL, La Liga).

**Not supported:** phone footage from stands, behind-goal angles, heavy zoom, amateur footage.

---

## Offside Logic

1. **Direction lock** — goalkeeper positions establish which team attacks which direction (done once per clip).
2. **Offside line** — 2nd-to-last defender (closest to their goal) sets the x-coordinate of the line, per FIFA rules.
3. **Check** — any attacker whose pitch-coordinate distance to the defending goal is less than the offside line distance is flagged.
4. **Fallback** — fewer than 4 pitch landmarks → pixel-coordinate comparison with LOW CONFIDENCE.

---

## Known Limitations

- Single fixed camera — real VAR uses 30+ synchronized cameras
- CPU processing takes 2–5 min per clip; GPU reduces this to seconds
- Broadcast footage only — other angles cause homography failure
- At least 4 pitch landmarks must be visible for full accuracy

---

## Models

| Model | HuggingFace | Purpose |
|-------|-------------|---------|
| `football-player-detection.pt` | `Sabkat/football-player-detection` | YOLOv8l — players, GK, referee, ball |
| `football-pitch-detection.pt`  | `Sabkat/football-pitch-detection`  | YOLOv8l-pose — 32 pitch keypoints |
