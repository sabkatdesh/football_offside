import uuid
import shutil
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, StreamingResponse

from app.models.schemas import JobResponse, StatusResponse, ResultResponse, ErrorResponse
from app.core.config import UPLOAD_DIR, OUTPUT_DIR, ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_MB
from app.services.pipeline import run_pipeline

router = APIRouter()

# ── In-memory job store ────────────────────────────────────────────────────────
# For production you'd use Redis — for this assignment in-memory is fine
jobs: Dict[str, Any] = {}


# ── Upload & start processing ──────────────────────────────────────────────────
@router.post("/upload", response_model=JobResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    # Validate extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}"
        )

    # Generate unique job ID
    job_id      = str(uuid.uuid4())
    input_path  = UPLOAD_DIR / f"{job_id}{suffix}"
    output_path = OUTPUT_DIR / f"{job_id}_output.mp4"

    # Save uploaded file to disk
    try:
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Register job as queued
    jobs[job_id] = {
        "status":      "queued",
        "progress":    0,
        "verdict":     None,
        "input_path":  str(input_path),
        "output_path": str(output_path),
        "message":     "Queued for processing",
    }

    # Run pipeline in background — non-blocking
    background_tasks.add_task(
        process_video, job_id, str(input_path), str(output_path)
    )

    return JobResponse(job_id=job_id, message="Upload successful. Processing started.")


# ── Background processing task ─────────────────────────────────────────────────
def process_video(job_id: str, input_path: str, output_path: str):
    """Runs the CV pipeline in the background and updates job state."""
    try:
        jobs[job_id]["status"]   = "processing"
        jobs[job_id]["progress"] = 5
        jobs[job_id]["message"]  = "Analyzing video..."

        result = run_pipeline(
            source_path=input_path,
            output_path=output_path,
            progress_callback=lambda p: _update_progress(job_id, p)
        )

        jobs[job_id]["status"]   = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["verdict"]  = result["verdict"]
        jobs[job_id]["message"]  = "Processing complete"

    except Exception as e:
        jobs[job_id]["status"]  = "error"
        jobs[job_id]["message"] = str(e)


def _update_progress(job_id: str, progress: int):
    """Called by pipeline to report frame-level progress."""
    if job_id in jobs:
        jobs[job_id]["progress"] = min(int(progress), 99)


# ── Poll status ────────────────────────────────────────────────────────────────
@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job.get("message"),
    )


# ── Get result ─────────────────────────────────────────────────────────────────
@router.get("/result/{job_id}", response_model=ResultResponse)
async def get_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job.get("message", "Processing failed"))

    if job["status"] != "done":
        raise HTTPException(status_code=202, detail="Processing not complete yet")

    if not Path(job["output_path"]).exists():
        raise HTTPException(status_code=500, detail="Output video not found")

    return ResultResponse(
        job_id=job_id,
        verdict=job["verdict"],
        video_url=f"/api/video/{job_id}",
        message="Processing complete",
    )


# ── Stream processed video (byte-range aware — required for browser <video>) ───
@router.get("/video/{job_id}")
async def stream_video(job_id: str, request: Request):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    output_path = Path(jobs[job_id]["output_path"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    file_size = output_path.stat().st_size
    range_header = request.headers.get("Range")

    def iter_file(path: Path, start: int, end: int, chunk: int = 1024 * 256):
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    if range_header:
        # Parse "bytes=start-end"
        range_val = range_header.strip().replace("bytes=", "")
        parts     = range_val.split("-")
        start     = int(parts[0]) if parts[0] else 0
        end       = int(parts[1]) if parts[1] else file_size - 1
        end       = min(end, file_size - 1)
        length    = end - start + 1

        headers = {
            "Content-Range":  f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges":  "bytes",
            "Content-Length": str(length),
            "Content-Type":   "video/mp4",
        }
        return StreamingResponse(
            iter_file(output_path, start, end),
            status_code=206,
            headers=headers,
            media_type="video/mp4",
        )
    else:
        headers = {
            "Accept-Ranges":  "bytes",
            "Content-Length": str(file_size),
            "Content-Type":   "video/mp4",
        }
        return StreamingResponse(
            iter_file(output_path, 0, file_size - 1),
            status_code=200,
            headers=headers,
            media_type="video/mp4",
        )


# ── Download processed video ───────────────────────────────────────────────────
@router.get("/download/{job_id}")
async def download_video(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    output_path = Path(jobs[job_id]["output_path"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=f"offside_analysis_{job_id[:8]}.mp4",
        headers={"Content-Disposition": f"attachment; filename=offside_analysis_{job_id[:8]}.mp4"}
    )