from pydantic import BaseModel
from typing import Optional


class JobResponse(BaseModel):
    """Returned immediately after upload — gives the client a job ID to poll."""
    job_id: str
    message: str = "Processing started"


class StatusResponse(BaseModel):
    """Returned while processing is in progress."""
    job_id: str
    status: str          # "queued" | "processing" | "done" | "error"
    progress: int        # 0 - 100
    message: Optional[str] = None


class ResultResponse(BaseModel):
    """Returned when processing is complete."""
    job_id:    str
    verdict:   str       # "OFFSIDE" | "ONSIDE" | "NO_FOOTBALL_DETECTED"
    video_url: str       # URL to stream the processed video
    message:   Optional[str] = None


class ErrorResponse(BaseModel):
    """Returned on any error."""
    error:  str
    detail: Optional[str] = None