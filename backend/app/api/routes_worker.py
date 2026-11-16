"""Endpoints used by the GPU worker agent. All require a worker bearer token, except registration,
which requires the admin token. The worker can only download images referenced by a job it holds."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from app.api import serializers as S
from app.api.deps import DB, current_worker, get_or_404, require_admin
from app.config.settings import get_settings
from app.database.models import Image, Job, Worker
from app.services import storage
from app.workers import queue
from app.workers.results import ResultError, apply_job_result, on_job_finalised, on_job_requeued, validate_result
from app.workers.schemas import image_ids_in_payload

router = APIRouter(prefix="/api/worker", tags=["worker"])


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    capabilities: dict[str, Any] = Field(default_factory=dict)


@router.post("/register", status_code=201, dependencies=[Depends(require_admin)])
def register(body: RegisterRequest, db: Session = DB) -> dict[str, str]:
    w, token = queue.register_worker(db, body.name, body.capabilities)
    return {"worker_id": w.id, "token": token, "note": "Store this token securely; it is shown only once."}


class HeartbeatRequest(BaseModel):
    capabilities: dict[str, Any] = Field(default_factory=dict)


@router.post("/heartbeat")
def heartbeat(body: HeartbeatRequest, worker: Worker = Depends(current_worker), db: Session = DB) -> dict[str, Any]:
    if body.capabilities:
        worker.capabilities = body.capabilities
        db.commit()
    return {"worker_id": worker.id, "name": worker.name, "lease_seconds": get_settings().job_lease_seconds}


class ClaimRequest(BaseModel):
    types: list[str] | None = None


@router.post("/jobs/claim")
def claim(body: ClaimRequest, response: Response, worker: Worker = Depends(current_worker), db: Session = DB
          ) -> dict[str, Any] | None:
    job = queue.claim_job(db, worker, body.types)
    if job is None:
        response.status_code = 204
        return None
    return S.job(job)


class ProgressRequest(BaseModel):
    progress: float = Field(ge=0, le=1)
    message: str = ""


@router.post("/jobs/{jid}/progress")
def progress(jid: str, body: ProgressRequest, worker: Worker = Depends(current_worker), db: Session = DB
             ) -> dict[str, Any]:
    try:
        job = queue.report_progress(db, jid, worker, body.progress, body.message)
    except queue.JobError as e:
        raise HTTPException(409, str(e)) from e
    return {"status": job.status, "cancel": job.status == "cancelled"}


@router.get("/jobs/{jid}/images/{image_id}")
def job_image(jid: str, image_id: str, worker: Worker = Depends(current_worker), db: Session = DB) -> FileResponse:
    job = get_or_404(db, Job, jid)
    if job.worker_id != worker.id or job.status != "running":
        raise HTTPException(403, "Job is not leased to this worker")
    if image_id not in image_ids_in_payload(job.type, job.payload):
        raise HTTPException(403, "Image is not part of this job")
    img = get_or_404(db, Image, image_id)
    return FileResponse(storage.image_path(img), media_type=img.mime)


@router.post("/jobs/{jid}/complete")
async def complete(jid: str, result: str = Form(...), image: UploadFile | None = File(None),
                   worker: Worker = Depends(current_worker), db: Session = DB) -> dict[str, Any]:
    job = get_or_404(db, Job, jid)
    if job.worker_id != worker.id or job.status != "running":
        raise HTTPException(409, "Job is not leased to this worker")
    try:
        payload = validate_result(job, json.loads(result))
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(422, f"Invalid result: {e}") from e
    data = await image.read(get_settings().max_upload_bytes + 1) if image is not None else None
    try:
        apply_job_result(db, job, payload, data)
    except (ResultError, storage.InvalidImage) as e:
        db.rollback()
        job = queue.fail_job(db, jid, worker, f"Result rejected by server: {e}", retryable=False)
        on_job_finalised(db, job)
        raise HTTPException(422, str(e)) from e
    job = queue.complete_job(db, jid, worker, payload)
    return S.job(job)


class FailRequest(BaseModel):
    error: str = Field(max_length=4000)
    retryable: bool = True


@router.post("/jobs/{jid}/fail")
def fail(jid: str, body: FailRequest, worker: Worker = Depends(current_worker), db: Session = DB) -> dict[str, Any]:
    try:
        job = queue.fail_job(db, jid, worker, body.error, body.retryable)
    except queue.JobError as e:
        raise HTTPException(409, str(e)) from e
    (on_job_requeued if job.status == "queued" else on_job_finalised)(db, job)
    return S.job(job)
