"""Database-backed job queue with leases.

Lifecycle::

    queued ──claim──▶ running ──complete──▶ succeeded
       ▲                 │  └──fail(retryable, attempts<max)──▶ queued
       │                 └──fail(final) / attempts exhausted──▶ failed
       └──lease expired (worker crashed / laptop slept) — requeued automatically

A worker holds a *lease* that it extends with progress heartbeats. If the lease expires the job is
re-queued (resume after interruption) until ``max_attempts`` is reached. All generation jobs are seeded,
so a re-run reproduces the same output.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import timedelta
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.config.settings import get_settings
from app.database.models import Job, Worker, utcnow
from app.database.session import as_utc
from app.workers.schemas import JOB_TYPES, PAYLOAD_MODELS


class JobError(Exception):
    pass


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def register_worker(db: Session, name: str, capabilities: dict[str, Any] | None = None) -> tuple[Worker, str]:
    token = "pew_" + secrets.token_urlsafe(32)
    w = Worker(name=name[:128], token_hash=hash_token(token), capabilities=capabilities or {})
    db.add(w)
    db.commit()
    return w, token


def authenticate_worker(db: Session, token: str) -> Worker | None:
    if not token:
        return None
    w = db.scalar(select(Worker).where(Worker.token_hash == hash_token(token)))
    if w is None or w.revoked:
        return None
    return w


def create_job(db: Session, job_type: str, payload: dict[str, Any], priority: int = 0, commit: bool = True) -> Job:
    if job_type not in JOB_TYPES:
        raise JobError(f"Unknown job type '{job_type}'")
    validated = PAYLOAD_MODELS[job_type].model_validate(payload).model_dump()
    job = Job(type=job_type, payload=validated, priority=priority, status="queued",
              max_attempts=get_settings().job_max_attempts)
    db.add(job)
    if commit:
        db.commit()
    else:
        db.flush()
    return job


def requeue_expired(db: Session) -> list[Job]:
    """Recover jobs whose worker stopped heart-beating. Returns jobs that changed state."""
    now = utcnow()
    changed: list[Job] = []
    for job in db.scalars(select(Job).where(Job.status == "running")):
        exp = as_utc(job.lease_expires_at)
        if exp is not None and exp < now:
            if job.attempts >= job.max_attempts:
                job.status = "failed"
                job.error = f"Lease expired after {job.attempts} attempt(s); giving up"
                job.finished_at = now
            else:
                job.status = "queued"
                job.progress_message = f"Re-queued after lease expiry (attempt {job.attempts} interrupted)"
            job.worker_id = None
            job.lease_expires_at = None
            changed.append(job)
    if changed:
        db.commit()
    return changed


def claim_job(db: Session, worker: Worker, types: list[str] | None = None) -> Job | None:
    from app.workers.results import on_job_finalised, on_job_requeued

    for j in requeue_expired(db):
        (on_job_finalised if j.status == "failed" else on_job_requeued)(db, j)
    types = [t for t in (types or list(JOB_TYPES)) if t in JOB_TYPES]
    lease = timedelta(seconds=get_settings().job_lease_seconds)
    candidates = db.scalars(
        select(Job.id).where(Job.status == "queued", Job.type.in_(types))
        .order_by(Job.priority.desc(), Job.created_at).limit(5)
    ).all()
    for job_id in candidates:
        now = utcnow()
        # Compare-and-set so two workers can never claim the same job.
        res = db.execute(
            update(Job).where(Job.id == job_id, Job.status == "queued")
            .values(status="running", worker_id=worker.id, lease_expires_at=now + lease, attempts=Job.attempts + 1,
                    started_at=now, progress=0.0, progress_message="Claimed", error=None)
        )
        if res.rowcount == 1:
            db.commit()
            job = db.get(Job, job_id)
            db.refresh(job)
            from app.workers.results import on_job_started

            on_job_started(db, job)
            return job
        db.rollback()
    return None


def _owned_running(db: Session, job_id: str, worker: Worker) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise JobError("Job not found")
    if job.worker_id != worker.id or job.status != "running":
        raise JobError("Job is not leased to this worker")
    return job


def report_progress(db: Session, job_id: str, worker: Worker, progress: float, message: str = "") -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise JobError("Job not found")
    if job.status == "cancelled":
        return job  # worker sees status and stops
    job = _owned_running(db, job_id, worker)
    job.progress = max(0.0, min(1.0, progress))
    job.progress_message = message[:255]
    job.lease_expires_at = utcnow() + timedelta(seconds=get_settings().job_lease_seconds)
    worker.last_seen_at = utcnow()
    db.commit()
    return job


def complete_job(db: Session, job_id: str, worker: Worker, result: dict[str, Any]) -> Job:
    job = _owned_running(db, job_id, worker)
    job.status = "succeeded"
    job.result = result
    job.progress = 1.0
    job.progress_message = "Completed"
    job.finished_at = utcnow()
    job.lease_expires_at = None
    db.commit()
    return job


def fail_job(db: Session, job_id: str, worker: Worker | None, error: str, retryable: bool = True) -> Job:
    job = db.get(Job, job_id) if worker is None else _owned_running(db, job_id, worker)
    if job is None:
        raise JobError("Job not found")
    job.error = error[:4000]
    job.lease_expires_at = None
    job.worker_id = None
    if retryable and job.attempts < job.max_attempts:
        job.status = "queued"
        job.progress_message = f"Retrying after failure (attempt {job.attempts}/{job.max_attempts})"
    else:
        job.status = "failed"
        job.finished_at = utcnow()
    db.commit()
    return job


def cancel_job(db: Session, job_id: str) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise JobError("Job not found")
    if job.status in ("succeeded", "failed", "cancelled"):
        raise JobError(f"Job already {job.status}")
    job.status = "cancelled"
    job.finished_at = utcnow()
    job.lease_expires_at = None
    db.commit()
    return job


def retry_job(db: Session, job_id: str) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise JobError("Job not found")
    if job.status not in ("failed", "cancelled"):
        raise JobError("Only failed or cancelled jobs can be retried")
    job.status = "queued"
    job.attempts = 0
    job.error = None
    job.finished_at = None
    job.progress = 0.0
    job.progress_message = "Manually re-queued"
    db.commit()
    return job
