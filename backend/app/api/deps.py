from __future__ import annotations

import hmac
from collections.abc import Callable
from typing import Any

from fastapi import BackgroundTasks, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.config.settings import get_settings
from app.database.models import Worker, utcnow
from app.database.session import get_db
from app.workers.queue import authenticate_worker

DB = Depends(get_db)


def get_or_404(db: Session, model: Any, id_: str) -> Any:
    obj = db.get(model, id_)
    if obj is None:
        raise HTTPException(404, f"{model.__name__} '{id_}' not found")
    return obj


def deferrer(background: BackgroundTasks) -> Callable[..., None]:
    return lambda fn, *args: background.add_task(fn, *args)


def require_admin(x_admin_token: str = Header(default="")) -> None:
    expected = get_settings().admin_token
    if not expected:
        raise HTTPException(503, "Worker registration disabled: set PE_ADMIN_TOKEN on the backend")
    if not hmac.compare_digest(x_admin_token, expected):
        raise HTTPException(401, "Invalid admin token")


def current_worker(authorization: str = Header(default=""), db: Session = DB) -> Worker:
    token = authorization.removeprefix("Bearer ").strip()
    w = authenticate_worker(db, token)
    if w is None:
        raise HTTPException(401, "Invalid or revoked worker token")
    w.last_seen_at = utcnow()
    db.commit()
    return w
