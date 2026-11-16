from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config.settings import get_settings
from app.database.models import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def as_utc(dt: datetime | None) -> datetime | None:
    """SQLite drops tzinfo; treat stored naive datetimes as UTC."""
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def init_engine(url: str | None = None) -> Engine:
    global _engine, _SessionLocal
    settings = get_settings()
    url = url or settings.database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    _engine = create_engine(url, connect_args=connect_args)
    if url.startswith("sqlite"):

        @event.listens_for(_engine, "connect")
        def _pragmas(dbapi_conn, _):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    if _SessionLocal is None:
        init_engine()
    assert _SessionLocal is not None
    return _SessionLocal


def get_db() -> Iterator[Session]:
    db = get_sessionmaker()()
    try:
        yield db
    finally:
        db.close()
