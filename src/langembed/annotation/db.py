"""Database engine/session factory for the annotation service (Phase 5)."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from langembed.annotation.models import Base
from langembed.config import get_settings

_engine: Engine | None = None
_Session: sessionmaker[Session] | None = None


def init() -> None:
    global _engine, _Session
    if _engine is None:
        _engine = create_engine(get_settings().database_url, future=True)
        Base.metadata.create_all(_engine)
        _Session = sessionmaker(bind=_engine, future=True)


def get_db() -> Iterator[Session]:
    init()
    assert _Session is not None
    db = _Session()
    try:
        yield db
    finally:
        db.close()
