"""SQLAlchemy models for the annotation service (Phase 5)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Item(Base):
    __tablename__ = "items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sentence_a: Mapped[str] = mapped_column(String, nullable=False)
    sentence_b: Mapped[str] = mapped_column(String, nullable=False)
    uncertainty: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String, default="pending")  # pending|labeled|gold
    gold_label: Mapped[float | None] = mapped_column(Float, nullable=True)


class Annotator(Base):
    __tablename__ = "annotators"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    reliability: Mapped[float] = mapped_column(Float, default=1.0)


class Annotation(Base):
    __tablename__ = "annotations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id"))
    annotator_id: Mapped[int] = mapped_column(ForeignKey("annotators.id"))
    label: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
