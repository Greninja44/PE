"""SQLAlchemy ORM schema.

Design notes
------------
* Every inference-derived row records *provenance* (provider, model id, status, error). A row whose
  status is not ``completed`` never carries a value — there are no placeholder numbers anywhere.
* Prompt versions are immutable: edits and refinements create new rows linked by ``parent_id``.
* Metric results carry a ``role`` (``optimization`` vs ``evaluation``) so the refinement objective is
  never confused with the independent evaluation.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Image(TimestampMixin, Base):
    __tablename__ = "images"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # "original" | "generated"
    filename: Mapped[str] = mapped_column(String(255))
    path: Mapped[str] = mapped_column(String(1024))  # relative to experiments_dir
    mime: Mapped[str] = mapped_column(String(64))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    size_bytes: Mapped[int] = mapped_column(Integer)


class Reconstruction(TimestampMixin, Base):
    """One reference image being reverse-engineered into a prompt."""

    __tablename__ = "reconstructions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255))
    notes: Mapped[str] = mapped_column(Text, default="")
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id"))
    experiment_id: Mapped[str | None] = mapped_column(ForeignKey("experiments.id"), nullable=True)

    image: Mapped[Image] = relationship()
    analyses: Mapped[list[Analysis]] = relationship(back_populates="reconstruction", order_by="Analysis.created_at")
    prompt_versions: Mapped[list[PromptVersion]] = relationship(
        back_populates="reconstruction", order_by="PromptVersion.version"
    )
    generations: Mapped[list[GenerationRun]] = relationship(
        back_populates="reconstruction", order_by="GenerationRun.created_at"
    )


class Analysis(TimestampMixin, Base):
    """Stage 1 output. ``mode`` is ``structured`` (full attribute JSON) or ``caption`` (single sentence)."""

    __tablename__ = "analyses"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    reconstruction_id: Mapped[str] = mapped_column(ForeignKey("reconstructions.id"), index=True)
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id"))  # usually the reference; may be a generated image
    mode: Mapped[str] = mapped_column(String(16), default="structured")
    provider: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|queued|running|completed|failed
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id"), nullable=True)
    runtime_info: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)  # device, dtype, tokens, memory

    reconstruction: Mapped[Reconstruction] = relationship(back_populates="analyses")


class PromptVersion(TimestampMixin, Base):
    """Immutable prompt snapshot. ``source``: caption|structured|edited|validated|refinement|manual."""

    __tablename__ = "prompt_versions"
    __table_args__ = (UniqueConstraint("reconstruction_id", "version"),)
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    reconstruction_id: Mapped[str] = mapped_column(ForeignKey("reconstructions.id"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("prompt_versions.id"), nullable=True)
    analysis_id: Mapped[str | None] = mapped_column(ForeignKey("analyses.id"), nullable=True)
    source: Mapped[str] = mapped_column(String(32))
    template_id: Mapped[str] = mapped_column(String(64))
    attributes: Mapped[dict[str, Any]] = mapped_column(JSON)
    positive: Mapped[str] = mapped_column(Text)
    negative: Mapped[str] = mapped_column(Text, default="")
    validation: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    note: Mapped[str] = mapped_column(Text, default="")

    reconstruction: Mapped[Reconstruction] = relationship(back_populates="prompt_versions")


class GenerationRun(TimestampMixin, Base):
    __tablename__ = "generation_runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    reconstruction_id: Mapped[str] = mapped_column(ForeignKey("reconstructions.id"), index=True)
    prompt_version_id: Mapped[str] = mapped_column(ForeignKey("prompt_versions.id"))
    condition: Mapped[str | None] = mapped_column(String(32), nullable=True)  # A_caption|B_structured|C_refined
    provider: Mapped[str] = mapped_column(String(64))
    model_key: Mapped[str] = mapped_column(String(64))
    model_id: Mapped[str] = mapped_column(String(255))
    params: Mapped[dict[str, Any]] = mapped_column(JSON)  # seed, steps, cfg_scale, width, height, negative_prompt
    status: Mapped[str] = mapped_column(String(16), default="pending")
    output_image_id: Mapped[str | None] = mapped_column(ForeignKey("images.id"), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    runtime_info: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)  # device, torch/diffusers versions
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id"), nullable=True)

    reconstruction: Mapped[Reconstruction] = relationship(back_populates="generations")
    prompt_version: Mapped[PromptVersion] = relationship()
    output_image: Mapped[Image | None] = relationship()
    metrics: Mapped[list[MetricResult]] = relationship(back_populates="generation", order_by="MetricResult.metric")


class MetricResult(TimestampMixin, Base):
    """One metric for one (reference, generated) pair. ``value`` is NULL unless ``status == 'completed'``."""

    __tablename__ = "metric_results"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    generation_id: Mapped[str] = mapped_column(ForeignKey("generation_runs.id"), index=True)
    metric: Mapped[str] = mapped_column(String(64))  # clip | dinov2 | ssim | lpips
    role: Mapped[str] = mapped_column(String(16))  # optimization | evaluation
    status: Mapped[str] = mapped_column(String(16))  # queued|completed|unavailable|failed
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    higher_is_better: Mapped[bool] = mapped_column(Boolean, default=True)
    backend: Mapped[str] = mapped_column(String(255), default="")  # e.g. "transformers:openai/clip-vit-base-patch32"
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id"), nullable=True)

    generation: Mapped[GenerationRun] = relationship(back_populates="metrics")


class RefinementSession(TimestampMixin, Base):
    __tablename__ = "refinement_sessions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    reconstruction_id: Mapped[str] = mapped_column(ForeignKey("reconstructions.id"), index=True)
    base_prompt_version_id: Mapped[str] = mapped_column(ForeignKey("prompt_versions.id"))
    optimization_metric: Mapped[str] = mapped_column(String(64))
    max_iterations: Mapped[int] = mapped_column(Integer)
    patience: Mapped[int] = mapped_column(Integer)
    min_delta: Mapped[float] = mapped_column(Float)
    generation_config: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(24), default="active")  # active|stopped_early|max_iterations|stopped
    stop_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    best_iteration_id: Mapped[str | None] = mapped_column(String(32), nullable=True)

    iterations: Mapped[list[RefinementIteration]] = relationship(
        back_populates="session", order_by="RefinementIteration.iteration"
    )


class RefinementIteration(TimestampMixin, Base):
    """Iteration 0 is the baseline. Each later iteration starts as a *proposal* awaiting approval."""

    __tablename__ = "refinement_iterations"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("refinement_sessions.id"), index=True)
    iteration: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))  # proposed|approved|rejected|generating|scored|failed
    proposal: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)  # edits + rationale
    prompt_version_id: Mapped[str | None] = mapped_column(ForeignKey("prompt_versions.id"), nullable=True)
    generation_id: Mapped[str | None] = mapped_column(ForeignKey("generation_runs.id"), nullable=True)
    optimization_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    session: Mapped[RefinementSession] = relationship(back_populates="iterations")


class Experiment(TimestampMixin, Base):
    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    hypothesis: Mapped[str] = mapped_column(Text, default="")
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    code_fingerprint: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class HumanRating(TimestampMixin, Base):
    """A human judgement of one generated image against its reference. Stored apart from ``metric_results`` so
    human evaluation is never mixed into automatic metric aggregates."""

    __tablename__ = "human_ratings"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    generation_id: Mapped[str] = mapped_column(ForeignKey("generation_runs.id"), index=True)
    rater: Mapped[str] = mapped_column(String(64))
    similarity: Mapped[int] = mapped_column(Integer)  # 1 (unrelated) … 5 (near-identical)
    subject_match: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1…5, optional
    style_match: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1…5, optional
    blind: Mapped[bool] = mapped_column(Boolean, default=False)  # rater did not see the condition/prompt
    notes: Mapped[str] = mapped_column(Text, default="")


class Job(TimestampMixin, Base):
    """Unit of GPU work. Payloads are typed data only — never commands."""

    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    type: Mapped[str] = mapped_column(String(32), index=True)  # analyze | generate | evaluate
    status: Mapped[str] = mapped_column(String(16), index=True, default="queued")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    worker_id: Mapped[str | None] = mapped_column(ForeignKey("workers.id"), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    progress_message: Mapped[str] = mapped_column(String(255), default="")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Worker(TimestampMixin, Base):
    __tablename__ = "workers"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    capabilities: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
