"""Experiment records, code fingerprinting and never-overwrite report export."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config.settings import BACKEND_DIR, REPO_DIR, get_settings
from app.database.models import (
    Analysis,
    Experiment,
    GenerationRun,
    HumanRating,
    Reconstruction,
    RefinementSession,
)
from app.services import storage

TRACKED_PACKAGES = ("fastapi", "pydantic", "sqlalchemy", "numpy", "pillow", "torch", "diffusers", "transformers",
                    "accelerate", "lpips")


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True, text=True, timeout=5,
                              check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def code_fingerprint() -> dict[str, Any]:
    """Identify exactly which code produced a result: git commit, dirty flag, and a content hash of the
    backend source (stable even outside git)."""
    h = hashlib.sha256()
    for p in sorted((BACKEND_DIR / "app").rglob("*.py")):
        h.update(p.relative_to(BACKEND_DIR).as_posix().encode())
        h.update(p.read_bytes())
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "source_sha256": h.hexdigest(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "captured_at": datetime.now(UTC).isoformat(),
    }


def _metrics(g: GenerationRun) -> list[dict[str, Any]]:
    return [{"metric": m.metric, "role": m.role, "status": m.status, "value": m.value,
             "higher_is_better": m.higher_is_better, "backend": m.backend, "detail": m.detail} for m in g.metrics]


def reconstruction_record(db: Session, r: Reconstruction) -> dict[str, Any]:
    sessions = db.scalars(select(RefinementSession).where(RefinementSession.reconstruction_id == r.id)).all()
    iter_by_gen = {i.generation_id: (s.id, i.iteration) for s in sessions for i in s.iterations if i.generation_id}
    return {
        "reconstruction_id": r.id,
        "name": r.name,
        "reference_image": {"id": r.image.id, "sha256": r.image.sha256, "path": r.image.path,
                            "width": r.image.width, "height": r.image.height},
        "analyses": [{"id": a.id, "image_id": a.image_id, "mode": a.mode, "provider": a.provider, "model": a.model,
                      "status": a.status, "result": a.result, "caption": a.caption, "error": a.error,
                      "duration_s": a.duration_s, "created_at": a.created_at.isoformat()}
                     for a in db.scalars(select(Analysis).where(Analysis.reconstruction_id == r.id))],
        "prompt_versions": [{"id": p.id, "version": p.version, "parent_id": p.parent_id, "source": p.source,
                             "template_id": p.template_id, "positive": p.positive, "negative": p.negative,
                             "attributes": p.attributes, "validation_ok": (p.validation or {}).get("ok"),
                             "note": p.note, "created_at": p.created_at.isoformat()} for p in r.prompt_versions],
        "generations": [{
            "id": g.id, "condition": g.condition, "prompt_version_id": g.prompt_version_id,
            "prompt_version": g.prompt_version.version, "provider": g.provider, "model_key": g.model_key,
            "model_id": g.model_id, "params": g.params, "seed": g.params.get("seed"), "status": g.status,
            "error": g.error, "duration_s": g.duration_s, "runtime_info": g.runtime_info,
            "output_image": g.output_image.path if g.output_image else None,
            "refinement_session_id": iter_by_gen.get(g.id, (None, None))[0],
            "iteration": iter_by_gen.get(g.id, (None, None))[1],
            "metrics": _metrics(g),
            "human_ratings": [{"rater": h.rater, "similarity": h.similarity, "subject_match": h.subject_match,
                               "style_match": h.style_match, "blind": h.blind, "notes": h.notes,
                               "created_at": h.created_at.isoformat()}
                              for h in db.scalars(select(HumanRating).where(HumanRating.generation_id == g.id))],
            "created_at": g.created_at.isoformat()} for g in r.generations],
        "refinement_sessions": [{
            "id": s.id, "optimization_metric": s.optimization_metric, "max_iterations": s.max_iterations,
            "patience": s.patience, "min_delta": s.min_delta, "status": s.status, "stop_reason": s.stop_reason,
            "best_iteration_id": s.best_iteration_id, "generation_config": s.generation_config,
            "iterations": [{"id": i.id, "iteration": i.iteration, "status": i.status,
                            "optimization_score": i.optimization_score, "prompt_version_id": i.prompt_version_id,
                            "generation_id": i.generation_id, "proposal": i.proposal} for i in s.iterations],
        } for s in sessions],
    }


def experiment_record(db: Session, exp: Experiment) -> dict[str, Any]:
    recons = db.scalars(select(Reconstruction).where(Reconstruction.experiment_id == exp.id)
                        .order_by(Reconstruction.created_at)).all()
    return {
        "schema": "pe.experiment.v1",
        "experiment": {"id": exp.id, "name": exp.name, "description": exp.description, "hypothesis": exp.hypothesis,
                       "config": exp.config, "created_at": exp.created_at.isoformat(),
                       "code_fingerprint_at_creation": exp.code_fingerprint},
        "code_fingerprint_at_export": code_fingerprint(),
        "exported_at": datetime.now(UTC).isoformat(),
        "reconstructions": [reconstruction_record(db, r) for r in recons],
        "comparison": _aggregate(db, exp.id),
    }


def _aggregate(db: Session, experiment_id: str) -> dict[str, Any]:
    from app.experiments.comparison import aggregate

    return aggregate(db, experiment_id)


CSV_FIELDS = ("experiment_id", "reconstruction_id", "reference_sha256", "generation_id", "condition",
              "refinement_session_id", "iteration", "prompt_version", "prompt_source", "template_id", "positive",
              "negative", "provider", "model_id", "seed", "steps", "cfg_scale", "width", "height", "status",
              "duration_s", "metric", "metric_role", "metric_status", "metric_value", "metric_backend", "git_commit",
              "source_sha256", "created_at")


def experiment_csv(record: dict[str, Any]) -> str:
    """Long format: one row per (generation, metric); generations without metrics get one row."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
    w.writeheader()
    fp = record["code_fingerprint_at_export"]
    for r in record["reconstructions"]:
        pvs = {p["id"]: p for p in r["prompt_versions"]}
        for g in r["generations"]:
            pv = pvs.get(g["prompt_version_id"], {})
            base = {
                "experiment_id": record["experiment"]["id"], "reconstruction_id": r["reconstruction_id"],
                "reference_sha256": r["reference_image"]["sha256"], "generation_id": g["id"],
                "condition": g["condition"], "refinement_session_id": g["refinement_session_id"],
                "iteration": g["iteration"], "prompt_version": pv.get("version"), "prompt_source": pv.get("source"),
                "template_id": pv.get("template_id"), "positive": pv.get("positive"), "negative": pv.get("negative"),
                "provider": g["provider"], "model_id": g["model_id"], "seed": g["params"].get("seed"),
                "steps": g["params"].get("steps"), "cfg_scale": g["params"].get("cfg_scale"),
                "width": g["params"].get("width"), "height": g["params"].get("height"), "status": g["status"],
                "duration_s": g["duration_s"], "git_commit": fp.get("git_commit"),
                "source_sha256": fp.get("source_sha256"), "created_at": g["created_at"],
            }
            for m in g["metrics"] or [None]:
                row = dict(base)
                if m:
                    row |= {"metric": m["metric"], "metric_role": m["role"], "metric_status": m["status"],
                            "metric_value": m["value"], "metric_backend": m["backend"]}
                w.writerow(row)
    return buf.getvalue()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "experiment"


def _exclusive_write(path: Path, data: bytes) -> Path:
    """Create a new file; never overwrite. On collision, append a counter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stem, suffix = path.stem, path.suffix
    for n in range(1000):
        cand = path if n == 0 else path.with_name(f"{stem}_{n}{suffix}")
        try:
            with open(cand, "xb") as f:
                f.write(data)
            return cand
        except FileExistsError:
            continue
    raise RuntimeError("Could not find a free report filename")


def write_report(db: Session, exp: Experiment) -> dict[str, str]:
    s = get_settings()
    record = experiment_record(db, exp)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = f"{stamp}_{_slug(exp.name)}_{exp.id[:8]}"
    js = _exclusive_write(s.reports_dir / f"{base}.json", json.dumps(record, indent=2, default=str).encode())
    cs = _exclusive_write(s.reports_dir / f"{base}.csv", experiment_csv(record).encode())
    prompts = [{"reconstruction_id": r["reconstruction_id"], "prompt_versions": r["prompt_versions"]}
               for r in record["reconstructions"]]
    ps = _exclusive_write(s.prompts_dir / f"{base}.prompts.json", json.dumps(prompts, indent=2).encode())
    metrics = [{"reconstruction_id": r["reconstruction_id"], "generation_id": g["id"], "condition": g["condition"],
                "iteration": g["iteration"], "metrics": g["metrics"]}
               for r in record["reconstructions"] for g in r["generations"]]
    ms = _exclusive_write(s.metrics_dir / f"{base}.metrics.json", json.dumps(metrics, indent=2).encode())
    root = s.experiments_dir
    return {k: p.relative_to(root).as_posix() for k, p in
            {"json": js, "csv": cs, "prompts": ps, "metrics": ms}.items()}


# ---------------------------------------------------------------------------------------- demo bundles
def _thumb(src: Path, dest: Path, max_side: int = 768) -> None:
    with PILImage.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, format="JPEG", quality=90)


def comparison(db: Session, r: Reconstruction) -> dict[str, Any]:
    """Latest completed generation per condition (A caption, D detailed, B structured) and, for C, the FINAL scored
    iteration of the latest refinement session — not the best one, which is selected post hoc by the optimisation
    metric and would overstate refinement (selection bias). The best iteration is reported alongside."""
    out: dict[str, Any] = {}
    for cond in ("A_caption", "D_detailed", "B_structured"):
        g = next((g for g in reversed(r.generations) if g.condition == cond and g.status == "completed"), None) or \
            next((g for g in reversed(r.generations) if g.condition == cond), None)
        out[cond] = g
    chosen = None
    for s in db.scalars(select(RefinementSession).where(RefinementSession.reconstruction_id == r.id)
                        .order_by(RefinementSession.created_at.desc())):
        scored = [i for i in s.iterations if i.status == "scored" and i.generation_id and i.iteration > 0]
        if scored:
            chosen = (s, max(scored, key=lambda i: i.iteration))
            break
    out["C_refined"] = db.get(GenerationRun, chosen[1].generation_id) if chosen else None
    out["_session"] = chosen
    return out


def export_demo_bundle(db: Session, r: Reconstruction) -> str:
    """Write a self-contained, git-trackable bundle of a genuine A/B/C comparison under experiments/demo/."""
    s = get_settings()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    rel = Path("demo") / f"{stamp}_{_slug(r.name)}_{r.id[:8]}"
    root = s.experiments_dir / rel
    root.mkdir(parents=True, exist_ok=False)
    _thumb(storage.image_path(r.image), root / "reference.jpg")
    comp = comparison(db, r)
    conditions: dict[str, Any] = {}
    for cond in ("A_caption", "D_detailed", "B_structured", "C_refined"):
        g: GenerationRun | None = comp[cond]
        if g is None:
            conditions[cond] = {"status": "not_executed"}
            continue
        entry: dict[str, Any] = {
            "status": g.status, "generation_id": g.id, "prompt": g.prompt_version.positive,
            "negative_prompt": g.params.get("negative_prompt"), "prompt_version": g.prompt_version.version,
            "prompt_source": g.prompt_version.source, "template_id": g.prompt_version.template_id,
            "provider": g.provider, "model_key": g.model_key, "model_id": g.model_id, "params": g.params,
            "duration_s": g.duration_s, "runtime_info": g.runtime_info, "metrics": _metrics(g),
        }
        if g.output_image:
            _thumb(storage.image_path(g.output_image), root / f"{cond}.jpg")
            entry["image"] = f"{cond}.jpg"
        conditions[cond] = entry
    if comp["_session"]:
        sess, it = comp["_session"]
        best = next((i for i in sess.iterations if i.id == sess.best_iteration_id), None)
        conditions["C_refined"]["refinement"] = {
            "session_id": sess.id, "shown_iteration": it.iteration, "shown": "final_scored_iteration",
            "best_iteration": best.iteration if best else None,
            "best_note": "Best iteration is selected by the optimisation metric (biased); the final one is shown.",
            "optimization_metric": sess.optimization_metric,
            "history": [{"iteration": i.iteration, "status": i.status, "score": i.optimization_score}
                        for i in sess.iterations],
        }
    manifest = {"schema": "pe.demo.v1", "name": r.name, "reconstruction_id": r.id, "reference_image": "reference.jpg",
                "reference_sha256": r.image.sha256, "created_at": datetime.now(UTC).isoformat(),
                "code_fingerprint": code_fingerprint(), "conditions": conditions,
                "provenance": "Exported from genuine pipeline runs recorded in the database."}
    _exclusive_write(root / "manifest.json", json.dumps(manifest, indent=2, default=str).encode())
    return rel.as_posix()


def list_demo_bundles() -> list[dict[str, Any]]:
    root = get_settings().experiments_dir / "demo"
    out = []
    if root.exists():
        for d in sorted(root.iterdir(), reverse=True):
            mf = d / "manifest.json"
            if mf.is_file():
                try:
                    m = json.loads(mf.read_text())
                except json.JSONDecodeError:
                    continue
                if m.get("schema") == "pe.demo.v1":
                    out.append({"id": d.name, "name": m.get("name"), "created_at": m.get("created_at"),
                                "conditions": {k: v.get("status") for k, v in m.get("conditions", {}).items()}})
    return out


def load_demo_bundle(bundle_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", bundle_id) or bundle_id.startswith("."):
        raise ValueError("invalid bundle id")
    mf = get_settings().experiments_dir / "demo" / bundle_id / "manifest.json"
    return json.loads(mf.read_text())


def new_experiment(db: Session, name: str, description: str, hypothesis: str, config: dict[str, Any]) -> Experiment:
    exp = Experiment(name=name, description=description, hypothesis=hypothesis, config=config,
                     code_fingerprint=code_fingerprint())
    db.add(exp)
    db.commit()
    return exp

