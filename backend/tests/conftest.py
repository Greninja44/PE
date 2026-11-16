"""Test fixtures.

IMPORTANT: everything here is a deterministic *test fixture* — synthetic gradient images and hand-picked
numbers used only to exercise control flow. None of it is, or is presented as, real model output.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

ADMIN = "test-admin-token"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PE_EXPERIMENTS_DIR", str(tmp_path / "experiments"))
    monkeypatch.setenv("PE_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("PE_ADMIN_TOKEN", ADMIN)
    monkeypatch.setenv("PE_HF_TOKEN", "")
    monkeypatch.setenv("PE_OPENAI_COMPAT_BASE_URL", "")
    from app.config.settings import get_settings

    get_settings.cache_clear()
    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()


def fixture_png(w: int = 96, h: int = 64, seed: int = 0) -> bytes:
    """Synthetic gradient test image (FIXTURE — not model output)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 255, w)[None, :, None]
    y = np.linspace(0, 255, h)[:, None, None]
    arr = np.concatenate([x.repeat(h, 0), y.repeat(w, 1), np.full((h, w, 1), 128.0)], axis=2)
    arr = np.clip(arr + rng.normal(0, 8, arr.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


FIXTURE_ATTRIBUTES = {
    "subject": "a red bicycle",
    "secondary_objects": ["brick wall", "potted plant"],
    "spatial_relationships": [{"subject": "bicycle", "relation": "in front of", "object": "brick wall"}],
    "composition": "centered subject",
    "camera_angle": "eye level",
    "lighting": "soft daylight",
    "colors": ["red", "terracotta"],
    "materials": ["steel"],
    "textures": ["rough brick"],
    "background": "old brick wall",
    "art_style": "photograph",
}


@pytest.fixture()
def recon(client):
    r = client.post("/api/reconstructions", files={"file": ("ref.png", fixture_png(), "image/png")},
                    data={"name": "fixture reconstruction"})
    assert r.status_code == 201, r.text
    return r.json()


@pytest.fixture()
def worker_token(client):
    r = client.post("/api/worker/register", headers={"X-Admin-Token": ADMIN}, json={"name": "test-worker"})
    assert r.status_code == 201, r.text
    return r.json()["token"]
