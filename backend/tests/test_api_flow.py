"""End-to-end API tests driving the full pipeline with a simulated worker (test fixtures only)."""
from __future__ import annotations

import json

from tests.conftest import FIXTURE_ATTRIBUTES, fixture_png


def W(token):  # noqa: N802
    return {"Authorization": f"Bearer {token}"}


def manual_prompts(client, rid):
    a = client.post(f"/api/reconstructions/{rid}/analyses/manual",
                    json={"attributes": FIXTURE_ATTRIBUTES, "annotator": "test"}).json()
    assert a["is_manual"] and a["provider"] == "manual"
    pvs = client.post(f"/api/reconstructions/{rid}/prompts/from-analysis",
                      json={"analysis_id": a["id"], "template_id": "structured_v1"}).json()
    return a, pvs


def worker_complete_generate(client, token, job, seed_offset=0):
    """Simulated worker: returns a synthetic FIXTURE image of the requested size."""
    p = job["payload"]
    r = client.post(f"/api/worker/jobs/{job['id']}/complete",
                    data={"result": json.dumps({"duration_s": 0.01, "runtime_info": {"fixture": True}})},
                    files={"image": ("o.png", fixture_png(p["width"], p["height"], seed=1 + seed_offset), "image/png")},
                    headers=W(token))
    assert r.status_code == 200, r.text
    return r.json()


def test_health_and_system(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    s = client.get("/api/system").json()
    assert {m["key"] for m in s["models"]} == {"sd15", "sd_turbo", "sdxl"}
    assert {m["key"] for m in s["metrics"]} >= {"clip_l14", "dinov2_base", "ssim", "lpips_alex"}
    vlm = {p["name"]: p for p in s["vlm_providers"]}
    assert vlm["openai_compatible"]["available"] is False  # not configured in tests
    assert vlm["hf_router"]["available"] is False
    assert s["gpu_worker_online"] is False
    assert s["code"]["source_sha256"]


def test_upload_rejects_non_images(client):
    r = client.post("/api/reconstructions", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_upload_dedupes_and_serves(client, recon):
    img = recon["image"]
    assert img["width"] == 96 and img["height"] == 64
    assert client.get(img["url"]).content == fixture_png()


def test_prompt_versions_preserve_original_and_validated(client, recon):
    _, pvs = manual_prompts(client, recon["id"])
    assert [p["source"] for p in pvs] == ["structured", "validated"]
    assert pvs[1]["parent_id"] == pvs[0]["id"]
    assert pvs[0]["positive"].startswith("photograph, a red bicycle")
    # edit creates a new immutable version
    attrs = dict(FIXTURE_ATTRIBUTES, lighting="golden hour")
    e = client.post(f"/api/reconstructions/{recon['id']}/prompts",
                    json={"parent_id": pvs[1]["id"], "attributes": attrs, "note": "warmer"}).json()
    assert e["version"] == 3 and e["source"] == "edited" and "golden hour" in e["positive"]
    d = client.get("/api/prompts/diff", params={"a": pvs[1]["id"], "b": e["id"]}).json()
    assert any(c["field"] == "lighting" for c in d["attributes"])
    assert any(o["op"] == "insert" for o in d["positive"])


def test_preview_flags_contradictions(client):
    r = client.post("/api/prompts/preview",
                    json={"attributes": dict(FIXTURE_ATTRIBUTES, lighting="moonlit night")}).json()
    codes = {i["code"] for i in r["original"]["validation"]["issues"]}
    assert "contradiction.time_of_day" not in codes  # daylight was replaced, no conflict
    r = client.post("/api/prompts/preview", json={
        "attributes": dict(FIXTURE_ATTRIBUTES, extra_positive=["moonlit night"])}).json()
    codes = {i["code"] for i in r["original"]["validation"]["issues"]}
    assert "contradiction.time_of_day" in codes


def test_unavailable_inline_provider_is_refused_not_faked(client, recon):
    r = client.post(f"/api/reconstructions/{recon['id']}/analyses", json={"provider": "openai_compatible"})
    assert r.status_code == 422 and "not configured" in r.text
    _, pvs = manual_prompts(client, recon["id"])
    r = client.post(f"/api/reconstructions/{recon['id']}/generations",
                    json={"prompt_version_id": pvs[1]["id"], "provider": "hf_inference"})
    assert r.status_code == 422


def test_generation_params_validated(client, recon):
    _, pvs = manual_prompts(client, recon["id"])
    r = client.post(f"/api/reconstructions/{recon['id']}/generations",
                    json={"prompt_version_id": pvs[1]["id"], "params": {"model_key": "sd_turbo", "cfg_scale": 7}})
    assert r.status_code == 422  # SD-Turbo is trained without CFG
    r = client.post(f"/api/reconstructions/{recon['id']}/generations",
                    json={"prompt_version_id": pvs[1]["id"], "params": {"model_key": "sdxl", "width": 1004}})
    assert r.status_code == 422  # not a multiple of 8


def test_worker_auth_required(client):
    assert client.post("/api/worker/jobs/claim", json={}).status_code == 401
    assert client.post("/api/worker/register", json={"name": "x"}, headers={"X-Admin-Token": "wrong"}
                       ).status_code == 401


def test_full_worker_flow_generation_and_evaluation(client, recon, worker_token):
    rid = recon["id"]
    _, pvs = manual_prompts(client, rid)
    g = client.post(f"/api/reconstructions/{rid}/generations",
                    json={"prompt_version_id": pvs[1]["id"], "condition": "B_structured",
                          "params": {"model_key": "sd15", "seed": 7, "width": 512, "height": 384}}).json()
    assert g["status"] == "queued" and g["params"]["steps"] == 30 and g["params"]["negative_prompt"]
    assert client.get("/api/system").json()["jobs"]["queued"] == 1

    job = client.post("/api/worker/jobs/claim", json={"types": ["generate"]}, headers=W(worker_token)).json()
    assert job["type"] == "generate" and job["payload"]["seed"] == 7 and job["payload"]["prompt"] == pvs[1]["positive"]
    assert client.get(f"/api/generations/{g['id']}").json()["status"] == "running"
    # worker cannot fetch images not referenced by its job
    assert client.get(f"/api/worker/jobs/{job['id']}/images/{recon['image']['id']}",
                      headers=W(worker_token)).status_code == 403
    pr = client.post(f"/api/worker/jobs/{job['id']}/progress", json={"progress": 0.5, "message": "denoising"},
                     headers=W(worker_token)).json()
    assert pr == {"status": "running", "cancel": False}

    # wrong-size upload is rejected and the job fails rather than storing a bogus result
    bad = client.post(f"/api/worker/jobs/{job['id']}/complete",
                      data={"result": json.dumps({"duration_s": 1.0})},
                      files={"image": ("o.png", fixture_png(64, 64), "image/png")}, headers=W(worker_token))
    assert bad.status_code == 422
    assert client.get(f"/api/generations/{g['id']}").json()["status"] == "failed"

    # retry and complete properly
    assert client.post(f"/api/jobs/{job['id']}/retry").json()["status"] == "queued"
    job = client.post("/api/worker/jobs/claim", json={}, headers=W(worker_token)).json()
    worker_complete_generate(client, worker_token, job)
    gen = client.get(f"/api/generations/{g['id']}").json()
    assert gen["status"] == "completed" and gen["output_image"]["width"] == 512

    metrics = {m["metric"]: m for m in gen["metrics"]}
    # SSIM really computed inline (pure numpy); torch metrics wait for the worker with NO value.
    assert metrics["ssim"]["status"] == "completed" and -1 <= metrics["ssim"]["value"] <= 1
    for k in ("clip_l14", "dinov2_base", "lpips_alex"):
        assert metrics[k]["status"] == "queued" and metrics[k]["value"] is None

    ev = client.post("/api/worker/jobs/claim", json={"types": ["evaluate"]}, headers=W(worker_token)).json()
    assert ev["type"] == "evaluate"
    ref = client.get(f"/api/worker/jobs/{ev['id']}/images/{recon['image']['id']}", headers=W(worker_token))
    assert ref.status_code == 200
    reqs = ev["payload"]["metrics"]
    # Simulated worker without torch reports 'unavailable' — the honest outcome.
    res = {"metrics": [{"metric_result_id": r["metric_result_id"], "status": "unavailable",
                        "detail": "torch not installed (test)"} for r in reqs]}
    r = client.post(f"/api/worker/jobs/{ev['id']}/complete", data={"result": json.dumps(res)}, headers=W(worker_token))
    assert r.status_code == 200, r.text
    metrics = {m["metric"]: m for m in client.get(f"/api/generations/{g['id']}").json()["metrics"]}
    assert all(metrics[k]["status"] == "unavailable" and metrics[k]["value"] is None
               for k in ("clip_l14", "dinov2_base", "lpips_alex"))


def test_completed_metric_without_value_rejected(client, recon, worker_token):
    rid = recon["id"]
    _, pvs = manual_prompts(client, rid)
    client.post(f"/api/reconstructions/{rid}/generations", json={"prompt_version_id": pvs[1]["id"]})
    job = client.post("/api/worker/jobs/claim", json={}, headers=W(worker_token)).json()
    worker_complete_generate(client, worker_token, job)
    ev = client.post("/api/worker/jobs/claim", json={}, headers=W(worker_token)).json()
    res = {"metrics": [{"metric_result_id": ev["payload"]["metrics"][0]["metric_result_id"], "status": "completed"}]}
    r = client.post(f"/api/worker/jobs/{ev['id']}/complete", data={"result": json.dumps(res)}, headers=W(worker_token))
    assert r.status_code == 422


def test_analysis_job_via_worker(client, recon, worker_token):
    a = client.post(f"/api/reconstructions/{recon['id']}/analyses",
                    json={"provider": "qwen3vl_local", "mode": "structured"}).json()
    assert a["status"] == "queued" and a["model"] == "Qwen/Qwen3-VL-2B-Instruct"
    job = client.post("/api/worker/jobs/claim", json={}, headers=W(worker_token)).json()
    img = client.get(f"/api/worker/jobs/{job['id']}/images/{recon['image']['id']}", headers=W(worker_token))
    assert img.status_code == 200
    raw = "```json\n" + json.dumps(FIXTURE_ATTRIBUTES) + "\n```"  # FIXTURE text standing in for VLM output
    client.post(f"/api/worker/jobs/{job['id']}/complete", headers=W(worker_token), data={"result": json.dumps(
        {"raw_output": raw, "model": "Qwen/Qwen3-VL-2B-Instruct", "duration_s": 0.1})})
    a2 = client.get(f"/api/analyses/{a['id']}").json()
    assert a2["status"] == "completed" and a2["result"]["subject"] == "a red bicycle"


def test_unparseable_vlm_output_marks_failed(client, recon, worker_token):
    a = client.post(f"/api/reconstructions/{recon['id']}/analyses", json={"mode": "structured"}).json()
    job = client.post("/api/worker/jobs/claim", json={}, headers=W(worker_token)).json()
    client.post(f"/api/worker/jobs/{job['id']}/complete", headers=W(worker_token), data={"result": json.dumps(
        {"raw_output": "I cannot see the image.", "model": "m", "duration_s": 0.1})})
    a2 = client.get(f"/api/analyses/{a['id']}").json()
    assert a2["status"] == "failed" and a2["result"] is None and a2["raw_output"] == "I cannot see the image."


def test_disallowed_model_rejected(client, recon):
    r = client.post(f"/api/reconstructions/{recon['id']}/analyses",
                    json={"provider": "qwen3vl_local", "model": "evil/remote-code-model"})
    assert r.status_code == 422


def test_worker_failure_retry_then_final_failure(client, recon, worker_token):
    rid = recon["id"]
    _, pvs = manual_prompts(client, rid)
    g = client.post(f"/api/reconstructions/{rid}/generations", json={"prompt_version_id": pvs[1]["id"]}).json()
    for attempt in range(3):
        job = client.post("/api/worker/jobs/claim", json={}, headers=W(worker_token)).json()
        assert job["attempts"] == attempt + 1
        client.post(f"/api/worker/jobs/{job['id']}/fail", json={"error": "CUDA OOM", "retryable": True},
                    headers=W(worker_token))
    assert client.post("/api/worker/jobs/claim", json={}, headers=W(worker_token)).status_code == 204
    gen = client.get(f"/api/generations/{g['id']}").json()
    assert gen["status"] == "failed" and "CUDA OOM" in gen["error"]


def test_lease_expiry_requeues_interrupted_job(client, recon, worker_token):
    from datetime import timedelta

    from app.database.models import Job, utcnow
    from app.database.session import get_sessionmaker

    rid = recon["id"]
    _, pvs = manual_prompts(client, rid)
    g = client.post(f"/api/reconstructions/{rid}/generations", json={"prompt_version_id": pvs[1]["id"]}).json()
    job = client.post("/api/worker/jobs/claim", json={}, headers=W(worker_token)).json()
    with get_sessionmaker()() as db:  # simulate the laptop going to sleep mid-job
        j = db.get(Job, job["id"])
        j.lease_expires_at = utcnow() - timedelta(seconds=1)
        db.commit()
    again = client.post("/api/worker/jobs/claim", json={}, headers=W(worker_token)).json()
    assert again["id"] == job["id"] and again["attempts"] == 2
    worker_complete_generate(client, worker_token, again)
    assert client.get(f"/api/generations/{g['id']}").json()["status"] == "completed"


def test_cancel_job(client, recon, worker_token):
    rid = recon["id"]
    _, pvs = manual_prompts(client, rid)
    g = client.post(f"/api/reconstructions/{rid}/generations", json={"prompt_version_id": pvs[1]["id"]}).json()
    job = client.post("/api/worker/jobs/claim", json={}, headers=W(worker_token)).json()
    client.post(f"/api/jobs/{job['id']}/cancel")
    pr = client.post(f"/api/worker/jobs/{job['id']}/progress", json={"progress": 0.3}, headers=W(worker_token))
    assert pr.json()["cancel"] is True
    assert client.get(f"/api/generations/{g['id']}").json()["status"] == "failed"
