"""Refinement loop, stopping policy, experiment export and worker-agent protocol tests.
Scores below are hand-picked FIXTURE numbers that exercise the control flow; they are not results."""
from __future__ import annotations

import csv
import io
import json

import numpy as np
import pytest

from app.refinement.policy import decide
from app.refinement.proposer import apply_edits, propose
from app.prompt_engine.schema import ImageAttributes
from tests.conftest import FIXTURE_ATTRIBUTES, fixture_png
from tests.test_api_flow import W, manual_prompts, worker_complete_generate


# ------------------------------------------------------------------------------------------- policy
def test_policy_tracks_best_and_stops_early():
    d = decide([0.50, 0.55, 0.551, 0.54], True, max_iterations=10, patience=2, min_delta=0.005)
    assert d.best_index == 1 and d.status == "stopped_early"


def test_policy_lower_is_better_and_max_iterations():
    d = decide([0.4, 0.3, 0.2], False, max_iterations=2, patience=5, min_delta=0.0)
    assert d.best_index == 2 and d.status == "max_iterations"


def test_policy_failed_iterations_count_but_never_best():
    d = decide([0.5, None, None], True, max_iterations=10, patience=2, min_delta=0.0)
    assert d.best_index == 0 and d.status == "stopped_early"
    assert decide([0.5], True, 3, 2, 0.0).status == "active"


# ------------------------------------------------------------------------------------------- proposer
def test_proposer_uses_generated_analysis_discrepancies():
    ref = ImageAttributes(**FIXTURE_ATTRIBUTES)
    gen = ImageAttributes(subject="a blue bicycle", secondary_objects=["fence", "dog"], colors=["blue"],
                          lighting="harsh midday sun", art_style="digital illustration")
    p = propose(ref, ref, "structured_v1", gen, excluded=set(), max_edits=10)
    ops = {(e.op, e.field, e.value) for e in p.edits}
    assert ("add", "extra_positive", "photorealistic") in ops
    p_styled = propose(ref, ref.model_copy(update={"art_style": "anime"}), "structured_v1", gen, set(), 10)
    assert ("replace", "art_style", "photograph") in {(e.op, e.field, e.value) for e in p_styled.edits}
    assert ("move_front", "secondary_objects", "brick wall") in ops
    assert ("add", "extra_negative", "dog") in ops
    assert p.used_generated_analysis
    new = apply_edits(ref, p.edits)
    assert "dog" in new.extra_negative and new.secondary_objects[0] == "brick wall"


def test_proposer_respects_exclusions_and_reports_no_change():
    ref = ImageAttributes(**FIXTURE_ATTRIBUTES)
    p = propose(ref, ref, "structured_v1", ref, excluded=set())
    assert p.edits == [] and "No rule fired" in p.rationale
    gen = ImageAttributes(subject="a red bicycle", secondary_objects=["brick wall", "potted plant", "dog"])
    first = propose(ref, ref, "structured_v1", gen, excluded=set())
    sigs = {e.signature() for e in first.edits}
    second = propose(ref, ref, "structured_v1", gen, excluded=sigs)
    assert not sigs & {e.signature() for e in second.edits}


# -------------------------------------------------------------------------------------- refinement API
def _drive_generation_and_score(client, token, score):
    """Simulated worker: complete generate job, then evaluation jobs with FIXTURE optimisation score."""
    job = client.post("/api/worker/jobs/claim", json={"types": ["generate"]}, headers=W(token)).json()
    worker_complete_generate(client, token, job, seed_offset=int(score * 1000))
    while True:
        r = client.post("/api/worker/jobs/claim", json={"types": ["evaluate"]}, headers=W(token))
        if r.status_code == 204:
            break
        ev = r.json()
        res = []
        for m in ev["payload"]["metrics"]:
            if m["metric"] == "clip_b32":
                res.append({"metric_result_id": m["metric_result_id"], "status": "completed", "value": score,
                            "backend": "FIXTURE"})
            else:
                res.append({"metric_result_id": m["metric_result_id"], "status": "unavailable", "detail": "test"})
        assert client.post(f"/api/worker/jobs/{ev['id']}/complete", data={"result": json.dumps({"metrics": res})},
                           headers=W(token)).status_code == 200


def test_refinement_session_lifecycle(client, recon, worker_token):
    rid = recon["id"]
    _, pvs = manual_prompts(client, rid)
    s = client.post(f"/api/reconstructions/{rid}/refinement", json={
        "base_prompt_version_id": pvs[1]["id"], "max_iterations": 3, "patience": 1, "min_delta": 0.01,
        "analysis_provider": None, "params": {"model_key": "sd_turbo", "seed": 3}}).json()
    assert s["status"] == "active" and s["iterations"][0]["status"] == "generating"
    assert client.post(f"/api/refinement/{s['id']}/propose", json={}).status_code == 422  # baseline not scored

    _drive_generation_and_score(client, worker_token, 0.60)
    s = client.get(f"/api/refinement/{s['id']}").json()
    assert s["iterations"][0]["status"] == "scored" and s["iterations"][0]["optimization_score"] == 0.60
    assert s["best_iteration_id"] == s["iterations"][0]["id"]

    # Optimisation and evaluation metrics are stored separately.
    gen = client.get(f"/api/generations/{s['iterations'][0]['generation_id']}").json()
    roles = {(m["metric"], m["role"]) for m in gen["metrics"]}
    assert ("clip_b32", "optimization") in roles and ("clip_l14", "evaluation") in roles
    assert ("clip_b32", "evaluation") not in roles

    s = client.post(f"/api/refinement/{s['id']}/propose", json={"max_edits": 2}).json()
    prop = s["iterations"][-1]
    assert prop["status"] == "proposed"
    # Prompt already matches the reference analysis and no generated-image analysis exists: the proposer
    # must say so rather than invent edits, and an empty proposal cannot be approved.
    assert prop["proposal"]["edits"] == [] and "No rule fired" in prop["proposal"]["rationale"]
    assert client.post(f"/api/refinement/iterations/{prop['id']}/approve", json={}).status_code == 422
    assert client.post(f"/api/refinement/iterations/{prop['id']}/reject", json={}).status_code == 200


def test_refinement_approve_reject_and_early_stop(client, recon, worker_token):
    rid = recon["id"]
    # Base prompt deliberately lacking lighting so the structural rule proposes restoring it.
    _, pvs = manual_prompts(client, rid)
    attrs = dict(FIXTURE_ATTRIBUTES, lighting="", camera_angle="", background="")
    base = client.post(f"/api/reconstructions/{rid}/prompts", json={"parent_id": pvs[1]["id"], "attributes": attrs,
                                                                     "note": "ablated"}).json()
    s = client.post(f"/api/reconstructions/{rid}/refinement", json={
        "base_prompt_version_id": base["id"], "max_iterations": 5, "patience": 1, "min_delta": 0.01,
        "analysis_provider": None, "params": {"model_key": "sd_turbo"}}).json()
    _drive_generation_and_score(client, worker_token, 0.50)

    s = client.post(f"/api/refinement/{s['id']}/propose", json={"max_edits": 1}).json()
    p1 = s["iterations"][-1]
    assert p1["proposal"]["edits"][0]["op"] == "replace"
    s = client.post(f"/api/refinement/iterations/{p1['id']}/reject", json={"reason": "test"}).json()
    assert s["iterations"][-1]["status"] == "rejected"

    s = client.post(f"/api/refinement/{s['id']}/propose", json={"max_edits": 1}).json()
    p2 = s["iterations"][-1]
    assert p2["proposal"]["edits"][0] != p1["proposal"]["edits"][0]  # rejected edit not re-proposed
    s = client.post(f"/api/refinement/iterations/{p2['id']}/approve", json={}).json()
    assert s["iterations"][-1]["status"] == "generating"
    _drive_generation_and_score(client, worker_token, 0.58)
    s = client.get(f"/api/refinement/{s['id']}").json()
    assert s["best_iteration_id"] == p2["id"] and s["status"] == "active"
    new_pv = client.get(f"/api/reconstructions/{rid}").json()["prompt_versions"][-1]
    assert new_pv["source"] == "refinement" and new_pv["parent_id"] == base["id"]

    s = client.post(f"/api/refinement/{s['id']}/propose", json={"max_edits": 1}).json()
    p3 = s["iterations"][-1]
    client.post(f"/api/refinement/iterations/{p3['id']}/approve", json={})
    _drive_generation_and_score(client, worker_token, 0.575)  # no improvement > min_delta → patience 1 → stop
    s = client.get(f"/api/refinement/{s['id']}").json()
    assert s["status"] == "stopped_early" and s["best_iteration_id"] == p2["id"]
    assert client.post(f"/api/refinement/{s['id']}/propose", json={}).status_code == 422

    comp = client.get(f"/api/reconstructions/{rid}/comparison").json()
    assert comp["conditions"]["C_refined"]["id"] == s["iterations"][2]["generation_id"]
    assert comp["refinement"]["best_iteration"] == p2["iteration"]


# ---------------------------------------------------------------------------------------- experiments
def test_experiment_export_and_reports_never_overwrite(client, worker_token):
    from app.config.settings import get_settings

    e = client.post("/api/experiments", json={"name": "Caption vs structured", "hypothesis": "B > A"}).json()
    assert e["code_fingerprint"]["source_sha256"]
    r = client.post("/api/reconstructions", files={"file": ("ref.png", fixture_png(seed=5), "image/png")},
                    data={"name": "exp recon", "experiment_id": e["id"]}).json()
    _, pvs = manual_prompts(client, r["id"])
    cap = client.post(f"/api/reconstructions/{r['id']}/analyses/manual",
                      json={"caption": "a red bicycle by a brick wall", "annotator": "test"}).json()
    cap_pv = client.post(f"/api/reconstructions/{r['id']}/prompts/from-analysis",
                         json={"analysis_id": cap["id"]}).json()[0]
    assert cap_pv["source"] == "caption" and cap_pv["template_id"] == "caption_v1"
    gens = client.post(f"/api/reconstructions/{r['id']}/comparison", json={
        "caption_prompt_version_id": cap_pv["id"], "structured_prompt_version_id": pvs[1]["id"],
        "params": {"model_key": "sd15", "seed": 11}}).json()
    assert [g["condition"] for g in gens] == ["A_caption", "B_structured"]
    assert gens[0]["params"]["seed"] == gens[1]["params"]["seed"] == 11
    for _ in gens:
        job = client.post("/api/worker/jobs/claim", json={"types": ["generate"]}, headers=W(worker_token)).json()
        worker_complete_generate(client, worker_token, job)

    rec = client.get(f"/api/experiments/{e['id']}/export.json").json()
    assert rec["schema"] == "pe.experiment.v1"
    g0 = rec["reconstructions"][0]["generations"][0]
    assert {"seed", "params", "model_id", "metrics", "duration_s"} <= g0.keys()
    rows = list(csv.DictReader(io.StringIO(client.get(f"/api/experiments/{e['id']}/export.csv").text)))
    assert rows and {"A_caption", "B_structured"} <= {row["condition"] for row in rows}
    ssim_rows = [row for row in rows if row["metric"] == "ssim"]
    assert ssim_rows and all(row["metric_status"] == "completed" for row in ssim_rows)
    clip_rows = [row for row in rows if row["metric"] == "clip_l14"]
    assert all(row["metric_value"] == "" for row in clip_rows)  # pending → empty, never invented

    a = client.post(f"/api/experiments/{e['id']}/reports").json()
    b = client.post(f"/api/experiments/{e['id']}/reports").json()
    assert a["json"] != b["json"]
    root = get_settings().experiments_dir
    assert (root / a["json"]).exists() and (root / b["json"]).exists()
    names = {x["name"] for x in client.get("/api/reports").json()}
    assert a["json"].split("/")[-1] in names

    bundle = client.post(f"/api/reconstructions/{r['id']}/demo-bundle").json()["bundle"]
    bid = bundle.split("/")[-1]
    m = client.get(f"/api/demo/bundles/{bid}").json()
    assert m["conditions"]["C_refined"]["status"] == "not_executed"
    assert m["conditions"]["A_caption"]["status"] == "completed"
    assert client.get(f"/api/demo/bundles/{bid}/files/A_caption.jpg").status_code == 200
    assert client.get(f"/api/demo/bundles/{bid}/files/..%2Fmanifest.json").status_code in (400, 404)


# ---------------------------------------------------------------------------------------------- metrics
def test_ssim_properties():
    from app.evaluation.metrics import ssim_gray

    rng = np.random.default_rng(0)
    x = rng.integers(0, 255, (64, 64)).astype(np.float64)
    assert ssim_gray(x, x) == pytest.approx(1.0)
    noisy = np.clip(x + rng.normal(0, 40, x.shape), 0, 255)
    assert 0 < ssim_gray(x, noisy) < 0.9
    assert ssim_gray(x, 255 - x) < 0


def test_ssim_matches_reference_formula_on_constant_images():
    from app.evaluation.metrics import ssim_gray

    a, b = np.full((32, 32), 100.0), np.full((32, 32), 150.0)
    c1 = (0.01 * 255) ** 2
    expected = (2 * 100 * 150 + c1) / (100**2 + 150**2 + c1)  # σ terms vanish for constant images
    assert ssim_gray(a, b) == pytest.approx(expected, rel=1e-9)


def test_torch_metrics_report_unavailable_without_torch():
    import importlib.util

    from app.evaluation.metrics import get_metric

    if importlib.util.find_spec("torch") is None:
        for k in ("clip_l14", "dinov2_base", "lpips_alex"):
            assert get_metric(k).availability().available is False


# ---------------------------------------------------------------------------------------- analysis parse
def test_structured_parse_handles_chatter_and_aliases():
    from app.analysis.base import parse_structured_output

    text = 'Sure! Here you go:\n{"main_subject": "a cat {sitting}", "colours": "orange, white", "style": "photo",' \
           ' "spatial_relationships": [{"subject": "cat", "relation": "on", "object": "sofa"}, "junk"]} thanks'
    a = parse_structured_output(text)
    assert a.subject == "a cat {sitting}" and a.colors == ["orange", "white"] and a.art_style == "photo"
    assert len(a.spatial_relationships) == 1


def test_openai_compatible_provider_request_shape(monkeypatch):
    from PIL import Image as PILImage

    from app.analysis.providers import OpenAICompatibleProvider

    captured = {}

    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "A gradient test pattern."}}]}

    def fake_post(url, json, headers, timeout):  # noqa: A002
        captured.update(url=url, body=json, headers=headers)
        return Resp()

    monkeypatch.setattr("app.analysis.providers.httpx.post", fake_post)
    p = OpenAICompatibleProvider(base_url="http://vlm.local/v1", api_key="k", model="qwen3-vl-2b")
    out = p.analyze(PILImage.new("RGB", (32, 32)), "caption")
    assert captured["url"] == "http://vlm.local/v1/chat/completions"
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert captured["headers"]["Authorization"] == "Bearer k"
    assert out.caption == "A gradient test pattern."


# ---------------------------------------------------------------------------------- worker agent protocol
def test_agent_executes_job_end_to_end_with_stub_handler(client, recon, worker_token, monkeypatch):
    """Runs the REAL agent code (WorkerClient/execute) against the API, with a stub handler standing in for
    diffusers (the stub returns a FIXTURE image)."""
    from app.workers import agent, handlers

    rid = recon["id"]
    _, pvs = manual_prompts(client, rid)
    g = client.post(f"/api/reconstructions/{rid}/generations",
                    json={"prompt_version_id": pvs[1]["id"], "params": {"model_key": "sd_turbo"}}).json()

    def stub_generate(payload, fetch, progress, mm):
        progress(0.5, "stub")
        return handlers.HandlerOutput({"duration_s": 0.0, "runtime_info": {"fixture": True}},
                                      fixture_png(payload["width"], payload["height"]))

    monkeypatch.setitem(agent.HANDLERS, "generate", stub_generate)
    wc = agent.WorkerClient.__new__(agent.WorkerClient)
    wc.server = "http://testserver"
    client.headers.update(W(worker_token))
    wc.http = client
    job = wc.claim(["generate"])
    agent.execute(wc, job, handlers.ModelManager())
    gen = client.get(f"/api/generations/{g['id']}").json()
    assert gen["status"] == "completed" and gen["runtime_info"]["fixture"] is True
