"""Condition comparison (A caption / D detailed / B structured / C refinement views), paired statistics, human
ratings and the additive DB migration. All images and scores here are FIXTURES driven through a simulated worker."""
from __future__ import annotations

import json

from tests.conftest import FIXTURE_ATTRIBUTES
from tests.test_api_flow import W, manual_prompts, worker_complete_generate
from tests.test_refinement_experiments import _drive_generation_and_score


def _complete_eval(client, token, values: dict[str, float]):
    """Simulated worker: answer every queued evaluate job with FIXTURE values (others unavailable)."""
    while True:
        r = client.post("/api/worker/jobs/claim", json={"types": ["evaluate"]}, headers=W(token))
        if r.status_code == 204:
            return
        ev = r.json()
        res = [{"metric_result_id": m["metric_result_id"], "status": "completed", "value": values[m["metric"]],
                "backend": "FIXTURE"} if m["metric"] in values else
               {"metric_result_id": m["metric_result_id"], "status": "unavailable", "detail": "test"}
               for m in ev["payload"]["metrics"]]
        assert client.post(f"/api/worker/jobs/{ev['id']}/complete", data={"result": json.dumps({"metrics": res})},
                           headers=W(token)).status_code == 200


def _setup_three_conditions(client, rid):
    a = client.post(f"/api/reconstructions/{rid}/analyses/manual",
                    json={"caption": "a red bicycle against a brick wall", "annotator": "test"}).json()
    cap = client.post(f"/api/reconstructions/{rid}/prompts/from-analysis",
                      json={"analysis_id": a["id"], "template_id": "caption_v1"}).json()[0]
    sa, structured = manual_prompts(client, rid)
    detailed = client.post(f"/api/reconstructions/{rid}/prompts/from-analysis",
                           json={"analysis_id": sa["id"], "template_id": "natural_v1"}).json()
    return cap, detailed[-1], structured[-1]


def test_three_condition_comparison_table_and_paired_stats(client, recon, worker_token):
    rid = recon["id"]
    cap, det, st = _setup_three_conditions(client, rid)
    assert det["template_id"] == "natural_v1" and det["positive"].endswith(".")
    gens = client.post(f"/api/reconstructions/{rid}/comparison", json={
        "caption_prompt_version_id": cap["id"], "detailed_prompt_version_id": det["id"],
        "structured_prompt_version_id": st["id"], "params": {"model_key": "sd_turbo", "seed": 7}}).json()
    assert [g["condition"] for g in gens] == ["A_caption", "D_detailed", "B_structured"]
    assert len({json.dumps({k: v for k, v in g["params"].items() if k != "negative_prompt"}) for g in gens}) == 1

    # FIXTURE scores: caption beats structured on CLIP-L, structured beats caption on LPIPS (lower is better).
    fixture = {"A_caption": {"clip_l14": 0.80, "lpips_alex": 0.60}, "D_detailed": {"clip_l14": 0.78, "lpips_alex": 0.55},
               "B_structured": {"clip_l14": 0.75, "lpips_alex": 0.50}}
    for _ in gens:
        job = client.post("/api/worker/jobs/claim", json={"types": ["generate"]}, headers=W(worker_token)).json()
        cond = next(g["condition"] for g in gens if g["id"] == job["payload"]["generation_id"])
        worker_complete_generate(client, worker_token, job)
        _complete_eval(client, worker_token, fixture[cond])

    table = client.get(f"/api/reconstructions/{rid}/comparison-table").json()
    assert len(table["rows"]) == 3 and len(table["config_groups"]) == 1  # all directly comparable
    row = next(r for r in table["rows"] if r["condition"] == "B_structured")
    assert row["prompt_words"] > 0 and row["prompt_clip_tokens_approx"] > 0
    assert row["metrics"]["clip_l14"]["value"] == 0.75 and row["metrics"]["lpips_alex"]["higher_is_better"] is False
    assert row["metrics"]["dinov2_base"]["value"] is None  # unavailable stays None, never filled in

    agg = client.get("/api/stats").json()["comparison"]
    assert set(agg["groups"]) == {"A_caption", "D_detailed", "B_structured"}
    pair = {(p["a"], p["b"], p["metric"]): p for p in agg["paired"]}
    clip = pair[("A_caption", "B_structured", "clip_l14")]
    assert clip["n_pairs"] == 1 and clip["b_worse"] == 1 and clip["b_better"] == 0  # structured lost on CLIP
    lp = pair[("A_caption", "B_structured", "lpips_alex")]
    assert lp["b_better"] == 1 and abs(lp["mean_diff_b_minus_a"] + 0.10) < 1e-9  # direction respected


def test_refinement_views_are_separate_and_best_is_flagged(client, recon, worker_token):
    rid = recon["id"]
    _, pvs = manual_prompts(client, rid)
    attrs = dict(FIXTURE_ATTRIBUTES, lighting="", camera_angle="", background="")
    base = client.post(f"/api/reconstructions/{rid}/prompts", json={"parent_id": pvs[1]["id"], "attributes": attrs,
                                                                     "note": "ablated"}).json()
    s = client.post(f"/api/reconstructions/{rid}/refinement", json={
        "base_prompt_version_id": base["id"], "max_iterations": 5, "patience": 1, "min_delta": 0.01,
        "analysis_provider": None, "params": {"model_key": "sd_turbo"}}).json()
    _drive_generation_and_score(client, worker_token, 0.50)
    for score in (0.58, 0.55):  # iteration 1 is best; iteration 2 (final) is worse
        s = client.post(f"/api/refinement/{s['id']}/propose", json={"max_edits": 1}).json()
        client.post(f"/api/refinement/iterations/{s['iterations'][-1]['id']}/approve", json={})
        _drive_generation_and_score(client, worker_token, score)

    table = client.get(f"/api/reconstructions/{rid}/comparison-table").json()
    groups = {r["refinement_iteration"]: r["groups"] for r in table["rows"]}
    assert groups[0] == ["C_iter0"] and groups[1] == ["C_refined_best"] and groups[2] == ["C_refined_final"]
    assert all(r["refinement_iterations_executed"] == 3 for r in table["rows"])
    agg = client.get("/api/stats").json()["comparison"]
    assert agg["groups"]["C_refined_best"]["selection_biased"] is True
    assert agg["groups"]["C_refined_final"]["selection_biased"] is False


def test_human_ratings_are_separate_and_validated(client, recon, worker_token):
    rid = recon["id"]
    _, pvs = manual_prompts(client, rid)
    g = client.post(f"/api/reconstructions/{rid}/generations",
                    json={"prompt_version_id": pvs[1]["id"], "params": {"model_key": "sd_turbo"}}).json()
    assert client.post(f"/api/generations/{g['id']}/ratings", json={"rater": "r1", "similarity": 3}).status_code == 422
    job = client.post("/api/worker/jobs/claim", json={"types": ["generate"]}, headers=W(worker_token)).json()
    worker_complete_generate(client, worker_token, job)
    assert client.post(f"/api/generations/{g['id']}/ratings", json={"rater": "r1", "similarity": 6}).status_code == 422
    r = client.post(f"/api/generations/{g['id']}/ratings",
                    json={"rater": "r1", "similarity": 4, "blind": True, "notes": "colours close"})
    assert r.status_code == 201 and r.json()["kind"] == "human"
    assert len(client.get(f"/api/generations/{g['id']}/ratings").json()) == 1
    gen = client.get(f"/api/generations/{g['id']}").json()
    assert all(m["backend"] != "human" for m in gen["metrics"])  # never mixed into metric rows
    row = client.get(f"/api/reconstructions/{rid}/comparison-table").json()["rows"][0]
    assert row["human"] == {"n": 1, "mean_similarity": 4.0}


def test_additive_migration_adds_runtime_info_to_old_database(tmp_path):
    import sqlite3

    from app.database import session

    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE analyses (id VARCHAR(32) PRIMARY KEY, reconstruction_id VARCHAR(32), "
                "image_id VARCHAR(32), mode VARCHAR(16), provider VARCHAR(64), model VARCHAR(255), "
                "status VARCHAR(16), result JSON, caption TEXT, raw_output TEXT, error TEXT, duration_s FLOAT, "
                "job_id VARCHAR(32), created_at DATETIME)")
    con.execute("INSERT INTO analyses (id, mode, provider, model, status) VALUES ('a1','caption','manual','h','completed')")
    con.commit()
    con.close()
    session.init_engine(f"sqlite:///{db}")
    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(analyses)")}
    assert "runtime_info" in cols
    assert con.execute("SELECT id, status FROM analyses").fetchall() == [("a1", "completed")]  # data preserved
    con.close()
