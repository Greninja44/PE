"""GPU worker agent — run on the RTX 5060 machine.

    python -m app.workers.agent doctor
    python -m app.workers.agent register --server http://HOST:8000 --admin-token XXX --name rtx5060
    python -m app.workers.agent run --server http://HOST:8000            # token from env or token file

The agent polls the backend, claims one job at a time, downloads only the images that job references,
runs the fixed handler for the job type, and uploads results. A background heartbeat extends the lease
during long operations (first-time model downloads); if the agent dies, the lease expires and the
backend re-queues the job.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image as PILImage

from app.workers.handlers import HANDLERS, ModelManager, runtime_snapshot
from app.workers.schemas import JOB_TYPES

log = logging.getLogger("pe.worker")
TOKEN_FILE = Path(os.environ.get("PE_WORKER_TOKEN_FILE", Path.home() / ".config" / "pe-worker" / "token"))
MAX_IMAGE_BYTES = 25 * 1024 * 1024


class JobCancelled(Exception):
    pass


class WorkerClient:
    def __init__(self, server: str, token: str, timeout: float = 60.0) -> None:
        self.server = server.rstrip("/")
        self.http = httpx.Client(base_url=self.server, headers={"Authorization": f"Bearer {token}"}, timeout=timeout)

    def heartbeat(self, capabilities: dict[str, Any]) -> dict[str, Any]:
        r = self.http.post("/api/worker/heartbeat", json={"capabilities": capabilities})
        r.raise_for_status()
        return r.json()

    def claim(self, types: list[str]) -> dict[str, Any] | None:
        r = self.http.post("/api/worker/jobs/claim", json={"types": types})
        if r.status_code == 204:
            return None
        r.raise_for_status()
        return r.json()

    def progress(self, job_id: str, progress: float, message: str) -> bool:
        """Returns True if the job was cancelled server-side."""
        r = self.http.post(f"/api/worker/jobs/{job_id}/progress", json={"progress": progress, "message": message})
        if r.status_code == 409:
            return True
        r.raise_for_status()
        return bool(r.json().get("cancel"))

    def fetch_image(self, job_id: str, image_id: str) -> PILImage.Image:
        with self.http.stream("GET", f"/api/worker/jobs/{job_id}/images/{image_id}") as r:
            r.raise_for_status()
            buf = io.BytesIO()
            for chunk in r.iter_bytes():
                buf.write(chunk)
                if buf.tell() > MAX_IMAGE_BYTES:
                    raise ValueError("image download exceeds size limit")
        return PILImage.open(io.BytesIO(buf.getvalue())).convert("RGB")

    def complete(self, job_id: str, result: dict[str, Any], image_png: bytes | None) -> None:
        files = {"image": ("output.png", image_png, "image/png")} if image_png else None
        r = self.http.post(f"/api/worker/jobs/{job_id}/complete", data={"result": json.dumps(result)}, files=files,
                           timeout=120)
        if r.status_code >= 400:
            raise RuntimeError(f"complete rejected ({r.status_code}): {r.text[:500]}")

    def fail(self, job_id: str, error: str, retryable: bool) -> None:
        r = self.http.post(f"/api/worker/jobs/{job_id}/fail", json={"error": error[:4000], "retryable": retryable})
        if r.status_code >= 400:
            log.warning("fail report rejected: %s", r.text[:300])


class Heartbeat(threading.Thread):
    """Extends the lease every ``interval`` seconds while a job runs and notices cancellation."""

    def __init__(self, client: WorkerClient, job_id: str, interval: float = 30.0) -> None:
        super().__init__(daemon=True)
        self.client, self.job_id, self.interval = client, job_id, interval
        self.state = (0.0, "Starting")
        self.cancelled = threading.Event()
        self._stop = threading.Event()

    def update(self, progress: float, message: str) -> None:
        self.state = (progress, message)
        if self.cancelled.is_set():
            raise JobCancelled()

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                if self.client.progress(self.job_id, *self.state):
                    self.cancelled.set()
            except httpx.HTTPError as e:
                log.warning("heartbeat failed: %s", e)

    def stop(self) -> None:
        self._stop.set()


def capabilities() -> dict[str, Any]:
    from importlib.util import find_spec

    return {"job_types": list(JOB_TYPES), "runtime": runtime_snapshot(),
            "packages": {m: find_spec(m) is not None for m in ("torch", "diffusers", "transformers", "lpips")}}


def run(server: str, token: str, types: list[str], poll: float, once: bool) -> int:
    client = WorkerClient(server, token)
    caps = capabilities()
    info = client.heartbeat(caps)
    log.info("connected as %s (%s); runtime=%s", info["name"], info["worker_id"], caps["runtime"])
    mm = ModelManager()
    stopping = {"flag": False}

    def _sig(*_: Any) -> None:
        log.info("shutdown requested; finishing current job")
        stopping["flag"] = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    last_hb = time.monotonic()
    while not stopping["flag"]:
        try:
            if time.monotonic() - last_hb > 30:
                client.heartbeat({})
                last_hb = time.monotonic()
            job = client.claim(types)
        except httpx.HTTPError as e:
            log.warning("server unreachable (%s); retrying in %ss", e, poll * 3)
            time.sleep(poll * 3)
            continue
        if job is None:
            if once:
                return 0
            time.sleep(poll)
            continue
        execute(client, job, mm)
        if once:
            return 0
    mm.unload_all()
    return 0


def execute(client: WorkerClient, job: dict[str, Any], mm: ModelManager) -> None:
    jid, jtype = job["id"], job["type"]
    log.info("job %s (%s) attempt %s", jid, jtype, job["attempts"])
    handler = HANDLERS.get(jtype)
    if handler is None:
        client.fail(jid, f"worker has no handler for '{jtype}'", retryable=False)
        return
    hb = Heartbeat(client, jid)
    hb.start()
    try:
        out = handler(job["payload"], lambda image_id: client.fetch_image(jid, image_id), hb.update, mm)
        hb.update(0.99, "Uploading results")
        client.complete(jid, out.result, out.image_png)
        log.info("job %s done", jid)
    except JobCancelled:
        log.info("job %s cancelled by server", jid)
    except (ValueError, KeyError) as e:
        log.exception("job %s rejected", jid)
        client.fail(jid, f"{type(e).__name__}: {e}", retryable=False)
    except Exception as e:  # noqa: BLE001
        log.exception("job %s failed", jid)
        oom = "out of memory" in str(e).lower()
        if oom:
            mm.unload_all()
        client.fail(jid, f"{type(e).__name__}: {e}", retryable=True)
    finally:
        hb.stop()


def doctor() -> int:
    snap = runtime_snapshot()
    print(json.dumps(capabilities(), indent=2))
    if not snap.get("torch"):
        print("\n✗ torch not installed. See docs/RTX5060_SETUP.md")
        return 1
    if not snap.get("cuda_available"):
        print("\n✗ CUDA not available to torch. Install a CUDA 12.8+ build (cu128) for RTX 50-series.")
        return 1
    if snap.get("capability", "").startswith("12") and not snap.get("cuda", "0").startswith(("12.8", "12.9", "13")):
        print("\n⚠ Blackwell GPU (sm_120) needs a torch build with CUDA ≥ 12.8.")
    print("\n✓ GPU ready:", snap.get("gpu"))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="python -m app.workers.agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check torch/CUDA/GPU")
    reg = sub.add_parser("register", help="register this machine and store its token")
    reg.add_argument("--server", required=True)
    reg.add_argument("--admin-token", default=os.environ.get("PE_ADMIN_TOKEN", ""))
    reg.add_argument("--name", default=os.uname().nodename if hasattr(os, "uname") else "gpu-worker")
    r = sub.add_parser("run", help="poll for and execute jobs")
    r.add_argument("--server", required=True)
    r.add_argument("--token", default=os.environ.get("PE_WORKER_TOKEN", ""))
    r.add_argument("--types", default=",".join(JOB_TYPES))
    r.add_argument("--poll", type=float, default=3.0)
    r.add_argument("--once", action="store_true", help="process at most one job then exit")
    a = ap.parse_args(argv)

    if a.cmd == "doctor":
        return doctor()
    if a.cmd == "register":
        resp = httpx.post(f"{a.server.rstrip('/')}/api/worker/register", headers={"X-Admin-Token": a.admin_token},
                          json={"name": a.name, "capabilities": capabilities()}, timeout=30)
        if resp.status_code != 201:
            print(f"registration failed ({resp.status_code}): {resp.text}", file=sys.stderr)
            return 1
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(resp.json()["token"])
        TOKEN_FILE.chmod(0o600)
        print(f"registered worker {resp.json()['worker_id']}; token saved to {TOKEN_FILE}")
        return 0
    token = a.token or (TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else "")
    if not token:
        print("no token: run `register` first or set PE_WORKER_TOKEN", file=sys.stderr)
        return 1
    types = [t for t in a.types.split(",") if t in JOB_TYPES]
    return run(a.server, token, types, a.poll, a.once)


if __name__ == "__main__":
    sys.exit(main())
