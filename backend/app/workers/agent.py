"""Inference worker agent — runs on this Mac (Apple MPS) or on a separate NVIDIA machine.

    python -m app.workers.agent doctor                                   # verify torch + device with real ops
    python -m app.workers.agent local                                    # same-machine worker (auto-registers)
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
LOCAL_TOKEN_FILE = TOKEN_FILE.with_name("token-local")
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


def capabilities(mm: ModelManager | None = None) -> dict[str, Any]:
    from importlib.util import find_spec

    from app.runtime import memory

    caps: dict[str, Any] = {
        "job_types": list(JOB_TYPES), "runtime": runtime_snapshot(), "memory": memory.snapshot(),
        "packages": {m: find_spec(m) is not None for m in ("torch", "diffusers", "transformers", "lpips")},
        "reported_at": time.time(),
    }
    if mm is not None:
        caps["resident_models"] = mm.resident()
        caps["model_events"] = mm.events[-8:]
    return caps


def run(server: str, token: str, types: list[str], poll: float, once: bool) -> int:
    from app.config.settings import get_settings

    idle_unload = get_settings().worker_idle_unload_s
    client = WorkerClient(server, token)
    mm = ModelManager()
    caps = capabilities(mm)
    info = client.heartbeat(caps)
    log.info("connected as %s (%s); device=%s", info["name"], info["worker_id"],
             caps["runtime"].get("selected_device"))
    stopping = {"flag": False}

    def _sig(*_: Any) -> None:
        log.info("shutdown requested; finishing current job")
        stopping["flag"] = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    last_hb = time.monotonic()
    while not stopping["flag"]:
        try:
            if time.monotonic() - last_hb > 15:
                client.heartbeat(capabilities(mm))  # live memory + resident models for the dashboard
                last_hb = time.monotonic()
            job = client.claim(types)
        except httpx.HTTPError as e:
            log.warning("server unreachable (%s); retrying in %ss", e, poll * 3)
            time.sleep(poll * 3)
            continue
        if job is None:
            if once:
                return 0
            if mm.unload_if_idle(idle_unload):
                last_hb = 0.0  # report the freed memory promptly
            time.sleep(poll)
            continue
        execute(client, job, mm)
        last_hb = 0.0
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
        from app.runtime.device import is_oom
        from app.runtime.lifecycle import InsufficientMemory

        if isinstance(e, InsufficientMemory):
            client.fail(jid, f"{type(e).__name__}: {e}", retryable=False)
            return
        if is_oom(e):
            mm.unload_all(reason="out of memory")
            msg = (f"{type(e).__name__}: {e}\nOut of memory: models were unloaded. Try a lower resolution, close "
                   "other applications, or set PE_DEVICE=cpu.")
            client.fail(jid, msg, retryable=True)
            return
        client.fail(jid, f"{type(e).__name__}: {e}", retryable=True)
    finally:
        hb.stop()


def doctor() -> int:
    """Check the installation by running real tensor operations on the selected device (an import alone proves
    nothing)."""
    snap = runtime_snapshot()
    print(json.dumps(capabilities(), indent=2, default=str))
    if not snap.get("torch"):
        print("\n✗ torch not installed. macOS: scripts/setup_mac.sh · NVIDIA: docs/RTX5060_SETUP.md")
        return 1
    import torch

    from app.runtime.device import select_device, synchronize

    dev = select_device().device
    try:
        dt = torch.float16 if dev != "cpu" else torch.float32
        x = torch.randn(512, 512, device=dev, dtype=dt)
        y = torch.nn.functional.conv2d(x[None, None], torch.ones(1, 1, 3, 3, device=dev, dtype=dt))
        val = float((x @ x).float().mean().item()) + float(y.float().mean().item())
        synchronize(dev)
        assert val == val, "NaN"
    except Exception as e:  # noqa: BLE001
        print(f"\n✗ tensor ops failed on {dev}: {type(e).__name__}: {e}")
        return 1
    if dev == "cuda":
        if snap.get("capability", "").startswith("12") and not snap.get("cuda", "0").startswith(("12.8", "12.9", "13")):
            print("\n⚠ Blackwell GPU (sm_120) needs a torch build with CUDA ≥ 12.8.")
        print("\n✓ CUDA ready:", snap.get("gpu"))
    elif dev == "mps":
        print(f"\n✓ Apple MPS ready: {snap.get('gpu')} · recommended max working set "
              f"{snap.get('mps_recommended_max_gb')} GB · CPU fallback for missing ops: "
              f"{'on' if snap.get('mps_fallback_enabled') else 'off'}")
    else:
        print("\n⚠ No GPU backend: running on CPU (functional, slow).")
    return 0


def _register(server: str, admin_token: str, name: str, token_file: Path) -> str:
    resp = httpx.post(f"{server.rstrip('/')}/api/worker/register", headers={"X-Admin-Token": admin_token},
                      json={"name": name, "capabilities": capabilities()}, timeout=30)
    if resp.status_code != 201:
        raise RuntimeError(f"registration failed ({resp.status_code}): {resp.text}")
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(resp.json()["token"])
    token_file.chmod(0o600)
    return resp.json()["token"]


def local(server: str, types: list[str], poll: float, wait_s: float = 60.0) -> int:
    """Same-machine worker: waits for the API, registers with the admin token from backend/.env if needed
    (re-registering if the local database was reset), then runs."""
    from app.config.settings import get_settings

    s = get_settings()
    deadline = time.monotonic() + wait_s
    while True:
        try:
            httpx.get(f"{server}/api/health", timeout=3).raise_for_status()
            break
        except httpx.HTTPError:
            if time.monotonic() > deadline:
                print(f"API not reachable at {server}", file=sys.stderr)
                return 1
            time.sleep(1)
    token = LOCAL_TOKEN_FILE.read_text().strip() if LOCAL_TOKEN_FILE.exists() else ""
    if token:
        r = httpx.post(f"{server}/api/worker/heartbeat", json={"capabilities": {}},
                       headers={"Authorization": f"Bearer {token}"}, timeout=10)
        if r.status_code == 401:
            token = ""
    if not token:
        if not s.admin_token:
            print("PE_ADMIN_TOKEN is empty in backend/.env — run scripts/setup_mac.sh", file=sys.stderr)
            return 1
        token = _register(server, s.admin_token, s.local_worker_name, LOCAL_TOKEN_FILE)
        log.info("registered local worker '%s'", s.local_worker_name)
    return run(server, token, types, poll, once=False)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # do not log every 3-second poll
    ap = argparse.ArgumentParser(prog="python -m app.workers.agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check torch and the selected device (MPS/CUDA/CPU) with real tensor ops")
    lo = sub.add_parser("local", help="run a worker for the API on this machine (auto-registers)")
    lo.add_argument("--server", default="http://127.0.0.1:8000")
    lo.add_argument("--types", default=",".join(JOB_TYPES))
    lo.add_argument("--poll", type=float, default=1.0)
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
    if a.cmd == "local":
        return local(a.server.rstrip("/"), [t for t in a.types.split(",") if t in JOB_TYPES], a.poll)
    if a.cmd == "register":
        try:
            _register(a.server, a.admin_token, a.name, TOKEN_FILE)
        except RuntimeError as e:
            print(e, file=sys.stderr)
            return 1
        print(f"registered worker; token saved to {TOKEN_FILE}")
        return 0
    token = a.token or (TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else "")
    if not token:
        print("no token: run `register` first or set PE_WORKER_TOKEN", file=sys.stderr)
        return 1
    types = [t for t in a.types.split(",") if t in JOB_TYPES]
    return run(a.server, token, types, a.poll, a.once)


if __name__ == "__main__":
    sys.exit(main())
