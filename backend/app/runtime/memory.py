"""Memory diagnostics for unified-memory (Apple Silicon) and discrete-GPU machines.

On Apple Silicon the CPU and GPU share one pool, so the meaningful per-process number is the macOS
*physical footprint* (what Activity Monitor shows as "Memory"), which includes Metal buffers. RSS alone
undercounts GPU allocations. We read the footprint through ``proc_pid_rusage`` and fall back to RSS
elsewhere. Every value is a measurement; nothing is estimated.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import threading
import time
from typing import Any

GB = 1024**3

_libc: Any = None
if sys.platform == "darwin":
    try:
        _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    except OSError:  # pragma: no cover
        _libc = None


def _phys_footprint(pid: int) -> int | None:
    """macOS: ri_phys_footprint from rusage_info_v2 (16-byte uuid, then uint64 fields; footprint is #7)."""
    if _libc is None:
        return None
    buf = (ctypes.c_uint64 * 64)()
    if _libc.proc_pid_rusage(pid, 2, ctypes.byref(buf)) != 0:
        return None
    return int(buf[2 + 7])  # uuid occupies two uint64 slots


def process_bytes(pid: int | None = None) -> tuple[int, str]:
    pid = pid or os.getpid()
    fp = _phys_footprint(pid)
    if fp:
        return fp, "phys_footprint"
    try:
        import psutil

        return psutil.Process(pid).memory_info().rss, "rss"
    except ImportError:  # pragma: no cover
        try:
            import resource  # Unix only
        except ImportError:
            return 0, "unavailable (install psutil)"
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "maxrss"


def snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {}
    b, kind = process_bytes()
    out["process_gb"] = round(b / GB, 2)
    out["process_metric"] = kind
    try:
        import psutil

        vm = psutil.virtual_memory()
        out["system_total_gb"] = round(vm.total / GB, 2)
        out["system_available_gb"] = round(vm.available / GB, 2)
        out["system_used_percent"] = vm.percent
        sw = psutil.swap_memory()
        out["swap_used_gb"] = round(sw.used / GB, 2)
    except ImportError:  # pragma: no cover
        pass
    if "torch" in sys.modules:  # only report accelerator memory if torch is already loaded
        import torch

        if torch.backends.mps.is_available():
            out["mps_allocated_gb"] = round(torch.mps.current_allocated_memory() / GB, 2)
            out["mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / GB, 2)
        if torch.cuda.is_available():
            out["cuda_allocated_gb"] = round(torch.cuda.memory_allocated() / GB, 2)
            out["cuda_peak_gb"] = round(torch.cuda.max_memory_allocated() / GB, 2)
    return out


def available_gb() -> float | None:
    try:
        import psutil
    except ImportError:  # pragma: no cover
        return None
    return psutil.virtual_memory().available / GB


class PeakSampler:
    """Context manager sampling process footprint (and MPS driver memory) to report the peak during a step.

    ``with PeakSampler() as p: ...; p.report()`` → ``{"peak_process_gb", "start_process_gb", ...}``."""

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_bytes = 0
        self.peak_bytes = 0
        self.peak_mps = 0
        self.kind = ""
        self.t0 = 0.0
        self.elapsed = 0.0

    def _sample(self) -> None:
        b, self.kind = process_bytes()
        self.peak_bytes = max(self.peak_bytes, b)
        if "torch" in sys.modules:
            import torch

            if torch.backends.mps.is_available():
                self.peak_mps = max(self.peak_mps, torch.mps.driver_allocated_memory())

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self) -> PeakSampler:
        self.start_bytes, self.kind = process_bytes()
        self.peak_bytes = self.start_bytes
        self.t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        self._sample()
        self.elapsed = time.perf_counter() - self.t0

    def report(self) -> dict[str, Any]:
        r = {"metric": self.kind, "start_process_gb": round(self.start_bytes / GB, 2),
             "peak_process_gb": round(self.peak_bytes / GB, 2), "sample_interval_s": self.interval}
        if self.peak_mps:
            r["peak_mps_driver_gb"] = round(self.peak_mps / GB, 2)
        return r
