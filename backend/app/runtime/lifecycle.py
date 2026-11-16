"""Sequential model lifecycle for memory-constrained machines (16 GB unified memory).

Rules
-----
* At most one *heavy* family is resident: a VLM **or** a diffusion pipeline **or** the metric models.
  Loading a different family unloads everything first. Small metric models may co-reside with each other.
* Every unload drops references, runs the garbage collector and releases the MPS/CUDA allocator cache.
* Before loading, free memory is compared with the model's documented footprint. By default a shortfall is
  logged and recorded (macOS can reclaim cache and compress memory); ``PE_STRICT_MEMORY_CHECK=true``
  refuses the load instead.
* Models idle for ``PE_WORKER_IDLE_UNLOAD_S`` seconds are unloaded so the Mac stays usable between runs.
"""
from __future__ import annotations

import gc
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.runtime import memory
from app.runtime.device import empty_cache

log = logging.getLogger("pe.runtime")


class InsufficientMemory(RuntimeError):
    pass


@dataclass
class Resident:
    category: str
    key: str
    obj: Any
    est_gb: float | None
    loaded_at: float
    load_s: float
    last_used: float = field(default_factory=time.monotonic)
    footprint_after_load_gb: float | None = None


class ModelManager:
    """Keeps at most one model *category* resident (vlm | diffusion | metrics)."""

    def __init__(self, strict_memory: bool | None = None, margin_gb: float | None = None) -> None:
        from app.config.settings import get_settings

        s = get_settings()
        self.strict = s.strict_memory_check if strict_memory is None else strict_memory
        self.margin_gb = s.memory_margin_gb if margin_gb is None else margin_gb
        self._cat: str | None = None
        self._models: dict[str, Resident] = {}
        self.events: list[dict[str, Any]] = []  # bounded log of load/unload events for diagnostics

    # ------------------------------------------------------------------------------------------------ api
    def get(self, category: str, key: str, factory: Callable[[], Any], est_gb: float | None = None) -> Any:
        full = f"{category}:{key}"
        if full in self._models:
            r = self._models[full]
            r.last_used = time.monotonic()
            return r.obj
        # Metric models are small and may co-reside; a VLM or diffusion pipeline always gets memory alone.
        if self._cat != category or category != "metrics":
            self.unload_all(reason=f"making room for {full}")
        self._check_memory(full, est_gb)
        log.info("loading %s", full)
        t0 = time.perf_counter()
        obj = factory()
        load_s = time.perf_counter() - t0
        self._cat = category
        fp = memory.snapshot().get("process_gb")
        self._models[full] = Resident(category, key, obj, est_gb, time.time(), round(load_s, 2),
                                      footprint_after_load_gb=fp)
        self._event("load", full, load_s=round(load_s, 2), process_gb=fp)
        return obj

    def resident(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        return [{"key": k, "category": r.category, "est_gb": r.est_gb, "load_s": r.load_s,
                 "idle_s": round(now - r.last_used, 1), "process_gb_after_load": r.footprint_after_load_gb}
                for k, r in self._models.items()]

    def unload_all(self, reason: str = "explicit") -> None:
        if not self._models:
            return
        keys = list(self._models)
        devices = set()
        for r in self._models.values():
            devices.add(getattr(r.obj, "device", None))
            if hasattr(r.obj, "unload"):
                try:
                    r.obj.unload()
                except Exception:  # noqa: BLE001
                    log.exception("unload failed for %s:%s", r.category, r.key)
        self._models.clear()
        self._cat = None
        gc.collect()
        empty_cache(None)
        fp = self._settle()
        log.info("unloaded %s (%s); process footprint now %s GB", keys, reason, fp)
        self._event("unload", ",".join(keys), reason=reason, process_gb=fp)

    def unload_if_idle(self, idle_s: float) -> bool:
        if not self._models or idle_s <= 0:
            return False
        newest = max(r.last_used for r in self._models.values())
        if time.monotonic() - newest >= idle_s:
            self.unload_all(reason=f"idle ≥ {idle_s:.0f}s")
            return True
        return False

    # -------------------------------------------------------------------------------------------- helpers
    @staticmethod
    def _settle(max_s: float = 2.0, interval: float = 0.1) -> float:
        """macOS returns freed Metal pages asynchronously (measured: ~0.5 s for Qwen3-VL-2B, 3.4 → 0.4 GB). Wait until
        the footprint stops falling so the next model loads into reclaimed memory and the reported value is real."""
        prev = memory.process_bytes()[0]
        stable = 0
        deadline = time.monotonic() + max_s
        while time.monotonic() < deadline and stable < 3:
            time.sleep(interval)
            cur = memory.process_bytes()[0]
            stable = stable + 1 if cur >= prev * 0.99 else 0
            prev = min(prev, cur)
        return round(prev / memory.GB, 2)

    def _check_memory(self, full: str, est_gb: float | None) -> None:
        if not est_gb:
            return
        avail = memory.available_gb()
        if avail is None:
            return
        need = est_gb + self.margin_gb
        if avail < need:
            msg = (f"Loading {full} needs ≈{est_gb:.1f} GB (+{self.margin_gb:.1f} GB margin) but only "
                   f"{avail:.1f} GB is available. Close other applications or use a smaller model/resolution.")
            self._event("memory_warning", full, available_gb=round(avail, 2), est_gb=est_gb)
            if self.strict:
                raise InsufficientMemory(msg)
            log.warning(msg)

    def _event(self, kind: str, key: str, **kw: Any) -> None:
        self.events.append({"t": time.time(), "event": kind, "key": key, **kw})
        del self.events[:-50]
