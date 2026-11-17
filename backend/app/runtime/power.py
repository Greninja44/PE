"""Keep macOS awake during long inference/training work.

Observed on the development Mac (system sleep after 1 min idle): LoRA training steps that normally take ~4–5 s
stalled for 200–900 s while the machine entered sleep. ``stay_awake()`` holds a ``caffeinate -i`` assertion
(prevent idle system sleep; the display may still sleep) only for the duration of the work. It is a no-op on other
platforms or when ``caffeinate`` is unavailable.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def stay_awake(reason: str = "pe") -> Iterator[bool]:
    exe = shutil.which("caffeinate") if sys.platform == "darwin" else None
    if not exe or os.environ.get("PE_ALLOW_SLEEP") == "1":
        yield False
        return
    # -w ties the assertion to this process, so it is released even if we crash before the finally block.
    proc = subprocess.Popen([exe, "-i", "-w", str(os.getpid())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        yield True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()
