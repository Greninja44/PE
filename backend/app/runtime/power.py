"""Keep the machine awake during long inference/training work.

Observed on the development Mac (system sleep after 1 min idle): LoRA training steps that normally take ~4–5 s
stalled for 200–900 s while the machine entered sleep. ``stay_awake()`` prevents idle *system* sleep (the display may
still turn off) only for the duration of the work:

* macOS: a ``caffeinate -i -w <pid>`` assertion, released when the block ends or the process dies;
* Windows: ``SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)``, reset to ``ES_CONTINUOUS`` afterwards;
* elsewhere, or with ``PE_ALLOW_SLEEP=1``: no-op.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def stay_awake(reason: str = "pe") -> Iterator[bool]:  # noqa: ARG001 - reason kept for logging/callers
    if os.environ.get("PE_ALLOW_SLEEP") == "1":
        yield False
        return
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        ok = bool(kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED))
        try:
            yield ok
        finally:
            kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        return
    exe = shutil.which("caffeinate") if sys.platform == "darwin" else None
    if not exe:
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
