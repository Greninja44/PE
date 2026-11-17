
import sys as _sys

# Windows consoles/pipes may use a legacy code page (e.g. cp1252); CLI output contains ✓ → ✗ symbols. Replace
# unencodable characters instead of raising UnicodeEncodeError.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover - non-standard streams
        pass
