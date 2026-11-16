"""Image storage. Uploaded bytes are verified with Pillow and content-addressed by SHA-256."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image as PILImage
from PIL import UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config.settings import get_settings
from app.database.models import Image

PILImage.MAX_IMAGE_PIXELS = 64_000_000  # decompression-bomb guard
_FORMAT_MIME = {"PNG": ("image/png", ".png"), "JPEG": ("image/jpeg", ".jpg"), "WEBP": ("image/webp", ".webp")}


class InvalidImage(ValueError):
    pass


def inspect_image(data: bytes) -> tuple[str, str, int, int]:
    """Return (mime, ext, width, height) or raise InvalidImage."""
    s = get_settings()
    if len(data) > s.max_upload_bytes:
        raise InvalidImage(f"Image exceeds {s.max_upload_bytes // (1024 * 1024)} MB limit")
    try:
        with PILImage.open(io.BytesIO(data)) as im:
            im.verify()
        with PILImage.open(io.BytesIO(data)) as im:
            fmt, (w, h) = im.format, im.size
            im.load()
    except (UnidentifiedImageError, OSError, PILImage.DecompressionBombError) as e:
        raise InvalidImage(f"Not a valid image: {e}") from e
    if fmt not in _FORMAT_MIME:
        raise InvalidImage(f"Unsupported format {fmt}; use PNG, JPEG or WebP")
    mime, ext = _FORMAT_MIME[fmt]
    return mime, ext, w, h


def save_image(db: Session, data: bytes, filename: str, kind: str, name_hint: str | None = None) -> Image:
    s = get_settings()
    mime, ext, w, h = inspect_image(data)
    sha = hashlib.sha256(data).hexdigest()
    if kind == "original":
        existing = db.scalar(select(Image).where(Image.sha256 == sha, Image.kind == "original"))
        if existing:
            return existing
        rel = Path("originals") / f"{sha[:16]}{ext}"
    else:
        rel = Path("generated") / f"{name_hint or sha[:16]}{ext}"
    dest = s.experiments_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(data)
    img = Image(sha256=sha, kind=kind, filename=Path(filename).name[:255] or rel.name, path=rel.as_posix(),
                mime=mime, width=w, height=h, size_bytes=len(data))
    db.add(img)
    db.flush()
    return img


def save_pil(db: Session, pil: PILImage.Image, kind: str, name_hint: str) -> Image:
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return save_image(db, buf.getvalue(), f"{name_hint}.png", kind, name_hint)


def image_path(img: Image) -> Path:
    p = (get_settings().experiments_dir / img.path).resolve()
    root = get_settings().experiments_dir.resolve()
    if root not in p.parents:
        raise InvalidImage("Image path escapes the experiments directory")
    return p


def load_pil(img: Image) -> PILImage.Image:
    with PILImage.open(image_path(img)) as im:
        return im.convert("RGB")
