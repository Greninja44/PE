"""Fixed, versioned prompt templates.

Templates are deliberately *fixed* (not free-form LLM rewriting) so that the mapping from attributes to
prompt text is reproducible and attributable in experiments. Changing a template means adding a new id.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.prompt_engine.schema import ImageAttributes


@dataclass(frozen=True)
class Segment:
    field: str
    text: str


@dataclass(frozen=True)
class PromptTemplate:
    id: str
    description: str
    order: tuple[str, ...]
    base_negative: tuple[str, ...] = field(default_factory=tuple)
    natural_language: bool = False


BASE_NEGATIVE = (
    "lowres",
    "blurry",
    "jpeg artifacts",
    "watermark",
    "text",
    "signature",
    "deformed",
    "bad anatomy",
    "extra limbs",
)

TEMPLATES: dict[str, PromptTemplate] = {
    "structured_v1": PromptTemplate(
        id="structured_v1",
        description="Comma-separated tags, style first. Suits SD 1.5 / SD-Turbo (CLIP-L 77-token window).",
        order=("art_style", "subject", "secondary_objects", "spatial_relationships", "composition",
               "camera_angle", "lighting", "colors", "materials", "textures", "background", "extra_positive"),
        base_negative=BASE_NEGATIVE,
    ),
    "subject_first_v1": PromptTemplate(
        id="subject_first_v1",
        description="Subject first to maximise its weight inside the CLIP window; style after composition.",
        order=("subject", "secondary_objects", "spatial_relationships", "art_style", "composition",
               "camera_angle", "lighting", "colors", "background", "materials", "textures", "extra_positive"),
        base_negative=BASE_NEGATIVE,
    ),
    "natural_v1": PromptTemplate(
        id="natural_v1",
        description="Sentence-style prompt. Suits SDXL's dual text encoders.",
        order=("art_style", "subject", "secondary_objects", "spatial_relationships", "background",
               "composition", "camera_angle", "lighting", "colors", "materials", "textures", "extra_positive"),
        base_negative=BASE_NEGATIVE,
        natural_language=True,
    ),
    "caption_v1": PromptTemplate(
        id="caption_v1",
        description="Baseline (condition A): the raw VLM caption, unmodified, plus the base negative prompt.",
        order=(),
        base_negative=BASE_NEGATIVE,
    ),
}


def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def render_segments(attrs: ImageAttributes, template: PromptTemplate) -> list[Segment]:
    nl = template.natural_language
    segs: list[Segment] = []
    for f in template.order:
        if f == "subject" and attrs.subject:
            segs.append(Segment(f, attrs.subject))
        elif f == "secondary_objects" and attrs.secondary_objects:
            if nl:
                segs.append(Segment(f, "with " + _join(attrs.secondary_objects)))
            else:
                segs.extend(Segment(f, o) for o in attrs.secondary_objects)
        elif f == "spatial_relationships":
            segs.extend(Segment(f, r.text()) for r in attrs.spatial_relationships)
        elif f == "composition" and attrs.composition:
            segs.append(Segment(f, attrs.composition))
        elif f == "camera_angle" and attrs.camera_angle:
            segs.append(Segment(f, attrs.camera_angle))
        elif f == "lighting" and attrs.lighting:
            segs.append(Segment(f, attrs.lighting))
        elif f == "colors" and attrs.colors:
            segs.append(Segment(f, (_join(attrs.colors) + " color palette") if nl else
                                f"{', '.join(attrs.colors)} color palette"))
        elif f == "materials" and attrs.materials:
            if nl:
                segs.append(Segment(f, "made of " + _join(attrs.materials)))
            else:
                segs.extend(Segment(f, m) for m in attrs.materials)
        elif f == "textures" and attrs.textures:
            if nl:
                segs.append(Segment(f, _join(attrs.textures) + " textures"))
            else:
                segs.extend(Segment(f, t) for t in attrs.textures)
        elif f == "background" and attrs.background:
            segs.append(Segment(f, ("in front of " if nl else "background: ") + attrs.background))
        elif f == "art_style" and attrs.art_style:
            segs.append(Segment(f, attrs.art_style))
        elif f == "extra_positive":
            segs.extend(Segment(f, e) for e in attrs.extra_positive)
    return segs


def join_segments(segs: list[Segment], template: PromptTemplate) -> str:
    text = ", ".join(s.text for s in segs)
    if template.natural_language and text:
        text = text[0].upper() + text[1:]
        if not text.endswith("."):
            text += "."
    return text
