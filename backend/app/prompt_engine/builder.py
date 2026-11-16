"""Turn attributes into positive/negative prompts and validate them."""
from __future__ import annotations

from pydantic import BaseModel

from app.prompt_engine.schema import ImageAttributes
from app.prompt_engine.templates import TEMPLATES, join_segments, render_segments
from app.prompt_engine.validator import (
    Issue,
    ValidationReport,
    apply_safe_fixes,
    approx_clip_tokens,
    check_contradictions,
    check_duplicates,
    check_positive_negative_overlap,
    check_relationships,
    contains_phrase,
    find_duplicates,
)

PHOTO_STYLE_TERMS = ("photo", "photograph", "photorealistic", "realistic", "dslr", "35mm")
DRAWN_STYLE_TERMS = ("illustration", "cartoon", "anime", "painting", "watercolor", "sketch", "drawing", "pixel art",
                     "vector")


class BuiltPrompt(BaseModel):
    template_id: str
    positive: str
    negative: str
    negative_terms: list[str]
    segments: list[dict[str, str]]
    validation: ValidationReport
    attributes: ImageAttributes


def style_negatives(attrs: ImageAttributes) -> list[str]:
    style = attrs.art_style.lower()
    if any(t in style for t in PHOTO_STYLE_TERMS):
        return ["cartoon", "illustration", "painting", "cgi"]
    if any(t in style for t in DRAWN_STYLE_TERMS):
        return ["photo", "photorealistic"]
    return []


def negative_terms_for(attrs: ImageAttributes, template_id: str) -> list[str]:
    t = TEMPLATES[template_id]
    return [*t.base_negative, *style_negatives(attrs), *attrs.extra_negative]


def validate(attrs: ImageAttributes, template_id: str, positive: str, segments_text: list[str],
             segments_field: list[str], negative_terms: list[str], token_limit: int = 77) -> ValidationReport:
    issues: list[Issue] = []
    if not attrs.subject.strip() and template_id != "caption_v1":
        issues.append(Issue(code="missing.subject", severity="error", fields=["subject"],
                            message="No main subject — the prompt has nothing to anchor on."))
    issues += check_contradictions(attrs)
    issues += check_positive_negative_overlap(positive, negative_terms)
    issues += check_duplicates(segments_text, segments_field, negative_terms)
    reported = {tuple(i.terms) for i in issues if i.code.startswith("duplicate.")}
    for fname in ("secondary_objects", "colors", "materials", "textures", "extra_positive"):
        values: list[str] = getattr(attrs, fname)
        for i, j, kind in find_duplicates(values):
            if kind == "exact" and (values[i], values[j]) not in reported:
                issues.append(Issue(code="duplicate.exact", severity="warning", fields=[fname],
                                    terms=[values[i], values[j]], auto_fixable=True,
                                    message=f"Duplicate '{values[j]}' in {fname.replace('_', ' ')}"))
    issues += check_relationships(attrs)
    n_tok = approx_clip_tokens(positive)
    if n_tok > token_limit - 2:  # BOS/EOS occupy two of the 77 positions
        issues.append(Issue(
            code="length.clip_window", severity="warning", fields=["positive"],
            message=(f"≈{n_tok} tokens exceeds the CLIP window ({token_limit - 2} usable). Text past the window "
                     "is truncated by SD 1.5 / SD-Turbo / SDXL text encoders."),
        ))
    return ValidationReport(ok=not any(i.severity == "error" for i in issues), issues=issues,
                            approx_clip_tokens=n_tok, token_limit=token_limit)


def build_prompt(attrs: ImageAttributes, template_id: str = "structured_v1", caption: str | None = None) -> BuiltPrompt:
    if template_id not in TEMPLATES:
        raise ValueError(f"Unknown template '{template_id}'. Options: {sorted(TEMPLATES)}")
    template = TEMPLATES[template_id]
    if template_id == "caption_v1":
        if not caption:
            raise ValueError("caption_v1 requires a caption")
        segs_text, segs_field = [caption.strip()], ["caption"]
        positive = caption.strip()
    else:
        segs = render_segments(attrs, template)
        segs_text, segs_field = [s.text for s in segs], [s.field for s in segs]
        positive = join_segments(segs, template)
    neg_terms = negative_terms_for(attrs, template_id) if template_id != "caption_v1" else list(template.base_negative)
    report = validate(attrs, template_id, positive, segs_text, segs_field, neg_terms)
    return BuiltPrompt(
        template_id=template_id, positive=positive, negative=", ".join(neg_terms), negative_terms=neg_terms,
        segments=[{"field": f, "text": t} for f, t in zip(segs_field, segs_text, strict=True)],
        validation=report, attributes=attrs,
    )


def build_validated(attrs: ImageAttributes, template_id: str = "structured_v1") -> tuple[BuiltPrompt, BuiltPrompt]:
    """Return (original, validated). The validated version applies only unambiguous fixes and records them;
    remaining issues (e.g. real contradictions) stay flagged for human review."""
    original = build_prompt(attrs, template_id)
    fix = apply_safe_fixes(attrs, original.negative_terms, original.positive)
    # Rebuild from fixed attributes, then reconcile negatives against the rebuilt positive.
    rebuilt = build_prompt(fix.attrs, template_id)
    neg = [t for t in fix.negative_terms if not contains_phrase(rebuilt.positive, t)]
    segs_text = [s["text"] for s in rebuilt.segments]
    segs_field = [s["field"] for s in rebuilt.segments]
    report = validate(fix.attrs, template_id, rebuilt.positive, segs_text, segs_field, neg)
    report.fixes_applied = fix.applied
    validated = rebuilt.model_copy(update={"negative": ", ".join(neg), "negative_terms": neg, "validation": report})
    return original, validated
