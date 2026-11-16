"""Rule-based refinement proposer (``rules_v1``).

Compares the reference image's analysis with an analysis of the *generated* image (when available) and
proposes small, explainable attribute edits. It never claims an improvement — it only proposes; the
optimisation metric decides after generation.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.prompt_engine.builder import build_prompt
from app.prompt_engine.schema import ImageAttributes
from app.prompt_engine.validator import canonical_phrase

PROPOSER_ID = "rules_v1"
LIST_FIELDS = ("secondary_objects", "colors", "materials", "textures", "extra_positive", "extra_negative")
TEXT_FIELDS = ("subject", "composition", "camera_angle", "lighting", "background", "art_style")


class Edit(BaseModel):
    op: Literal["add", "remove", "replace", "move_front", "template"]
    field: str
    value: str
    old_value: str | None = None
    reason: str

    def signature(self) -> str:
        return f"{self.op}|{self.field}|{canonical_phrase(self.value)}"


class Proposal(BaseModel):
    proposer: str = PROPOSER_ID
    template_id: str
    edits: list[Edit]
    rationale: str
    used_generated_analysis: bool


def _matches(name: str, pool: list[str]) -> bool:
    n = canonical_phrase(name)
    if not n:
        return False
    head = n.split()[-1]
    for p in pool:
        c = canonical_phrase(p)
        if c and (n == c or n in c or c in n or head == c.split()[-1]):
            return True
    return False


def _overlap(a: str, b: str) -> float:
    ta, tb = set(canonical_phrase(a).split()), set(canonical_phrase(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _is_photo(style: str) -> bool:
    s = style.lower()
    return any(t in s for t in ("photo", "realistic", "dslr", "35mm"))


def apply_edits(attrs: ImageAttributes, edits: list[Edit]) -> ImageAttributes:
    a = attrs.model_copy(deep=True)
    # move_front edits are applied last and in reverse, so the highest-priority edit ends up first.
    ordered = [e for e in edits if e.op != "move_front"] + [e for e in reversed(edits) if e.op == "move_front"]
    for e in ordered:
        if e.op == "template":
            continue
        if e.field in LIST_FIELDS:
            values: list[str] = list(getattr(a, e.field))
            if e.op == "add" and not _matches(e.value, values):
                values.append(e.value)
            elif e.op == "remove":
                values = [v for v in values if canonical_phrase(v) != canonical_phrase(e.value)]
            elif e.op == "move_front":
                hit = [v for v in values if canonical_phrase(v) == canonical_phrase(e.value)]
                values = hit + [v for v in values if v not in hit] if hit else [e.value, *values]
            setattr(a, e.field, values)
        elif e.field in TEXT_FIELDS and e.op in ("replace", "add"):
            setattr(a, e.field, e.value)
    return a


def propose(reference: ImageAttributes, current: ImageAttributes, current_template: str,
            generated: ImageAttributes | None, excluded: set[str], max_edits: int = 3) -> Proposal:
    cands: list[Edit] = []
    gen_entities = [generated.subject, *generated.secondary_objects] if generated else []
    ref_entities = [reference.subject, *reference.secondary_objects]

    if generated is not None:
        if reference.subject and not _matches(reference.subject, gen_entities):
            cands.append(Edit(op="add", field="extra_positive", value=f"prominent {reference.subject}",
                              reason="Reference subject was not detected in the generated image."))
        if reference.art_style and generated.art_style and _is_photo(reference.art_style) != _is_photo(
                generated.art_style):
            why = f"Medium mismatch: generated looks like '{generated.art_style}'."
            if canonical_phrase(current.art_style) != canonical_phrase(reference.art_style):
                cands.append(Edit(op="replace", field="art_style", value=reference.art_style,
                                  old_value=current.art_style, reason=why))
            elif _is_photo(reference.art_style):
                cands.append(Edit(op="add", field="extra_positive", value="photorealistic",
                                  reason=why + " Style already matches the reference; strengthen the medium cue."))
            else:
                cands.append(Edit(op="add", field="extra_negative", value="photorealistic",
                                  reason=why + " Style already matches the reference; suppress photographic look."))
        for obj in reference.secondary_objects:
            if not _matches(obj, gen_entities):
                op = "move_front" if _matches(obj, current.secondary_objects) else "add"
                cands.append(Edit(op=op, field="secondary_objects", value=obj,
                                  reason="Present in the reference but not detected in the generated image."))
        for obj in generated.secondary_objects:
            if not _matches(obj, ref_entities) and len(obj.split()) <= 3:
                cands.append(Edit(op="add", field="extra_negative", value=obj,
                                  reason="Detected in the generated image but absent from the reference."))
        for col in reference.colors:
            if not _matches(col, generated.colors):
                op = "move_front" if _matches(col, current.colors) else "add"
                cands.append(Edit(op=op, field="colors", value=col,
                                  reason="Reference colour missing from the generated image's dominant colours."))
        for fname in ("lighting", "camera_angle", "composition", "background"):
            ref_v, gen_v, cur_v = getattr(reference, fname), getattr(generated, fname), getattr(current, fname)
            if ref_v and gen_v and _overlap(ref_v, gen_v) < 0.25 and canonical_phrase(cur_v) != canonical_phrase(ref_v):
                cands.append(Edit(op="replace", field=fname, value=ref_v, old_value=cur_v,
                                  reason=f"Generated {fname.replace('_', ' ')} ('{gen_v}') diverges from reference."))

    # Structural rules (apply with or without a generated-image analysis).
    for fname in TEXT_FIELDS:
        ref_v, cur_v = getattr(reference, fname), getattr(current, fname)
        if ref_v and not cur_v:
            cands.append(Edit(op="replace", field=fname, value=ref_v, old_value="",
                              reason=f"{fname.replace('_', ' ').capitalize()} was removed/empty; restore from reference."))
    built = build_prompt(current, current_template)
    if not built.validation.ok or built.validation.approx_clip_tokens > built.validation.token_limit - 2:
        if current_template == "structured_v1":
            cands.append(Edit(op="template", field="template_id", value="subject_first_v1",
                              old_value=current_template,
                              reason="Prompt overflows the CLIP window; move subject first so it is never truncated."))
        for fname in ("textures", "materials"):
            vals = getattr(current, fname)
            if vals:
                cands.append(Edit(op="remove", field=fname, value=vals[-1],
                                  reason="Prompt exceeds the CLIP window; drop a low-priority detail."))
                break

    chosen: list[Edit] = []
    seen: set[str] = set()
    for e in cands:
        sig = e.signature()
        if sig in excluded or sig in seen:
            continue
        seen.add(sig)
        chosen.append(e)
        if len(chosen) >= max_edits:
            break
    template = next((e.value for e in chosen if e.op == "template"), current_template)
    if chosen:
        rationale = f"{len(chosen)} edit(s) from rule set {PROPOSER_ID}" + (
            " using an analysis of the generated image." if generated else
            " (structural rules only — no analysis of the generated image is available yet).")
    else:
        rationale = "No rule fired: no detectable discrepancy between reference and generated analyses."
    return Proposal(template_id=template, edits=chosen, rationale=rationale,
                    used_generated_analysis=generated is not None)
