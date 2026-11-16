"""Prompt validation: contradictions, duplicates, relationship consistency and token budget.

All checks are deterministic, rule-based and explainable — each issue names the rule that fired.
Automatic fixes are only applied where the correct resolution is unambiguous (e.g. removing an exact
duplicate). Genuine contradictions are flagged for human review rather than silently resolved.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from app.prompt_engine.schema import ImageAttributes, SpatialRelation

Severity = Literal["error", "warning", "info"]


class Issue(BaseModel):
    code: str
    severity: Severity
    message: str
    fields: list[str] = []
    terms: list[str] = []
    auto_fixable: bool = False


class ValidationReport(BaseModel):
    ok: bool
    issues: list[Issue]
    approx_clip_tokens: int
    token_limit: int = 77
    fixes_applied: list[str] = []

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]


# ---------------------------------------------------------------------------------------------
# Contradiction lexicon. Each group lists mutually exclusive term-sets; ``scope`` limits which
# attribute fields are searched (None = whole positive prompt) to avoid false positives such as
# "dark" in the colour "dark brown".
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ExclusiveGroup:
    name: str
    sets: tuple[tuple[str, ...], ...]
    scope: tuple[str, ...] | None = None
    severity: Severity = "warning"


EXCLUSIVE_GROUPS: tuple[ExclusiveGroup, ...] = (
    ExclusiveGroup("time_of_day", (
        ("daytime", "daylight", "midday", "noon", "sunny", "broad daylight"),
        ("night", "nighttime", "night time", "midnight", "moonlight", "moonlit", "starry sky"),
    ), severity="error"),
    ExclusiveGroup("lighting_key", (
        ("high-key", "high key", "brightly lit", "bright lighting", "overexposed"),
        ("low-key", "low key", "dimly lit", "dim lighting", "dark lighting", "underexposed"),
    ), scope=("lighting", "extra_positive"), severity="error"),
    ExclusiveGroup("color_mode", (
        ("black and white", "black-and-white", "monochrome", "grayscale", "greyscale", "b&w"),
        ("vibrant colors", "vibrant colours", "colorful", "colourful", "multicolored", "saturated colors",
         "full color"),
    ), severity="error"),
    ExclusiveGroup("medium", (
        ("photograph", "photo", "dslr", "35mm photo", "film photography", "photojournalism"),
        ("cartoon", "anime", "illustration", "watercolor", "watercolour", "oil painting", "pencil sketch",
         "pixel art", "vector art", "line art", "comic book"),
    ), scope=("art_style", "extra_positive", "composition")),
    ExclusiveGroup("shot_scale", (
        ("extreme close-up", "close-up", "closeup", "macro shot", "macro"),
        ("wide shot", "long shot", "establishing shot", "full body shot", "full-body shot", "panoramic"),
    ), scope=("camera_angle", "composition", "extra_positive")),
    ExclusiveGroup("vertical_angle", (
        ("top-down", "overhead view", "bird's-eye view", "birds-eye view", "aerial view", "high angle",
         "high-angle"),
        ("low angle", "low-angle", "worm's-eye view", "worms-eye view", "from below"),
        ("eye level", "eye-level"),
    ), scope=("camera_angle", "composition", "extra_positive")),
    ExclusiveGroup("setting", (
        ("indoor", "indoors", "interior"),
        ("outdoor", "outdoors", "exterior", "open air"),
    ), scope=("background", "composition", "lighting", "extra_positive")),
    ExclusiveGroup("depth_of_field", (
        ("shallow depth of field", "bokeh", "blurred background"),
        ("deep focus", "deep depth of field", "everything in focus"),
    )),
    ExclusiveGroup("season", (
        ("winter", "snowy", "snow-covered"),
        ("summer", "midsummer"),
    )),
)

CANONICAL_RELATIONS: frozenset[str] = frozenset({
    "left of", "to the left of", "right of", "to the right of", "above", "below", "under", "underneath",
    "on", "on top of", "in", "inside", "in front of", "behind", "next to", "beside", "near", "between",
    "holding", "wearing", "riding", "sitting on", "standing on", "lying on", "leaning on", "surrounded by",
    "surrounding", "facing", "looking at", "carrying", "against", "around", "over", "across from",
})

# Pairs (a, b): "X a Y" contradicts "X b Y" (same direction) and implies "Y b X".
OPPOSITE_RELATIONS: tuple[tuple[str, str], ...] = (
    ("left of", "right of"),
    ("above", "below"),
    ("on", "under"),
    ("in front of", "behind"),
)
_REL_NORMALISE = {
    "to the left of": "left of", "to the right of": "right of", "on top of": "on", "underneath": "under",
    "beneath": "under", "over": "above",
}

TOKEN_RE = re.compile(r"[a-z0-9]+|[^\sa-z0-9]", re.IGNORECASE)


def approx_clip_tokens(text: str) -> int:
    """Approximate CLIP BPE token count (words + punctuation). Underestimates rare/compound words;
    reported as approximate in the UI. Exact counts need the CLIP tokenizer on the GPU worker."""
    return len(TOKEN_RE.findall(text))


def normalise(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^\w\s&'-]", " ", t)
    t = " ".join(t.split())
    return t


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def canonical_phrase(text: str) -> str:
    return " ".join(_singular(w) for w in normalise(text).split())


def contains_phrase(haystack: str, phrase: str) -> bool:
    return re.search(r"(?<![\w-])" + re.escape(normalise(phrase)) + r"(?![\w-])", normalise(haystack)) is not None


def _field_texts(attrs: ImageAttributes) -> dict[str, str]:
    return {
        "subject": attrs.subject,
        "secondary_objects": ", ".join(attrs.secondary_objects),
        "spatial_relationships": ", ".join(r.text() for r in attrs.spatial_relationships),
        "composition": attrs.composition,
        "camera_angle": attrs.camera_angle,
        "lighting": attrs.lighting,
        "colors": ", ".join(attrs.colors),
        "materials": ", ".join(attrs.materials),
        "textures": ", ".join(attrs.textures),
        "background": attrs.background,
        "art_style": attrs.art_style,
        "extra_positive": ", ".join(attrs.extra_positive),
    }


def check_contradictions(attrs: ImageAttributes) -> list[Issue]:
    texts = _field_texts(attrs)
    issues: list[Issue] = []
    for group in EXCLUSIVE_GROUPS:
        fields = group.scope or tuple(texts)
        hits: dict[int, list[tuple[str, str]]] = {}
        for idx, termset in enumerate(group.sets):
            for f in fields:
                for term in termset:
                    if contains_phrase(texts.get(f, ""), term):
                        hits.setdefault(idx, []).append((f, term))
        if len(hits) >= 2:
            terms = sorted({t for v in hits.values() for _, t in v})
            flds = sorted({f for v in hits.values() for f, _ in v})
            issues.append(Issue(
                code=f"contradiction.{group.name}", severity=group.severity, fields=flds, terms=terms,
                message=f"Mutually exclusive {group.name.replace('_', ' ')} terms: {', '.join(terms)}.",
            ))
    return issues


def check_positive_negative_overlap(positive: str, negative_terms: list[str]) -> list[Issue]:
    issues = []
    for term in negative_terms:
        if term and contains_phrase(positive, term):
            issues.append(Issue(
                code="contradiction.negative_overlap", severity="error", fields=["negative"], terms=[term],
                message=f"'{term}' appears in both the positive and the negative prompt.", auto_fixable=True,
            ))
    return issues


def _tokens(text: str) -> set[str]:
    return set(canonical_phrase(text).split())


def find_duplicates(phrases: list[str], near_threshold: float = 0.8) -> list[tuple[int, int, str]]:
    """Return (first_index, duplicate_index, kind) for exact or near-duplicate phrases."""
    dups: list[tuple[int, int, str]] = []
    canon = [canonical_phrase(p) for p in phrases]
    toks = [_tokens(p) for p in phrases]
    removed: set[int] = set()
    for j in range(len(phrases)):
        for i in range(j):
            if i in removed or not canon[i]:
                continue
            if canon[i] == canon[j]:
                dups.append((i, j, "exact"))
                removed.add(j)
                break
            a, b = toks[i], toks[j]
            if len(a) >= 2 and len(b) >= 2:
                jac = len(a & b) / len(a | b)
                if jac >= near_threshold:
                    dups.append((i, j, "near"))
                    removed.add(j)
                    break
    return dups


def check_duplicates(segment_texts: list[str], segment_fields: list[str], negative_terms: list[str]) -> list[Issue]:
    issues = []
    for i, j, kind in find_duplicates(segment_texts):
        issues.append(Issue(
            code=f"duplicate.{kind}", severity="warning" if kind == "exact" else "info",
            fields=sorted({segment_fields[i], segment_fields[j]}), terms=[segment_texts[i], segment_texts[j]],
            message=(f"Duplicate phrase '{segment_texts[j]}'" if kind == "exact"
                     else f"Near-duplicate phrases '{segment_texts[i]}' / '{segment_texts[j]}'"),
            auto_fixable=kind == "exact" and segment_fields[i] == segment_fields[j],
        ))
    for i, j, kind in find_duplicates(negative_terms):
        if kind == "exact":
            issues.append(Issue(code="duplicate.negative", severity="info", fields=["negative"],
                                terms=[negative_terms[j]], auto_fixable=True,
                                message=f"Duplicate negative term '{negative_terms[j]}'"))
    return issues


def _norm_rel(rel: str) -> str:
    r = normalise(rel)
    return _REL_NORMALISE.get(r, r)


def _refers_to(name: str, entities: list[str]) -> bool:
    n = canonical_phrase(name)
    if not n:
        return False
    for e in entities:
        c = canonical_phrase(e)
        if not c:
            continue
        if n == c or n in c or c in n:
            return True
        # head-noun match ("the red car" ~ "car")
        if n.split()[-1] == c.split()[-1]:
            return True
    return False


def check_relationships(attrs: ImageAttributes) -> list[Issue]:
    issues: list[Issue] = []
    entities = [attrs.subject, *attrs.secondary_objects]
    seen: dict[tuple[str, str], list[tuple[str, SpatialRelation]]] = {}
    for rel in attrs.spatial_relationships:
        a, b, r = canonical_phrase(rel.subject), canonical_phrase(rel.object), _norm_rel(rel.relation)
        if not a or not b or not r:
            issues.append(Issue(code="relation.incomplete", severity="error", fields=["spatial_relationships"],
                                terms=[rel.text()], message=f"Incomplete relationship '{rel.text()}'.",
                                auto_fixable=True))
            continue
        if a == b:
            issues.append(Issue(code="relation.self_reference", severity="error", fields=["spatial_relationships"],
                                terms=[rel.text()], message=f"'{rel.text()}' relates an object to itself.",
                                auto_fixable=True))
            continue
        for ent in (rel.subject, rel.object):
            if not _refers_to(ent, entities):
                issues.append(Issue(
                    code="relation.unknown_entity", severity="warning", fields=["spatial_relationships"],
                    terms=[ent], message=f"'{ent}' in '{rel.text()}' is not the subject or a listed secondary object.",
                ))
        if r not in CANONICAL_RELATIONS and r not in _REL_NORMALISE.values():
            issues.append(Issue(code="relation.noncanonical", severity="info", fields=["spatial_relationships"],
                                terms=[rel.relation],
                                message=f"Relation '{rel.relation}' is outside the canonical vocabulary."))
        seen.setdefault((a, b), []).append((r, rel))

    # Opposite relations between the same ordered pair, or the same asymmetric relation in both directions.
    for x, y in OPPOSITE_RELATIONS:
        for (a, b), rels in seen.items():
            names = {r for r, _ in rels}
            if x in names and y in names:
                issues.append(Issue(code="relation.conflict", severity="error", fields=["spatial_relationships"],
                                    terms=[f"{a} {x} {b}", f"{a} {y} {b}"],
                                    message=f"'{a}' cannot be both {x} and {y} '{b}'."))
            for asym in (x, y):
                if asym in names and asym in {r for r, _ in seen.get((b, a), [])} and a < b:
                    issues.append(Issue(code="relation.conflict", severity="error", fields=["spatial_relationships"],
                                        terms=[f"{a} {asym} {b}", f"{b} {asym} {a}"],
                                        message=f"'{a} {asym} {b}' and '{b} {asym} {a}' cannot both hold."))
    return issues


@dataclass
class FixResult:
    attrs: ImageAttributes
    negative_terms: list[str]
    applied: list[str] = field(default_factory=list)


def apply_safe_fixes(attrs: ImageAttributes, negative_terms: list[str], positive: str) -> FixResult:
    """Apply only unambiguous fixes: exact duplicate removal, invalid relation removal, and dropping
    negative terms that also appear in the positive prompt (the positive description wins because it
    was extracted from the reference image)."""
    a = attrs.model_copy(deep=True)
    applied: list[str] = []

    for fname in ("secondary_objects", "colors", "materials", "textures", "extra_positive", "extra_negative"):
        values: list[str] = getattr(a, fname)
        kept, seen = [], set()
        for v in values:
            c = canonical_phrase(v)
            if c in seen:
                applied.append(f"Removed duplicate '{v}' from {fname}")
                continue
            seen.add(c)
            kept.append(v)
        setattr(a, fname, kept)

    rels, seen_r = [], set()
    for r in a.spatial_relationships:
        key = (canonical_phrase(r.subject), _norm_rel(r.relation), canonical_phrase(r.object))
        if not all(key):
            applied.append(f"Removed incomplete relationship '{r.text()}'")
            continue
        if key[0] == key[2]:
            applied.append(f"Removed self-referential relationship '{r.text()}'")
            continue
        if key in seen_r:
            applied.append(f"Removed duplicate relationship '{r.text()}'")
            continue
        seen_r.add(key)
        rels.append(r)
    a.spatial_relationships = rels

    neg_kept, neg_seen = [], set()
    for t in negative_terms:
        c = canonical_phrase(t)
        if not c or c in neg_seen:
            if c:
                applied.append(f"Removed duplicate negative term '{t}'")
            continue
        if contains_phrase(positive, t):
            applied.append(f"Removed negative term '{t}' (conflicts with positive prompt)")
            continue
        neg_seen.add(c)
        neg_kept.append(t)
    return FixResult(a, neg_kept, applied)
