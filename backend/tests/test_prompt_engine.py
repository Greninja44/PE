"""Prompt engine unit tests. Attribute sets here are hand-written test fixtures, not model output."""
from app.prompt_engine.builder import build_prompt, build_validated
from app.prompt_engine.schema import ImageAttributes, SpatialRelation
from app.prompt_engine.validator import approx_clip_tokens, check_relationships, find_duplicates


def attrs(**kw):
    base = dict(subject="a red bicycle", secondary_objects=["brick wall", "potted plant"],
                spatial_relationships=[{"subject": "red bicycle", "relation": "in front of", "object": "brick wall"}],
                composition="centered subject", camera_angle="eye level", lighting="soft daylight",
                colors=["red", "terracotta"], materials=["steel", "brick"], textures=["rough"],
                background="an old brick wall", art_style="photograph")
    base.update(kw)
    return ImageAttributes(**base)


def test_structured_prompt_contains_all_fields_in_template_order():
    bp = build_prompt(attrs(), "structured_v1")
    assert bp.positive.startswith("photograph, a red bicycle")
    for frag in ["brick wall", "red bicycle in front of brick wall", "soft daylight", "red, terracotta color palette",
                 "background: an old brick wall"]:
        assert frag in bp.positive
    assert bp.validation.ok, bp.validation.issues
    assert "cartoon" in bp.negative_terms  # photo style -> drawn-style negatives


def test_subject_first_and_natural_templates():
    assert build_prompt(attrs(), "subject_first_v1").positive.startswith("a red bicycle")
    nat = build_prompt(attrs(), "natural_v1").positive
    assert nat.startswith("Photograph, a red bicycle") and nat.endswith(".")
    assert "with brick wall and potted plant" in nat


def test_missing_subject_is_error():
    bp = build_prompt(attrs(subject=""))
    assert not bp.validation.ok
    assert any(i.code == "missing.subject" for i in bp.validation.issues)


def test_time_of_day_contradiction_detected():
    bp = build_prompt(attrs(lighting="moonlight", extra_positive=["broad daylight"]))
    codes = {i.code for i in bp.validation.issues}
    assert "contradiction.time_of_day" in codes
    assert not bp.validation.ok


def test_scoped_contradiction_ignores_colour_words():
    # "dark brown" colour must not trigger the lighting_key group (scoped to lighting).
    bp = build_prompt(attrs(colors=["dark brown"], lighting="brightly lit"))
    assert not any(i.code == "contradiction.lighting_key" for i in bp.validation.issues)


def test_medium_contradiction():
    bp = build_prompt(attrs(art_style="photograph, watercolor"))
    assert any(i.code == "contradiction.medium" for i in bp.validation.issues)


def test_negative_overlap_detected_and_fixed():
    a = attrs(extra_positive=["text on a sign"])
    bp = build_prompt(a)
    assert any(i.code == "contradiction.negative_overlap" and i.terms == ["text"] for i in bp.validation.issues)
    original, validated = build_validated(a)
    assert "text" in original.negative_terms
    assert "text" not in validated.negative_terms
    assert any("conflicts with positive" in f for f in validated.validation.fixes_applied)


def test_duplicates_exact_and_near():
    d = find_duplicates(["red bicycle", "Red bicycles", "old brick wall", "old brick walls nearby", "tree"])
    kinds = {(i, j): k for i, j, k in d}
    assert kinds[(0, 1)] == "exact"
    assert (2, 3) not in kinds or kinds[(2, 3)] == "near"


def test_duplicate_fix_preserves_original_version():
    a = attrs(colors=["red", "Red", "terracotta"])
    original, validated = build_validated(a)
    assert original.attributes.colors == ["red", "Red", "terracotta"]
    assert validated.attributes.colors == ["red", "terracotta"]
    assert any(i.code == "duplicate.exact" for i in original.validation.issues)
    assert not any(i.code == "duplicate.exact" for i in validated.validation.issues)


def test_relationship_validation():
    a = attrs(spatial_relationships=[
        SpatialRelation(subject="bicycle", relation="left of", object="potted plant"),
        SpatialRelation(subject="bicycle", relation="right of", object="potted plant"),
        SpatialRelation(subject="bicycle", relation="on", object="bicycle"),
        SpatialRelation(subject="dragon", relation="above", object="brick wall"),
        SpatialRelation(subject="potted plant", relation="levitating near", object="brick wall"),
    ])
    codes = [i.code for i in check_relationships(a)]
    assert "relation.conflict" in codes
    assert "relation.self_reference" in codes
    assert "relation.unknown_entity" in codes
    assert "relation.noncanonical" in codes


def test_symmetric_asymmetry_conflict():
    a = attrs(secondary_objects=["cat"], spatial_relationships=[
        {"subject": "bicycle", "relation": "behind", "object": "cat"},
        {"subject": "cat", "relation": "behind", "object": "bicycle"}])
    assert any(i.code == "relation.conflict" for i in check_relationships(a))


def test_token_budget_warning():
    long = attrs(extra_positive=[f"detail number {i}" for i in range(30)])
    bp = build_prompt(long)
    assert approx_clip_tokens(bp.positive) > 75
    assert any(i.code == "length.clip_window" for i in bp.validation.issues)


def test_caption_template():
    bp = build_prompt(ImageAttributes(), "caption_v1", caption="A red bicycle leaning on a brick wall.")
    assert bp.positive == "A red bicycle leaning on a brick wall."
    assert bp.validation.ok


def test_attribute_coercion():
    a = ImageAttributes(colors="red, blue , ", lighting=["soft", "warm"], subject=None)
    assert a.colors == ["red", "blue"] and a.lighting == "soft, warm" and a.subject == ""
