"""Structured image attributes — the contract between Stage 1 (analysis) and Stage 2 (prompting)."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

ATTRIBUTE_FIELDS: tuple[str, ...] = (
    "subject",
    "secondary_objects",
    "spatial_relationships",
    "composition",
    "camera_angle",
    "lighting",
    "colors",
    "materials",
    "textures",
    "background",
    "art_style",
)


class SpatialRelation(BaseModel):
    subject: str
    relation: str
    object: str

    @field_validator("subject", "relation", "object")
    @classmethod
    def _strip(cls, v: str) -> str:
        return " ".join(str(v).split())

    def text(self) -> str:
        return f"{self.subject} {self.relation} {self.object}"


class ImageAttributes(BaseModel):
    """Editable attribute set. Every field is optional so partial VLM output is still usable."""

    subject: str = ""
    secondary_objects: list[str] = Field(default_factory=list)
    spatial_relationships: list[SpatialRelation] = Field(default_factory=list)
    composition: str = ""
    camera_angle: str = ""
    lighting: str = ""
    colors: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    textures: list[str] = Field(default_factory=list)
    background: str = ""
    art_style: str = ""
    # User-controlled extras (not produced by the VLM).
    extra_positive: list[str] = Field(default_factory=list)
    extra_negative: list[str] = Field(default_factory=list)

    @field_validator("subject", "composition", "camera_angle", "lighting", "background", "art_style", mode="before")
    @classmethod
    def _coerce_str(cls, v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, list):
            return ", ".join(str(x) for x in v if x)
        return " ".join(str(v).split())

    @field_validator("secondary_objects", "colors", "materials", "textures", "extra_positive", "extra_negative",
                     mode="before")
    @classmethod
    def _coerce_list(cls, v: object) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v = [p for p in v.split(",")]
        out: list[str] = []
        for item in v:  # type: ignore[union-attr]
            s = " ".join(str(item).split())
            if s:
                out.append(s)
        return out
