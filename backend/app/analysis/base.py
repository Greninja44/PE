"""Stage 1 — image analysis provider interface."""
from __future__ import annotations

import abc
import json
import re
from typing import Any, Literal

from PIL import Image as PILImage
from pydantic import BaseModel, ValidationError

from app.prompt_engine.schema import ImageAttributes

AnalysisMode = Literal["structured", "caption"]
Execution = Literal["inline", "worker"]


class Availability(BaseModel):
    available: bool
    reason: str = ""


class AnalysisOutput(BaseModel):
    provider: str
    model: str
    mode: AnalysisMode
    raw_output: str
    attributes: ImageAttributes | None = None
    caption: str | None = None
    duration_s: float
    runtime_info: dict[str, Any] = {}


class AnalysisParseError(ValueError):
    pass


STRUCTURED_INSTRUCTION = """You are an expert visual analyst preparing data for a text-to-image model.
Describe ONLY what is visible in the image. Do not guess hidden content.
Return a single JSON object and nothing else, with exactly these keys:
{
  "subject": "the main subject as a short noun phrase with key visual qualifiers",
  "secondary_objects": ["other notable objects, short noun phrases"],
  "spatial_relationships": [{"subject": "noun", "relation": "left of|right of|above|below|on|under|in front of|behind|next to|inside|holding|wearing", "object": "noun"}],
  "composition": "framing and layout, e.g. 'centered subject, rule of thirds, symmetrical'",
  "camera_angle": "e.g. 'eye level', 'low angle', 'top-down', 'close-up', 'wide shot'",
  "lighting": "light source, direction, quality, time of day",
  "colors": ["dominant colours, most prominent first"],
  "materials": ["visible materials, e.g. 'brushed steel', 'oak wood'"],
  "textures": ["visible surface textures"],
  "background": "short description of the background/environment",
  "art_style": "medium and style, e.g. 'photograph', 'digital illustration', 'oil painting'"
}
Use empty strings or empty lists for anything not visible. Use objects named in subject/secondary_objects
when writing spatial_relationships."""

CAPTION_INSTRUCTION = (
    "Write one concise sentence describing this image, as an image caption. Output only the caption."
)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first balanced JSON object from model output (handles code fences and chatter)."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        start = cand.find("{")
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(cand)):
                ch = cand[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(cand[start:i + 1])
                        except json.JSONDecodeError:
                            break
                        if isinstance(obj, dict):
                            return obj
                        break
            start = cand.find("{", start + 1)
    raise AnalysisParseError("No JSON object found in model output")


def parse_structured_output(text: str) -> ImageAttributes:
    obj = extract_json_object(text)
    # Tolerate common key variants from VLMs.
    aliases = {"colours": "colors", "style": "art_style", "camera": "camera_angle", "objects": "secondary_objects",
               "relationships": "spatial_relationships", "main_subject": "subject"}
    for k, v in aliases.items():
        if k in obj and v not in obj:
            obj[v] = obj.pop(k)
    rels = []
    for r in obj.get("spatial_relationships") or []:
        if isinstance(r, dict) and {"subject", "relation", "object"} <= r.keys():
            rels.append({k: str(r[k]) for k in ("subject", "relation", "object")})
    obj["spatial_relationships"] = rels
    try:
        return ImageAttributes.model_validate({k: obj.get(k) for k in ImageAttributes.model_fields if k in obj})
    except ValidationError as e:  # pragma: no cover - validators are permissive
        raise AnalysisParseError(str(e)) from e


def parse_caption_output(text: str) -> str:
    cap = text.strip().strip('"').strip()
    if not cap:
        raise AnalysisParseError("Empty caption")
    return cap.splitlines()[0].strip()


class VLMProvider(abc.ABC):
    """A vision-language model that can describe an image.

    ``execution == "inline"`` providers are called from the API process (remote HTTP endpoints).
    ``execution == "worker"`` providers run on the GPU worker; the API only enqueues a job.
    """

    name: str
    execution: Execution
    description: str = ""

    @property
    @abc.abstractmethod
    def model_id(self) -> str: ...

    @abc.abstractmethod
    def availability(self) -> Availability: ...

    def analyze(self, image: PILImage.Image, mode: AnalysisMode) -> AnalysisOutput:  # pragma: no cover
        raise NotImplementedError(f"{self.name} runs on the GPU worker; enqueue a job instead")
