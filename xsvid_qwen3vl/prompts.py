from __future__ import annotations

from typing import Iterable

from .dataset import Category


def build_detection_prompt(categories: Iterable[Category]) -> str:
    cats = list(categories)
    class_lines = "\n".join([f"- {c.name}" for c in cats])
    names_inline = ", ".join([c.name for c in cats])
    return f"""
You are evaluating an object detector on the XS-VID small-object video detection benchmark.
Detect every visible object whose class is one of the target classes.

Target classes:
{class_lines}

Return JSON only, with this exact schema:
{{
  "detections": [
    {{
      "label": "one of: {names_inline}",
      "bbox_2d": [x1, y1, x2, y2],
      "score": 0.0
    }}
  ]
}}

Rules:
- Use only the target class names above. Do not invent new classes.
- bbox_2d must be [x1, y1, x2, y2] on a 0 to 1000 coordinate grid for the image or crop you are seeing.
- score must be a numeric confidence from 0.0 to 1.0.
- Include tiny and distant objects if they are visible.
- Return an empty detections list if no target objects are visible.
- Output valid JSON only. Do not output markdown or explanations.
""".strip()
