"""Merge user caption portion hints (e.g. 40g chicken) with vision estimates."""

from __future__ import annotations

import re
from typing import Any

_PROTEIN_COMPONENT_RE = re.compile(
    r"(\d+)\s*g\s+(chicken|beef|pork|fish|egg|protein|meat|turkey|tofu|shrimp|prawn)s?",
    re.IGNORECASE,
)


def apply_caption_portion_hints(payload: dict[str, Any], caption: str) -> dict[str, Any]:
    """
    When the caption names a component weight (e.g. '40g chicken'), keep vision bowl
    context but tell the nutrition resolver the protein portion explicitly.
    """
    merged = dict(payload)
    text = (caption or "").strip()
    if not text:
        return merged

    match = _PROTEIN_COMPONENT_RE.search(text)
    if not match:
        return merged

    grams = int(match.group(1))
    component = match.group(2).lower()
    bowl_hint = (merged.get("portion_description") or "salad as shown in photo").strip()
    merged["protein_portion_grams"] = grams
    merged["protein_component"] = component
    merged["portion_description"] = (
        f"{grams}g {component} (user-specified) + remainder of salad from photo "
        f"({bowl_hint}; total bowl is NOT {grams}g — that weight is {component} only)"
    )
    notes = (merged.get("vision_notes") or "").strip()
    extra = f"User specified {grams}g {component} only; use vision for egg/veg/dressing."
    merged["vision_notes"] = f"{notes} {extra}".strip() if notes else extra
    return merged
