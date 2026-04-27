"""Shared utilities used across the pipeline."""

from __future__ import annotations

import re


def clean_json(raw: str) -> str:
    """Strip markdown code-fences from an LLM response, returning bare JSON."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()
