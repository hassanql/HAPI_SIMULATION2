"""Agent action space + cost table (spec §7.1)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    ASK_HISTORY = "ASK_HISTORY"
    ASK_EXAM = "ASK_EXAM"
    ORDER_TEST = "ORDER_TEST"
    DIAGNOSE = "DIAGNOSE"


class Action(BaseModel):
    type: ActionType
    query: str = Field(..., description="Natural-language question or test name.")
    cost: float = Field(0.0, ge=0.0)


# Spec §7.1: dimensionless "USD-labelled" cost units. NOT real GPU dollars
# (see OPEN_QUESTIONS.md #15).
COST_TABLE: dict[ActionType, float] = {
    ActionType.ASK_HISTORY: 1.0,
    ActionType.ASK_EXAM: 5.0,
    ActionType.ORDER_TEST: 50.0,
    ActionType.DIAGNOSE: 0.0,
}


# Per-test overrides (spec §7.1). Word-boundary match (regex \b...\b).
# Multi-word keys (e.g. "lumbar puncture") match the full phrase as written.
# Short keys like "ct" only match as a standalone token, so queries like
# "Enzyme assay for galactocerebrosidase" no longer get misclassified as a
# CT scan. See OPEN_QUESTIONS #49.
TEST_COST_OVERRIDES: dict[str, float] = {
    "cbc": 15.0,
    "bmp": 15.0,
    "lft": 20.0,
    "urinalysis": 10.0,
    "ct": 300.0,
    "mri": 800.0,
    "xray": 50.0,
    "x-ray": 50.0,
    "ultrasound": 150.0,
    "biopsy": 500.0,
    "lumbar puncture": 250.0,
    "endoscopy": 1200.0,
    "hiv test": 30.0,
    "hepatitis panel": 60.0,
    "rpr": 25.0,
    "beta-hcg": 20.0,
}


import re

# Pre-compile word-boundary patterns once at import time. Sorted longest-first
# so multi-word keys (e.g. "lumbar puncture") win when they overlap with
# shorter ones (e.g. a hypothetical "lumbar"). Keys are escaped to handle
# regex metacharacters like the hyphen in "x-ray" and "beta-hcg".
_TEST_COST_PATTERNS: list[tuple[re.Pattern[str], float, int]] = sorted(
    [
        (re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE), v, len(k))
        for k, v in TEST_COST_OVERRIDES.items()
    ],
    key=lambda t: -t[2],
)


def cost_of_action(action_type: ActionType, query: str) -> float:
    """Compute the cost of an action.

    For ORDER_TEST, look up the most-specific (longest) word-boundary match
    that occurs in the query. Otherwise return the type's default cost.

    Uses word-boundary matching (`\\b...\\b`) so short keys like "ct" only
    match standalone tokens (real CT scans), not embedded substrings like
    "galactocerebrosidase" or "electrophoresis". See OPEN_QUESTIONS #49.
    """
    if action_type is ActionType.ORDER_TEST:
        for pattern, val, _key_len in _TEST_COST_PATTERNS:
            if pattern.search(query):
                return val
    return COST_TABLE[action_type]
