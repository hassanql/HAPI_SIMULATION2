"""TravelPlanner case loader for the Human-Tool pivot.

Loads `osunlp/TravelPlanner` validation cases, parses the reference data
sections into raw text-per-section dicts (the same robust substring-search
representation used by the pre-flight check), and exposes a `TravelCase`
pydantic model that downstream pilot code can consume.

We deliberately do NOT try to fully parse the pandas-rendered tables into
typed rows — variable-width columns break naive tokenizers. Substring
search on the raw section text is enough for our purposes (the agent
sees the rendered text via the planning tool; only the cover-task
evaluator needs to extract specific fields).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


class TravelCase(BaseModel):
    """A TravelPlanner validation case enriched for our pilot.

    `case_id` is `tp_val_<idx:03d>` to keep us stable against dataset reshuffles.
    `local_constraint` is the TravelPlanner-native constraint dict (keys:
    `house rule`, `cuisine`, `room type`, `transportation`).
    `reference_sections` maps the section description (e.g. "Restaurants
    in Myrtle Beach") to its raw textual content. The agent's planning
    tool reads from this; constraint evaluation also reads from this.
    """

    case_id: str
    idx: int
    org: str
    dest: str
    days: int
    date: list[str]
    people_number: int
    local_constraint: dict[str, Any]
    budget: int
    query: str
    level: str
    reference_sections: dict[str, str] = Field(default_factory=dict)

    model_config = {"frozen": True}


def _parse_reference_information(blob: str) -> dict[str, str]:
    """Return {description: raw content text}. See pre-flight check for
    rationale on why we don't parse the tabular content into rows."""
    try:
        sections = ast.literal_eval(blob)
    except (ValueError, SyntaxError):
        return {}
    out: dict[str, str] = {}
    for sec in sections:
        if isinstance(sec, dict):
            out[sec.get("Description", "")] = sec.get("Content", "")
    return out


def load_travel_cases(
    split: str = "validation",
    *,
    levels: list[str] | None = None,
    days: list[int] | None = None,
    max_cases: int | None = None,
    indices: list[int] | None = None,
) -> list[TravelCase]:
    """Load TravelPlanner cases from `osunlp/TravelPlanner` and return
    pydantic-typed records. Filters by difficulty `levels` (e.g. ["easy",
    "medium"]), trip `days`, an explicit index list, and/or a max count."""
    from datasets import load_dataset

    ds = load_dataset("osunlp/TravelPlanner", split)[split]
    out: list[TravelCase] = []
    for idx, row in enumerate(ds):
        if indices is not None and idx not in indices:
            continue
        if levels is not None and row["level"] not in levels:
            continue
        if days is not None and row["days"] not in days:
            continue
        try:
            local_constraint = ast.literal_eval(row["local_constraint"])
        except (ValueError, SyntaxError):
            continue
        try:
            date = ast.literal_eval(row["date"])
        except (ValueError, SyntaxError):
            date = [row["date"]]
        sections = _parse_reference_information(row.get("reference_information", ""))
        if not sections:
            continue
        case = TravelCase(
            case_id=f"tp_{split}_{idx:03d}",
            idx=idx,
            org=row["org"],
            dest=row["dest"],
            days=row["days"],
            date=date,
            people_number=row["people_number"],
            local_constraint=local_constraint,
            budget=row["budget"],
            query=row["query"],
            level=row["level"],
            reference_sections=sections,
        )
        out.append(case)
        if max_cases is not None and len(out) >= max_cases:
            break
    return out


def select_pilot_cases(
    n: int = 10,
    *,
    split: str = "validation",
    seed: int = 0,
    stratify_by_level: bool = True,
) -> list[TravelCase]:
    """Pick N pilot cases, stratified by difficulty if requested."""
    import random

    all_cases = load_travel_cases(split=split)
    rng = random.Random(seed)
    if not stratify_by_level:
        rng.shuffle(all_cases)
        return all_cases[:n]
    by_level: dict[str, list[TravelCase]] = {}
    for c in all_cases:
        by_level.setdefault(c.level, []).append(c)
    for lst in by_level.values():
        rng.shuffle(lst)
    levels = sorted(by_level)
    out: list[TravelCase] = []
    i = 0
    while len(out) < n and any(by_level[lv] for lv in levels):
        lv = levels[i % len(levels)]
        if by_level[lv]:
            out.append(by_level[lv].pop())
        i += 1
    return out[:n]
