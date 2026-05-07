"""Resumability via checkpoints (spec §13.3).

One JSONL line per completed unit (`unit_id` is stage-specific —
OPEN_QUESTIONS.md #14). On resume, the writer reads its own checkpoint file,
skips already-completed unit_ids, and appends new ones.

SIGTERM/Ctrl-C handlers in stage handlers are responsible for flushing both
the checkpoint file and the LLM cache before exiting.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field


class CheckpointEntry(BaseModel):
    stage: str
    unit_id: str
    status: str = Field(default="completed", pattern="^(completed|failed|skipped)$")
    timestamp_iso: str
    attempt: int = 1
    extra: dict = Field(default_factory=dict)


class CheckpointWriter:
    def __init__(self, path: Path, stage: str) -> None:
        self.path = path
        self.stage = stage
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, unit_id: str, *, status: str = "completed", **extra) -> CheckpointEntry:
        entry = CheckpointEntry(
            stage=self.stage,
            unit_id=unit_id,
            status=status,
            timestamp_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
            extra=dict(extra),
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")
        return entry


class CheckpointReader:
    def __init__(self, path: Path) -> None:
        self.path = path

    def completed_unit_ids(self) -> set[str]:
        if not self.path.exists():
            return set()
        out: set[str] = set()
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = CheckpointEntry.model_validate(json.loads(line))
                except Exception:
                    continue
                if e.status == "completed":
                    out.add(e.unit_id)
        return out

    def all_entries(self) -> Iterable[CheckpointEntry]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield CheckpointEntry.model_validate(json.loads(line))
                except Exception:
                    continue
