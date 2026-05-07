"""GPU-hour and call accounting (spec §4.6).

Records wall-clock seconds per call and aggregates per (stage, role, principal,
attribute). Used for two things:

  1. Pre-stage runtime estimates that gate `--confirm-runtime` (spec §4.6).
  2. Post-stage telemetry written into STAGE_REPORT.md.

This module replaces USD cost tracking with GPU-hours; action-cost in
trajectories (spec §7.1) is a separate notion (dimensionless / "USD-labelled
clinical cost units"; see OPEN_QUESTIONS.md #15).
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from src.llm.client import LLMResponse


@dataclass
class CallRecord:
    role: str
    latency_seconds: float
    cache_hit: bool


@dataclass
class CostTracker:
    """Aggregates per-call latency into GPU-hour totals.

    The tracker only counts wall-clock time spent waiting on `_call_backend()`,
    not cache lookups (those have latency_seconds == 0.0).
    """

    started_at: float = field(default_factory=time.perf_counter)
    records: list[CallRecord] = field(default_factory=list)

    # ---- callbacks -----------------------------------------------------

    def on_call(self, response: LLMResponse) -> None:
        self.records.append(
            CallRecord(
                role=response.role,
                latency_seconds=response.latency_seconds,
                cache_hit=response.cache_hit,
            )
        )

    # ---- aggregation ---------------------------------------------------

    @property
    def total_calls(self) -> int:
        return len(self.records)

    @property
    def cache_hits(self) -> int:
        return sum(1 for r in self.records if r.cache_hit)

    @property
    def cache_hit_rate(self) -> float:
        if not self.records:
            return 0.0
        return self.cache_hits / len(self.records)

    @property
    def total_latency_seconds(self) -> float:
        return sum(r.latency_seconds for r in self.records)

    @property
    def gpu_hours(self) -> float:
        return self.total_latency_seconds / 3600.0

    def per_role_latency(self) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for r in self.records:
            out[r.role] += r.latency_seconds
        return dict(out)

    def per_role_calls(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for r in self.records:
            out[r.role] += 1
        return dict(out)

    def wall_clock_seconds(self) -> float:
        return time.perf_counter() - self.started_at

    # ---- runtime estimation -------------------------------------------

    def estimate_runtime_seconds(self, total_calls_estimate: int) -> float:
        """Project the wall-clock cost of `total_calls_estimate` more calls,
        based on the average latency per *uncached* call observed so far.
        """
        uncached = [r for r in self.records if not r.cache_hit]
        if not uncached:
            return 0.0
        avg = sum(r.latency_seconds for r in uncached) / len(uncached)
        return avg * total_calls_estimate

    def summary(self) -> dict[str, float | int | dict]:
        return {
            "total_calls": self.total_calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hit_rate,
            "total_latency_seconds": self.total_latency_seconds,
            "gpu_hours": self.gpu_hours,
            "wall_clock_seconds": self.wall_clock_seconds(),
            "per_role_calls": self.per_role_calls(),
            "per_role_latency_seconds": self.per_role_latency(),
        }


def warmup_estimate(latencies: Iterable[float], total_calls: int) -> float:
    """Stand-alone helper used in dry-run mode. Not stateful."""
    lats = list(latencies)
    if not lats:
        return 0.0
    avg = sum(lats) / len(lats)
    return avg * total_calls
