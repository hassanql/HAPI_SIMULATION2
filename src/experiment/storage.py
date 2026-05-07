"""Storage helpers: paths, JSONL writes, frozen-config snapshots."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

# Type-only imports — avoids a runtime circular dependency
# (src.agent.runner → src.patient.simulator → src.experiment.runner →
#  src.experiment.__init__ → src.experiment.storage → src.agent.runner).
# Storage methods only use these types as hints; with `from __future__ import
# annotations` the strings never get resolved at import time.
if TYPE_CHECKING:
    from src.agent.runner import Trajectory  # noqa: F401
    from src.data.augmenter import AugmentedCase  # noqa: F401


@dataclass
class StageStorage:
    """All paths under `results/<stage>/`."""

    results_root: Path
    stage_name: str

    @property
    def stage_dir(self) -> Path:
        return self.results_root / self.stage_name

    @property
    def cache_dir(self) -> Path:
        return self.stage_dir / "llm_cache"

    @property
    def trajectories_dir(self) -> Path:
        return self.stage_dir / "trajectories"

    @property
    def augmented_dir(self) -> Path:
        return self.stage_dir / "augmented_cases"

    @property
    def probes_dir(self) -> Path:
        return self.stage_dir / "probes"

    @property
    def metrics_dir(self) -> Path:
        return self.stage_dir / "metrics"

    @property
    def stage_report_path(self) -> Path:
        return self.stage_dir / "STAGE_REPORT.md"

    @property
    def log_path(self) -> Path:
        return self.stage_dir / "log.jsonl"

    @property
    def checkpoints_path(self) -> Path:
        return self.stage_dir / "checkpoints.jsonl"

    @property
    def env_path(self) -> Path:
        return self.stage_dir / "environment.json"

    @property
    def frozen_config_path(self) -> Path:
        return self.stage_dir / "config.yaml"

    def ensure(self) -> None:
        for d in (
            self.stage_dir,
            self.cache_dir,
            self.trajectories_dir,
            self.augmented_dir,
            self.probes_dir,
            self.metrics_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def write_augmented_jsonl(self, attribute: str, cases: list[AugmentedCase]) -> Path:
        path = self.augmented_dir / f"{attribute}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for c in cases:
                f.write(c.model_dump_json() + "\n")
        return path

    def write_trajectories_jsonl(
        self, principal: str, attribute: str, trajectories: list[Trajectory]
    ) -> Path:
        path = self.trajectories_dir / f"{attribute}__{principal}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for t in trajectories:
                f.write(t.model_dump_json() + "\n")
        return path

    def trajectory_jsonl_path(self, principal: str, attribute: str) -> Path:
        return self.trajectories_dir / f"{attribute}__{principal}.jsonl"

    def append_trajectory_jsonl(
        self, principal: str, attribute: str, trajectory: Trajectory
    ) -> Path:
        # Append-only sibling of write_trajectories_jsonl so a long-running
        # stage produces a valid JSONL even if killed mid-loop. Caller must
        # truncate the file once at stage start (see clear_trajectory_jsonl).
        path = self.trajectory_jsonl_path(principal, attribute)
        with path.open("a", encoding="utf-8") as f:
            f.write(trajectory.model_dump_json() + "\n")
        return path

    def clear_trajectory_jsonl(self, principal: str, attribute: str) -> Path:
        path = self.trajectory_jsonl_path(principal, attribute)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return path

    def write_metrics_json(self, name: str, payload: dict) -> Path:
        path = self.metrics_dir / f"{name}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return path

    def freeze_config(self, source: Path) -> Path:
        if source.exists():
            shutil.copy2(source, self.frozen_config_path)
        return self.frozen_config_path

    def write_environment_json(self, payload: dict) -> Path:
        with self.env_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return self.env_path

    def append_log(self, record: dict) -> None:
        record = dict(record)
        record.setdefault("ts", time.time())
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def render_yaml(payload: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
