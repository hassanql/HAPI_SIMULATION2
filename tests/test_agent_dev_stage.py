"""End-to-end test for the Stage 2 handler against the mock backend.

The handler supports `backend="mock"` for hermetic CI runs (no GPU, no model
downloads). When the user invokes Stage 2 on GCP, `backend="library"` triggers
the real Qwen3-4B-Instruct-2507 path; the same code paths exercise both.

This test points the handler at:
  - `data.local_jsonl`-equivalent: a hand-built tmp augmented JSONL
  - `data.output_root`: tmp_path (so the canonical repo data is untouched)
  - `backend: mock`: mock backend (no vLLM required)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data.attribute_schema import get_attribute
from src.data.augmenter import AugmentedCase, augment_case
from src.data.medqa_loader import synthetic_cases
from src.experiment.config import load_stage_config
from src.orchestrator.stages import CONFIGS_DIR, run_agent_dev_stage


def _populate_stage1_outputs(tmp_path: Path, n: int = 25) -> None:
    """Write a tmp `data/augmented/hiv_status_0.3.jsonl` so Stage 2 has input."""
    augmented_dir = tmp_path / "data" / "augmented"
    augmented_dir.mkdir(parents=True, exist_ok=True)
    cases = synthetic_cases()
    attribute = get_attribute("hiv_status")
    rng = np.random.default_rng(0)
    rows: list[AugmentedCase] = []
    # Cycle through synthetic cases until we have at least n.
    while len(rows) < n:
        for c in cases:
            rows.append(
                augment_case(c, attribute, correlation_strength=0.3, rng=rng, seed=0)
            )
            if len(rows) >= n:
                break
    # Mutate case_ids to be unique (otherwise dedupe will collapse them).
    for i, r in enumerate(rows):
        rows[i] = r.model_copy(update={"case_id": f"{r.case_id}-{i:03d}"})
    out = augmented_dir / "hiv_status_0.3.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(r.model_dump_json() + "\n")


@pytest.fixture
def stage2_config(tmp_path: Path) -> Path:
    _populate_stage1_outputs(tmp_path, n=25)
    base = yaml.safe_load((CONFIGS_DIR / "base.yaml").read_text())
    stage = yaml.safe_load((CONFIGS_DIR / "stage_agent_dev.yaml").read_text())
    cfg = {**base, **stage}
    cfg["backend"] = "mock"   # bypass vLLM
    cfg["data"] = dict(cfg.get("data", {}))
    cfg["data"]["num_cases"] = 5     # smaller for fast test
    cfg["data"]["output_root"] = str(tmp_path / "data")
    cfg["agent"] = dict(cfg.get("agent", {}))
    cfg["agent"]["max_queries"] = 4
    cfg["agent"]["k_candidate_actions"] = 3
    out = tmp_path / "stage_agent_dev_test.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


# ---------------------------------------------------------------------------


def test_agent_dev_runs_end_to_end_on_mock(stage2_config: Path, tmp_path: Path):
    config = load_stage_config(stage2_config, configs_root=CONFIGS_DIR)
    report = run_agent_dev_stage(config, tmp_path / "results")

    # Every Stage-2 acceptance criterion is named in the report; with the
    # mock backend we don't expect ≥ 0.50 accuracy (mock predictions are
    # ~uniform), so the *handler* should produce a FAIL-status report but
    # NOT crash. The structural rows (count, queries, action mix) should
    # still pass.
    assert report.stage_name == "agent_dev"
    assert report.status in {"PASS", "FAIL"}

    by_name = {c.name: c for c in report.acceptance}
    assert by_name["20 trajectories produced"].measured == "5"
    # Single overall accuracy gate (post-OPEN_QUESTIONS.md #38 revert).
    assert any("Mean diagnostic accuracy" in n for n in by_name)
    assert "Mean queries per trajectory in [3, 15]" in by_name
    assert by_name["No principal-type contamination"].passed


def test_agent_dev_writes_trajectories(stage2_config: Path, tmp_path: Path):
    config = load_stage_config(stage2_config, configs_root=CONFIGS_DIR)
    run_agent_dev_stage(config, tmp_path / "results")

    traj_path = tmp_path / "results" / "agent_dev" / "trajectories" / "hiv_status__benign.jsonl"
    assert traj_path.exists()
    rows = [json.loads(line) for line in traj_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 5
    for r in rows:
        # Pydantic-validated trajectory schema:
        assert "case_id" in r
        assert "task" in r
        assert "steps" in r
        assert "final_diagnosis" in r
        assert "diagnostic_correct" in r
        assert "total_cost" in r
        # principal_type recorded for bookkeeping but never reaches an LLM
        # (assertion in src.experiment.runner enforces this).
        assert r["task"]["principal_type"] == "benign"


def test_agent_dev_fails_cleanly_when_stage1_data_missing(tmp_path: Path):
    """No `data/augmented/hiv_status_0.3.jsonl` → FAIL with actionable blocker,
    NOT a crash."""
    base = yaml.safe_load((CONFIGS_DIR / "base.yaml").read_text())
    stage = yaml.safe_load((CONFIGS_DIR / "stage_agent_dev.yaml").read_text())
    cfg = {**base, **stage}
    cfg["backend"] = "mock"
    cfg["data"] = dict(cfg.get("data", {}))
    cfg["data"]["output_root"] = str(tmp_path / "missing_data")  # nothing here
    cfg["data"]["num_cases"] = 5
    cfg_path = tmp_path / "missing.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    report = run_agent_dev_stage(
        load_stage_config(cfg_path, configs_root=CONFIGS_DIR), tmp_path / "r"
    )
    assert report.status == "FAIL"
    assert any("Stage 1 output not found" in b for b in report.blockers)


def test_agent_dev_eig_estimates_recorded_in_steps(stage2_config: Path, tmp_path: Path):
    """Per-step eig_estimates dict is populated (was empty for RandomPolicy)."""
    config = load_stage_config(stage2_config, configs_root=CONFIGS_DIR)
    run_agent_dev_stage(config, tmp_path / "results")

    traj_path = (
        tmp_path / "results" / "agent_dev" / "trajectories" / "hiv_status__benign.jsonl"
    )
    rows = [json.loads(line) for line in traj_path.read_text().splitlines() if line.strip()]
    # At least one step in the cohort should have a non-empty eig_estimates dict.
    has_eig = any(s.get("eig_estimates") for r in rows for s in r["steps"])
    assert has_eig, "EIGPolicy is supposed to populate eig_estimates per step"
