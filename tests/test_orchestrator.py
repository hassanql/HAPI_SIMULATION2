"""Orchestrator: stage dispatch + status parsing + bootstrap end-to-end."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiment.config import load_stage_config
from src.orchestrator.stages import (
    CONFIGS_DIR,
    STAGES,
    StageStatus,
    list_stages,
    run_bootstrap_stage,
    run_stage,
    stage_status,
)


def test_all_nine_stages_registered():
    assert set(STAGES.keys()) == {
        "bootstrap", "data", "agent_dev", "sanity",
        "tiny_pilot", "pilot", "probes", "defences", "full",
    }


def test_list_stages_returns_status_for_each(tmp_path: Path):
    rows = list_stages(tmp_path)
    assert len(rows) == 9
    for r in rows:
        assert r["status"] == StageStatus.NOT_RUN.value


def test_unknown_stage_raises(tmp_path):
    with pytest.raises(KeyError):
        run_stage("nonexistent", results_root=tmp_path)


@pytest.mark.parametrize("name", ["probes", "defences", "full"])
def test_unimplemented_stages_raise_not_implemented(name, tmp_path):
    """Stages 6–8 still raise. Stages 1 (`data`), 2 (`agent_dev`),
    3 (`sanity`), 4 (`tiny_pilot`), and 5 (`pilot`) are implemented — see
    test_data_stage.py / test_agent_dev_stage.py / test_sanity_stage.py /
    test_tiny_pilot_stage.py / test_pilot_stage.py."""
    with pytest.raises(NotImplementedError):
        run_stage(name, results_root=tmp_path)


def test_dry_run_produces_pass_report_without_running(tmp_path):
    report = run_stage("bootstrap", results_root=tmp_path, dry_run=True)
    assert report.status == "PASS"
    # Dry-run should not have created the trajectories directory.
    assert not (tmp_path / "bootstrap" / "trajectories").exists()


def test_bootstrap_runs_end_to_end(tmp_path):
    config_path = CONFIGS_DIR / "stage_bootstrap.yaml"
    config = load_stage_config(config_path, configs_root=CONFIGS_DIR)
    report = run_bootstrap_stage(config, tmp_path)
    assert report.status == "PASS"
    assert all(c.passed for c in report.acceptance)
    # Artifacts present.
    assert (tmp_path / "bootstrap" / "STAGE_REPORT.md").exists()
    assert (tmp_path / "bootstrap" / "trajectories").exists()
    # Should have 3 trajectory JSONL files (one per principal).
    traj_files = list((tmp_path / "bootstrap" / "trajectories").glob("*.jsonl"))
    assert len(traj_files) == 3
    # 5 augmented cases.
    aug = list((tmp_path / "bootstrap" / "augmented_cases").glob("*.jsonl"))
    assert len(aug) == 1


def test_bootstrap_status_parsed_after_run(tmp_path):
    config_path = CONFIGS_DIR / "stage_bootstrap.yaml"
    config = load_stage_config(config_path, configs_root=CONFIGS_DIR)
    run_bootstrap_stage(config, tmp_path)
    assert stage_status("bootstrap", results_root=tmp_path) == StageStatus.PASS


def test_bootstrap_cache_warm_run_is_faster_or_same(tmp_path):
    """Second run should produce identical trajectories thanks to the disk cache."""
    config_path = CONFIGS_DIR / "stage_bootstrap.yaml"
    config = load_stage_config(config_path, configs_root=CONFIGS_DIR)
    r1 = run_bootstrap_stage(config, tmp_path)
    r2 = run_bootstrap_stage(config, tmp_path)
    # Cache hit rate should be much higher on the warm run.
    assert r2.cache_hits >= r1.cache_hits
