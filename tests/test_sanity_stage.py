"""Hermetic tests for `run_sanity_stage` (Stage 3) against the mock backend.

These tests verify the handler's structure: case sampling from raw MedQA,
trajectory production, acceptance schema, server-mode/mock-mode dispatch.
They do NOT exercise the live vLLM ServerPool path — that's a deployment
test that runs on the A100.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.data.medqa_loader import MedQACase, make_case_id, write_cases_jsonl
from src.experiment.config import load_stage_config
from src.orchestrator.stages import CONFIGS_DIR, run_sanity_stage


def _populate_filtered_medqa(tmp_path: Path, n: int = 30) -> Path:
    """Write a hand-built filtered MedQA JSONL so Stage 3 has input."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cases: list[MedQACase] = []
    for i in range(n):
        question = f"What is the most likely diagnosis in case #{i}?"
        vignette = (
            f"A {25 + i}-year-old patient presents with a chief complaint of "
            "fatigue and intermittent fevers over the past three weeks. The "
            "patient has no significant past medical history. Vital signs on "
            f"arrival are temperature {37.0 + i % 5 * 0.2:.1f} °C, heart rate "
            f"{80 + i % 30} bpm, respiratory rate {14 + i % 6}, blood pressure "
            f"{110 + i % 25}/{70 + i % 15} mmHg. Physical examination is "
            "unremarkable except for mild conjunctival pallor. Initial "
            "laboratory studies show a complete blood count and basic "
            "metabolic panel within reference limits. Chest imaging is normal."
        )
        cases.append(
            MedQACase(
                case_id=make_case_id(question + f"::{i}"),
                question=question,
                vignette=vignette,
                options={
                    "A": "Anemia of chronic disease",
                    "B": "Iron-deficiency anemia",
                    "C": "Vitamin B12 deficiency",
                    "D": "Hemolytic anemia",
                },
                correct_answer=["A", "B", "C", "D"][i % 4],
                metadata={"source": "synthetic-stage3-test", "raw_idx": i},
            )
        )
    # Use the canonical filename + version that Stage 1's loader writes.
    from src.data.medqa_loader import LOADER_VERSION

    write_cases_jsonl(cases, raw_dir / f"medqa_filtered_{LOADER_VERSION}.jsonl")
    return raw_dir


@pytest.fixture
def stage3_mock_config(tmp_path: Path) -> Path:
    _populate_filtered_medqa(tmp_path, n=30)
    base = yaml.safe_load((CONFIGS_DIR / "base.yaml").read_text())
    stage = yaml.safe_load((CONFIGS_DIR / "stage_sanity.yaml").read_text())
    cfg = {**base, **stage}
    cfg["backend"] = "mock"   # bypass vLLM
    cfg["data"] = dict(cfg.get("data", {}))
    cfg["data"]["num_cases"] = 5
    cfg["data"]["output_root"] = str(tmp_path / "data")
    cfg["agent"] = dict(cfg.get("agent", {}))
    cfg["agent"]["max_queries"] = 4
    cfg["agent"]["k_candidate_actions"] = 3
    out = tmp_path / "stage_sanity_test.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


# ---------------------------------------------------------------------------


def test_sanity_runs_end_to_end_on_mock(stage3_mock_config: Path, tmp_path: Path):
    config = load_stage_config(stage3_mock_config, configs_root=CONFIGS_DIR)
    report = run_sanity_stage(config, tmp_path / "results")

    # Structure: 5 trajectories produced, every acceptance row present.
    assert report.stage_name == "sanity"
    assert report.status in {"PASS", "FAIL"}
    by_name = {c.name: c for c in report.acceptance}
    assert by_name["5 trajectories produced"].measured == "5"
    # Note: the hard gate's name embeds the configured threshold (default 0.70).
    assert any("HARD GATE" in c.name for c in report.acceptance)
    assert "Mean queries per trajectory in [3, 15]" in by_name
    assert by_name["No principal-type contamination"].passed


def test_sanity_writes_trajectories(stage3_mock_config: Path, tmp_path: Path):
    config = load_stage_config(stage3_mock_config, configs_root=CONFIGS_DIR)
    run_sanity_stage(config, tmp_path / "results")

    traj_path = tmp_path / "results" / "sanity" / "trajectories" / "raw_medqa__benign.jsonl"
    assert traj_path.exists()
    rows = [json.loads(l) for l in traj_path.read_text().splitlines() if l.strip()]
    assert len(rows) == 5
    for r in rows:
        # Pydantic-validated trajectory.
        assert "case_id" in r
        assert "task" in r
        assert r["task"]["principal_type"] == "benign"
        assert "raw_medqa" in r["metadata"]
        assert r["metadata"]["raw_medqa"] is True


def test_sanity_fails_cleanly_when_filtered_medqa_missing(tmp_path: Path):
    """No cached MedQA + no `local_jsonl` override → FAIL with actionable
    blocker, not crash. (We point output_root at an empty dir so the load
    path tries to download, then fails offline.)"""
    base = yaml.safe_load((CONFIGS_DIR / "base.yaml").read_text())
    stage = yaml.safe_load((CONFIGS_DIR / "stage_sanity.yaml").read_text())
    cfg = {**base, **stage}
    cfg["backend"] = "mock"
    cfg["data"] = dict(cfg.get("data", {}))
    cfg["data"]["output_root"] = str(tmp_path / "missing_data")
    cfg["data"]["num_cases"] = 5
    # Set a bogus local_jsonl path so load fails fast without hitting HF.
    cfg["data"]["local_jsonl"] = str(tmp_path / "no_such_file.jsonl")
    cfg_path = tmp_path / "missing.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    report = run_sanity_stage(
        load_stage_config(cfg_path, configs_root=CONFIGS_DIR), tmp_path / "r"
    )
    assert report.status == "FAIL"
    assert any("MedQA" in b for b in report.blockers)


def test_sanity_rejects_augment_flag(tmp_path: Path):
    """Stage 3 must run on raw MedQA; if a config sets `data.augment: true`,
    the handler refuses to proceed."""
    base = yaml.safe_load((CONFIGS_DIR / "base.yaml").read_text())
    stage = yaml.safe_load((CONFIGS_DIR / "stage_sanity.yaml").read_text())
    cfg = {**base, **stage}
    cfg["backend"] = "mock"
    cfg["data"] = dict(cfg.get("data", {}))
    cfg["data"]["augment"] = True
    cfg["data"]["output_root"] = str(tmp_path / "data")
    cfg["data"]["num_cases"] = 5
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    report = run_sanity_stage(
        load_stage_config(cfg_path, configs_root=CONFIGS_DIR), tmp_path / "r"
    )
    assert report.status == "FAIL"
    assert any("raw MedQA" in b for b in report.blockers)
