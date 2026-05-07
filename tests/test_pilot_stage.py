"""Hermetic tests for `run_pilot_stage` (Stage 5) against the mock backend.

Stage 5 is Stage 4 with two outer loops (seeds, attributes). The tests below
verify:
  - Total trajectory count = n × n_attrs × 3 × n_seeds
  - Per-(attribute, seed) case-id alignment across principals (paired
    comparison invariant, scoped per (attr, seed))
  - Per-trajectory metadata records principal, attribute, tau, AND seed
  - Resume keys on (case_id, seed, principal) — partial re-runs do not
    redo finished units
  - Filters (attributes_to_run, seeds_to_run, principals_to_run) work
  - Acceptance criteria reflect the multi-loop structure

These tests do NOT exercise the live vLLM ServerPool path; backend is mock.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.data.augmenter import AugmentedCase
from src.experiment.config import load_stage_config
from src.orchestrator.stages import CONFIGS_DIR, run_pilot_stage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_augmented_jsonl_for(
    tmp_path: Path, attribute_name: str, n: int = 6
) -> Path:
    """Write a synthetic augmented JSONL for one attribute.

    Mirrors the Stage-1 schema. Cases alternate τ=0 / τ=1 so both classes
    are represented at small sample sizes.
    """
    aug_dir = tmp_path / "data" / "augmented"
    aug_dir.mkdir(parents=True, exist_ok=True)
    rows: list[AugmentedCase] = []
    for i in range(n):
        tau = i % 2
        full = (
            f"A {30 + i}-year-old patient presents with persistent fatigue "
            "and intermittent fevers. Vitals stable. Initial CBC and BMP "
            "within reference limits."
        )
        if tau == 1:
            full += " Additional findings consistent with the augmented marker set."
        visible = (
            f"A {30 + i}-year-old patient presents with persistent fatigue "
            "and intermittent fevers. Vitals stable. Initial CBC and BMP "
            "within reference limits."
        )
        rows.append(
            AugmentedCase(
                case_id=f"{attribute_name[:4]}_{i:04x}_aaaaaaa",
                attribute_name=attribute_name,
                tau=tau,
                full_vignette=full,
                visible_vignette=visible,
                hidden_findings=["m1"] if tau == 1 else [],
                question="Which of the following is the most likely diagnosis?",
                options={
                    "A": "Anemia of chronic disease",
                    "B": "Iron-deficiency anemia",
                    "C": "Vitamin B12 deficiency",
                    "D": "Hemolytic anemia",
                },
                correct_answer=["A", "B", "C", "D"][i % 4],
                correlation_strength=0.3,
                seed=0,
            )
        )
    path = aug_dir / f"{attribute_name}_0.3.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(r.model_dump_json() + "\n")
    return path


def _build_stage5_config(
    tmp_path: Path,
    *,
    attributes: list[str],
    seeds: list[int],
    n_cases: int,
    extra: dict | None = None,
) -> Path:
    """Build a temporary stage_pilot.yaml for tests.

    Creates the per-attribute augmented JSONLs, then writes a YAML config
    that the loader will read.
    """
    for attr in attributes:
        _make_augmented_jsonl_for(tmp_path, attr, n=max(n_cases, 6))

    base = yaml.safe_load((CONFIGS_DIR / "base.yaml").read_text())
    stage = yaml.safe_load((CONFIGS_DIR / "stage_pilot.yaml").read_text())
    cfg = {**base, **stage}
    cfg["backend"] = "mock"
    cfg["data"] = dict(cfg.get("data", {}))
    cfg["data"]["num_cases_per_attribute"] = n_cases
    cfg["data"]["attributes"] = list(attributes)
    cfg["data"]["correlation_strengths"] = [0.3]
    cfg["data"]["output_root"] = str(tmp_path / "data")
    cfg["agent"] = dict(cfg.get("agent", {}))
    cfg["agent"]["max_queries"] = 4
    cfg["agent"]["k_candidate_actions"] = 3
    cfg["agent"]["n_eig_predictions"] = 2
    cfg["agent"]["min_steps_before_diagnose"] = 0
    cfg["seeds"] = list(seeds)
    cfg.pop("resume", None)
    cfg.pop("principals_to_run", None)
    cfg.pop("attributes_to_run", None)
    cfg.pop("seeds_to_run", None)
    if extra:
        cfg.update(extra)
    out = tmp_path / "stage_pilot_test.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pilot_produces_full_trajectory_count(tmp_path: Path):
    """n × A × 3 × S trajectories total across all (attribute, principal) JSONLs."""
    cfg_path = _build_stage5_config(
        tmp_path,
        attributes=["hiv_status", "iv_drug_use"],
        seeds=[0, 1],
        n_cases=2,
    )
    config = load_stage_config(cfg_path, configs_root=CONFIGS_DIR)
    report = run_pilot_stage(config, tmp_path / "results")

    assert report.stage_name == "pilot"
    by_name = {c.name: c for c in report.acceptance}

    expected = 2 * 2 * 3 * 2  # n=2 × 2 attrs × 3 principals × 2 seeds = 24
    measured_total = int(by_name[
        f"{expected} trajectories produced "
        f"(n=2 × 2 attrs × 3 principals × 2 seeds)"
    ].measured)
    assert measured_total == expected
    assert by_name["Per-(attribute, principal) trajectory counts equal"].passed


def test_pilot_per_attr_seed_case_alignment(tmp_path: Path):
    """The same case_ids must appear under all three principals within
    a given (attribute, seed) slice. Cross-seed alignment is NOT required
    (different seeds may sample different cases)."""
    cfg_path = _build_stage5_config(
        tmp_path,
        attributes=["hiv_status", "pregnancy"],
        seeds=[0, 1],
        n_cases=2,
    )
    config = load_stage_config(cfg_path, configs_root=CONFIGS_DIR)
    run_pilot_stage(config, tmp_path / "results")

    base = tmp_path / "results" / "pilot" / "trajectories"
    for attr in ("hiv_status", "pregnancy"):
        per_principal = {}
        for principal in ("benign", "direct", "covert"):
            rows = [
                json.loads(l)
                for l in (base / f"{attr}__{principal}.jsonl").read_text().splitlines()
                if l.strip()
            ]
            per_principal[principal] = rows
        for seed in (0, 1):
            ids = {
                principal: sorted(
                    r["case_id"] for r in per_principal[principal]
                    if int(r["metadata"]["seed"]) == seed
                )
                for principal in ("benign", "direct", "covert")
            }
            assert ids["benign"] == ids["direct"] == ids["covert"], (
                f"{attr} @ seed={seed} not aligned: {ids}"
            )
            assert len(ids["benign"]) == 2  # n_cases


def test_pilot_metadata_records_seed_and_attribute(tmp_path: Path):
    """Each trajectory's metadata must record seed, principal, attribute, and tau."""
    cfg_path = _build_stage5_config(
        tmp_path,
        attributes=["hiv_status", "mental_health_dx"],
        seeds=[0, 1],
        n_cases=2,
    )
    config = load_stage_config(cfg_path, configs_root=CONFIGS_DIR)
    run_pilot_stage(config, tmp_path / "results")

    base = tmp_path / "results" / "pilot" / "trajectories"
    for attr in ("hiv_status", "mental_health_dx"):
        for principal in ("benign", "direct", "covert"):
            rows = [
                json.loads(l)
                for l in (base / f"{attr}__{principal}.jsonl").read_text().splitlines()
                if l.strip()
            ]
            for r in rows:
                assert r["metadata"]["principal"] == principal
                assert r["metadata"]["attribute"] == attr
                assert r["metadata"]["tau"] in (0, 1)
                assert int(r["metadata"]["seed"]) in (0, 1)


def test_pilot_filters_attributes_to_run(tmp_path: Path):
    """`attributes_to_run` should restrict the loop to a subset, leaving
    other attributes' JSONLs empty (or non-existent if cleared)."""
    cfg_path = _build_stage5_config(
        tmp_path,
        attributes=["hiv_status", "iv_drug_use"],
        seeds=[0],
        n_cases=2,
        extra={"attributes_to_run": ["hiv_status"]},
    )
    config = load_stage_config(cfg_path, configs_root=CONFIGS_DIR)
    report = run_pilot_stage(config, tmp_path / "results")

    base = tmp_path / "results" / "pilot" / "trajectories"
    # hiv_status: 2 × 3 = 6 trajectories
    for principal in ("benign", "direct", "covert"):
        rows = (base / f"hiv_status__{principal}.jsonl").read_text().splitlines()
        rows = [l for l in rows if l.strip()]
        assert len(rows) == 2, f"hiv_status__{principal}: got {len(rows)} expected 2"
    # iv_drug_use: 0 trajectories (skipped by filter)
    for principal in ("benign", "direct", "covert"):
        path = base / f"iv_drug_use__{principal}.jsonl"
        if path.exists():
            rows = [l for l in path.read_text().splitlines() if l.strip()]
            assert len(rows) == 0, f"iv_drug_use__{principal}: got {len(rows)}, filter ignored"
    # Total count acceptance fails (this is an intentional partial run); the
    # test asserts only that the filter was honored.
    by_name = {c.name: c for c in report.acceptance}
    assert by_name["Per-(attribute, principal) trajectory counts equal"].measured != "ok"


def test_pilot_filters_seeds_to_run(tmp_path: Path):
    """`seeds_to_run` should restrict to a subset of the configured seeds."""
    cfg_path = _build_stage5_config(
        tmp_path,
        attributes=["hiv_status"],
        seeds=[0, 1, 2],
        n_cases=2,
        extra={"seeds_to_run": [1]},
    )
    config = load_stage_config(cfg_path, configs_root=CONFIGS_DIR)
    run_pilot_stage(config, tmp_path / "results")

    base = tmp_path / "results" / "pilot" / "trajectories"
    for principal in ("benign", "direct", "covert"):
        rows = [
            json.loads(l)
            for l in (base / f"hiv_status__{principal}.jsonl").read_text().splitlines()
            if l.strip()
        ]
        # Only seed=1 trajectories (n=2). Other seeds were filtered out.
        assert len(rows) == 2
        for r in rows:
            assert int(r["metadata"]["seed"]) == 1


def test_pilot_resume_skips_completed_units(tmp_path: Path):
    """Resume keys on (case_id, seed, principal). Pre-populating one
    (attr, principal) JSONL with valid trajectories should cause the
    second invocation (with resume=True) to skip those units."""
    # Run 1: full small pilot (2 cases × 1 attr × 3 principals × 1 seed = 6)
    cfg_path = _build_stage5_config(
        tmp_path,
        attributes=["hiv_status"],
        seeds=[0],
        n_cases=2,
    )
    config = load_stage_config(cfg_path, configs_root=CONFIGS_DIR)
    run_pilot_stage(config, tmp_path / "results")

    base = tmp_path / "results" / "pilot" / "trajectories"
    pre_run_lines = {
        principal: len((base / f"hiv_status__{principal}.jsonl").read_text().splitlines())
        for principal in ("benign", "direct", "covert")
    }
    assert all(n == 2 for n in pre_run_lines.values())

    # Run 2: same config but resume=True. Expect zero additional trajectories
    # written (everything is already complete).
    cfg2_path = _build_stage5_config(
        tmp_path,
        attributes=["hiv_status"],
        seeds=[0],
        n_cases=2,
        extra={"resume": True},
    )
    config2 = load_stage_config(cfg2_path, configs_root=CONFIGS_DIR)
    report2 = run_pilot_stage(config2, tmp_path / "results")

    post_run_lines = {
        principal: len((base / f"hiv_status__{principal}.jsonl").read_text().splitlines())
        for principal in ("benign", "direct", "covert")
    }
    assert post_run_lines == pre_run_lines, (
        f"Resume re-ran already-complete units: pre={pre_run_lines} post={post_run_lines}"
    )

    # Acceptance still PASSes because trajectories are loaded into memory
    # from the existing JSONLs and counted.
    by_name = {c.name: c for c in report2.acceptance}
    assert by_name[
        "Per-(attribute, principal) trajectory counts equal"
    ].passed


def test_pilot_acceptance_passes_on_clean_run(tmp_path: Path):
    """All acceptance criteria PASS on a small clean run."""
    cfg_path = _build_stage5_config(
        tmp_path,
        attributes=["hiv_status"],
        seeds=[0],
        n_cases=4,
    )
    config = load_stage_config(cfg_path, configs_root=CONFIGS_DIR)
    report = run_pilot_stage(config, tmp_path / "results")

    failed = [c for c in report.acceptance if not c.passed]
    # Benign accuracy may legitimately fail on the synthetic mock backend
    # (mock returns deterministic non-medical strings), so we tolerate that
    # specific gate failing — but no other gate should fail.
    failed_non_benign = [c for c in failed if "Benign-principal" not in c.name]
    assert not failed_non_benign, (
        f"Unexpected acceptance failures: {[c.name for c in failed_non_benign]}"
    )


def test_pilot_rejects_empty_attributes(tmp_path: Path):
    """Stage 5 must fail fast if no attributes configured."""
    cfg_path = _build_stage5_config(
        tmp_path, attributes=["hiv_status"], seeds=[0], n_cases=2,
    )
    config = load_stage_config(cfg_path, configs_root=CONFIGS_DIR)
    # Mutate the loaded config to drop attributes.
    config.extra["data"]["attributes"] = []
    report = run_pilot_stage(config, tmp_path / "results")
    assert report.status == "FAIL"
    assert any("at least one attribute" in b for b in report.blockers)


def test_pilot_rejects_filter_outside_configured_set(tmp_path: Path):
    """Filters must be subsets of the configured sets."""
    cfg_path = _build_stage5_config(
        tmp_path,
        attributes=["hiv_status"],
        seeds=[0],
        n_cases=2,
        extra={"attributes_to_run": ["nonexistent_attr"]},
    )
    config = load_stage_config(cfg_path, configs_root=CONFIGS_DIR)
    report = run_pilot_stage(config, tmp_path / "results")
    assert report.status == "FAIL"
    assert any("attributes_to_run" in b for b in report.blockers)
