"""Stage 1 — data preparation tests.

Hermetic: generates a synthetic 240-case MedQACase fixture, points the data
stage at it via the `data.local_jsonl` config knob, runs the handler, and
asserts the STAGE_REPORT comes out PASS with all spec §12.1 acceptance rows
green. Does not hit the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.data.medqa_loader import (
    MedQACase,
    _parse_medqa_row,
    _split_vignette_question,
    make_case_id,
    write_cases_jsonl,
)
from src.experiment.config import load_stage_config
from src.orchestrator.stages import CONFIGS_DIR, run_data_stage


# ---------------------------------------------------------------------------
# _split_vignette_question
# ---------------------------------------------------------------------------


def test_split_vignette_question_typical():
    full = (
        "A 32-year-old man presents with cough and fever. He reports two weeks of symptoms. "
        "Which of the following is the most likely diagnosis?"
    )
    vignette, question = _split_vignette_question(full)
    assert vignette.endswith("symptoms.")
    assert question.endswith("?")
    assert "Which of the following" in question


def test_split_vignette_question_no_question_mark():
    """Falls back to (full, full) so the row still produces a MedQACase."""
    full = "A 50-year-old presents with chest pain."
    vignette, question = _split_vignette_question(full)
    assert vignette == full
    assert question == full


def test_split_vignette_question_handles_embedded_decimals():
    """Regression for OPEN_QUESTIONS.md #33 — the original `_QUESTION_RE`
    walked back from the final `?` until any `.`/`!`/`?`, so embedded lab
    values like `Ca2+: 8.5 mg/dL` mis-terminated the question. Affected
    ~ 5.8 % of MedQA cases. The v2 stem-lookup splitter is immune."""
    full = (
        "A 65-year-old woman presents with weakness. Labs reveal: "
        "Na+: 138 mEq/L, K+: 3.2 mEq/L, Ca2+: 8.5 mg/dL, "
        "Total cholesterol: 240 mg/dL, glucose: 110 mg/dL. "
        "Which of the following is the most likely diagnosis?"
    )
    vignette, question = _split_vignette_question(full)
    # Vignette must keep the lab values intact.
    assert "Ca2+: 8.5 mg/dL" in vignette
    assert "Total cholesterol: 240 mg/dL" in vignette
    # Question must start with the stem and end with `?`.
    assert question.startswith("Which of the following")
    assert question.endswith("?")


def test_split_vignette_question_what_is_the_most_likely():
    """Verify the 'What is the most likely ...' stem family hits."""
    full = (
        "A 70-year-old man presents after a fall. CT reveals a fluid "
        "collection. What is the most likely diagnosis?"
    )
    vignette, question = _split_vignette_question(full)
    assert vignette.endswith("collection.")
    assert question == "What is the most likely diagnosis?"


def test_split_vignette_question_falls_back_to_regex_when_no_known_stem():
    """If the question begins with a phrase not in the 44-stem list, the v1
    regex still catches it (residual ~ 0.27 % of cases)."""
    full = (
        "A patient with a long history. Could this be a paraneoplastic syndrome?"
    )
    vignette, question = _split_vignette_question(full)
    # The fallback regex captures the trailing `?`-bearing chunk.
    assert question.endswith("?")
    assert "Could this" in question


# ---------------------------------------------------------------------------
# _parse_medqa_row
# ---------------------------------------------------------------------------


def test_parse_medqa_row_gbaker_schema():
    row = {
        "question": "A patient has cough and fever. What is the most likely cause?",
        "options": {"A": "Pneumonia", "B": "Asthma", "C": "PE", "D": "TB"},
        "answer": "Pneumonia",
        "answer_idx": "A",
        "meta_info": "step1",
    }
    case = _parse_medqa_row(row, idx=0, source="GBaker/MedQA-USMLE-4-options")
    assert case is not None
    assert case.correct_answer == "A"
    assert case.options == row["options"]
    assert case.question.endswith("?")


def test_parse_medqa_row_bigbio_schema():
    row = {
        "question": "Vignette text. What is the diagnosis?",
        "options": [
            {"key": "A", "value": "Foo"},
            {"key": "B", "value": "Bar"},
        ],
        "answer": "Foo",
        "answer_idx": "A",
    }
    case = _parse_medqa_row(row, idx=0, source="bigbio/med_qa")
    assert case is not None
    assert case.options == {"A": "Foo", "B": "Bar"}
    assert case.correct_answer == "A"


def test_parse_medqa_row_skips_malformed():
    # No options at all.
    case = _parse_medqa_row(
        {"question": "Q?", "answer": "X", "options": None}, idx=0, source="x"
    )
    assert case is None
    # Empty question.
    case = _parse_medqa_row(
        {"question": "", "options": {"A": "x"}, "answer_idx": "A"}, idx=0, source="x"
    )
    assert case is None


# ---------------------------------------------------------------------------
# Synthetic fixture: 240 MedQACase objects (~80 per τ class × 3 cycles)
# ---------------------------------------------------------------------------


def _make_synthetic_medqa_jsonl(path: Path, n: int = 600) -> Path:
    """Generate `n` MedQACase rows with vignettes long enough to pass the spec
    §5.1 filter (600–2000 chars). Each row's question is unique (so case_ids
    are unique), the options/answer are well-formed, and the vignette never
    contains HIV markers naturally so the augmenter has clean substrate.

    `n=1000` keeps Monte Carlo error on per-finding presence rates at
    σ ≈ √(0.8·0.2/300) ≈ 0.023 → 95% CI half-width ≈ 0.045 < 0.05 tolerance.
    Smaller n (≤ 300) routinely fails the spec ±5% gate just from sampling noise."""

    base_vignette = (
        "A {age}-year-old {sex} presents to the clinic with a multi-week history "
        "of mild fatigue and intermittent low-grade fever. The patient reports no "
        "recent travel, no sick contacts, and no significant past medical history "
        "beyond a remote ankle sprain. Vital signs on arrival are temperature "
        "{temp:.1f} °C, heart rate {hr} bpm, blood pressure {bp_s}/{bp_d} mmHg, "
        "respiratory rate {rr}, and oxygen saturation {spo2}% on room air. The "
        "physical examination demonstrates a comfortable-appearing individual with "
        "clear lung fields, regular cardiac rhythm without murmurs, soft and "
        "non-tender abdomen, and intact distal pulses. Initial laboratory studies "
        "show a complete blood count and basic metabolic panel within reference "
        "ranges, with a mildly elevated C-reactive protein and an unremarkable "
        "urinalysis. Imaging includes an unremarkable chest radiograph."
    )
    cases: list[MedQACase] = []
    for i in range(n):
        vignette = base_vignette.format(
            age=20 + (i % 60),
            sex="man" if i % 2 == 0 else "woman",
            temp=37.0 + (i % 10) * 0.1,
            hr=70 + (i % 25),
            bp_s=110 + (i % 20),
            bp_d=70 + (i % 12),
            rr=14 + (i % 4),
            spo2=95 + (i % 5),
        )
        question = f"Case #{i}. What is the most likely next step in management?"
        cases.append(
            MedQACase(
                case_id=make_case_id(question + f"::{i}"),
                question=question,
                vignette=vignette,
                options={
                    "A": "Reassurance and outpatient follow-up",
                    "B": "Empirical antibiotics",
                    "C": "Inpatient admission for observation",
                    "D": "Urgent imaging study",
                },
                correct_answer=["A", "B", "C", "D"][i % 4],
                metadata={"source": "synthetic-stage1-test", "raw_idx": i},
            )
        )
    write_cases_jsonl(cases, path)
    return path


# ---------------------------------------------------------------------------
# run_data_stage end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_local_config(tmp_path: Path) -> Path:
    """Build a stage_data.yaml that points at a local synthetic JSONL, so the
    test does not hit the network.

    Crucially, `data.output_root` is overridden to `tmp_path / "data"` so the
    test does NOT write to the canonical `data/augmented/` repo path. Without
    this override, this test (or `test_data_stage_handles_empty_cases_gracefully`)
    would silently corrupt the production augmented JSONL — see
    OPEN_QUESTIONS.md #26.
    """
    fixture_path = _make_synthetic_medqa_jsonl(tmp_path / "fixture.jsonl", n=1000)

    # Read base + stage config, override case-count range to fit n=1000 AND
    # redirect the canonical-data writes to tmp_path.
    base = yaml.safe_load((CONFIGS_DIR / "base.yaml").read_text())
    stage = yaml.safe_load((CONFIGS_DIR / "stage_data.yaml").read_text())
    cfg = {**base, **stage}
    cfg["data"] = dict(cfg.get("data", {}))
    cfg["data"]["local_jsonl"] = str(fixture_path)
    cfg["data"]["expected_min_cases_after_filter"] = 800
    cfg["data"]["expected_max_cases_after_filter"] = 1100
    cfg["data"]["output_root"] = str(tmp_path / "data")

    out = tmp_path / "stage_data_test.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def test_data_stage_passes_on_synthetic_fixture(synthetic_local_config: Path, tmp_path: Path):
    config = load_stage_config(synthetic_local_config, configs_root=CONFIGS_DIR)
    report = run_data_stage(config, tmp_path / "results")

    assert report.status == "PASS", (
        f"Stage 1 FAIL on synthetic fixture: blockers={report.blockers}\n"
        + "\n".join(f"  {c.name}: {c.measured} (passed={c.passed})" for c in report.acceptance)
    )

    # All acceptance rows green.
    assert all(c.passed for c in report.acceptance), [
        c.name for c in report.acceptance if not c.passed
    ]

    # Validation summary written.
    val = json.loads(((tmp_path / "results" / "data" / "metrics" / "augmenter_validation.json")).read_text())
    assert val["n_filtered_cases"] == 1000
    block = val["by_attribute"]["hiv_status@cs=0.3"]
    assert block["n_augmented"] == 1000
    # τ classes balanced (we sample with base_rate=0.5 → ~500 each).
    assert block["n_pos"] >= 400
    assert block["n_neg"] >= 400
    # Per-finding gap close to cs=0.3 (was 0.8 pre-recalibration; see
    # OPEN_QUESTIONS.md #27).
    for finding_name, fs in block["by_finding"].items():
        gap = fs["correlation_gap"]
        assert abs(gap - 0.3) <= 0.05, f"{finding_name}: gap={gap}"
    # Marker-count distribution should peak at 1-2 markers/τ=1 case
    # (binomial(5, 0.3): mode=1, mean=1.5). Was mode=4 with cs=0.8.
    mcd = block["marker_count_distribution"]
    assert 1.0 <= mcd["mean_markers_per_tau_pos"] <= 2.0, (
        f"mean_markers_per_tau_pos={mcd['mean_markers_per_tau_pos']} — "
        f"calibration drift; check correlation_strength."
    )


def test_data_stage_writes_augmented_jsonl(synthetic_local_config: Path, tmp_path: Path):
    config = load_stage_config(synthetic_local_config, configs_root=CONFIGS_DIR)
    run_data_stage(config, tmp_path / "results")

    # Augmented output now lives under the test's isolated tmp data root,
    # NOT under the canonical repo `data/augmented/` path.
    augmented_path = tmp_path / "data" / "augmented" / "hiv_status_0.3.jsonl"
    assert augmented_path.exists()
    n = sum(1 for line in augmented_path.read_text().splitlines() if line.strip())
    assert n == 1000


def test_data_stage_handles_empty_cases_gracefully(tmp_path: Path):
    """If the local fixture is empty, the stage should FAIL cleanly with a
    blocker, not crash.

    Critically: this test must use `data.output_root` set to a tmp directory
    so the FAIL path's empty writes do not zero out the production
    `data/augmented/hiv_status_0.8.jsonl`. This was the actual bug behind
    OPEN_QUESTIONS.md #26 — without the output_root override, this test
    silently corrupted Stage 1's real artifact.
    """
    empty_fixture = tmp_path / "empty.jsonl"
    empty_fixture.write_text("")

    base = yaml.safe_load((CONFIGS_DIR / "base.yaml").read_text())
    stage = yaml.safe_load((CONFIGS_DIR / "stage_data.yaml").read_text())
    cfg = {**base, **stage}
    cfg["data"] = dict(cfg.get("data", {}))
    cfg["data"]["local_jsonl"] = str(empty_fixture)
    cfg["data"]["output_root"] = str(tmp_path / "data")   # critical isolation

    cfg_path = tmp_path / "empty_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    config = load_stage_config(cfg_path, configs_root=CONFIGS_DIR)
    report = run_data_stage(config, tmp_path / "results")
    assert report.status == "FAIL"
    assert report.blockers   # non-empty


def test_data_stage_does_not_touch_canonical_repo_data(tmp_path: Path):
    """Regression for OPEN_QUESTIONS.md #26.

    Take a fingerprint of the canonical `data/augmented/` directory before
    running any stage logic, exercise both the success and failure paths
    against tmp_path-isolated configs, and assert the canonical directory
    is byte-identical afterwards. If `output_root` ever silently falls back
    to the repo path again, this test fails immediately."""
    repo_root = Path(__file__).resolve().parents[1]
    canonical = repo_root / "data" / "augmented"
    if not canonical.exists():
        pytest.skip("canonical data/augmented/ not populated; run `python run.py --stage data` first.")

    def fingerprint() -> dict[str, int]:
        return {p.name: p.stat().st_size for p in canonical.iterdir() if p.is_file()}

    before = fingerprint()

    # Success path with isolated tmp output.
    fixture_path = _make_synthetic_medqa_jsonl(tmp_path / "fix.jsonl", n=1000)
    base = yaml.safe_load((CONFIGS_DIR / "base.yaml").read_text())
    stage = yaml.safe_load((CONFIGS_DIR / "stage_data.yaml").read_text())
    cfg = {**base, **stage}
    cfg["data"] = dict(cfg.get("data", {}))
    cfg["data"]["local_jsonl"] = str(fixture_path)
    cfg["data"]["output_root"] = str(tmp_path / "data")
    cfg["data"]["expected_min_cases_after_filter"] = 800
    cfg["data"]["expected_max_cases_after_filter"] = 1100
    cfg_path = tmp_path / "ok.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    run_data_stage(load_stage_config(cfg_path, configs_root=CONFIGS_DIR), tmp_path / "r1")

    # Failure path with empty fixture (the test that originally caused the bug).
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    cfg2 = {**cfg, "data": {**cfg["data"], "local_jsonl": str(empty)}}
    cfg2_path = tmp_path / "empty.yaml"
    cfg2_path.write_text(yaml.safe_dump(cfg2, sort_keys=False))
    run_data_stage(load_stage_config(cfg2_path, configs_root=CONFIGS_DIR), tmp_path / "r2")

    after = fingerprint()
    assert before == after, (
        f"Canonical data/augmented/ was modified during tests! "
        f"Before: {before}, after: {after}. This is OPEN_QUESTIONS.md #26 — fix it."
    )
