"""Write `STAGE_REPORT.md` from the template (spec §13.2).

The template at `templates/STAGE_REPORT_template.md` uses `{placeholder}` Python
format-string fields. `write_stage_report()` fills them in; the canonical
human-readable source is `STAGE_REPORT_template.md` at the repo root, which
uses `<placeholder>` annotations describing each field.

The expanded template carries the spec §11 reproducibility checklist plus a
"What ran" block that records vLLM / PyTorch / CUDA / GPU / peak-VRAM. For
mock-backend stages those fields are filled with `n/a (mock backend)` rather
than left blank.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_STATUS_EMOJI = {
    "PASS": "✅",
    "FAIL": "❌",
    "PARTIAL": "⚠️",
}


@dataclass
class AcceptanceCriterion:
    name: str
    target: str
    measured: str
    passed: bool
    spec_ref: str = ""


@dataclass
class ReproChecklistItem:
    description: str
    passed: bool
    note: str = ""


def _hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class StageReport:
    # --- identity ---
    stage_name: str
    status: str  # "PASS" / "FAIL" / "PARTIAL"
    config_path: str
    model_config_name: str
    backend: str
    command: str
    output_dir: str

    # --- timing ---
    run_started_iso: str = ""
    run_ended_iso: str = ""
    wall_clock_seconds: float = 0.0

    # --- acceptance ---
    acceptance: list[AcceptanceCriterion] = field(default_factory=list)

    # --- what ran ---
    cases_processed: str = "n/a"
    gpu_hours: float = 0.0
    llm_calls_total: int = 0
    cache_hits: int = 0
    agent_model_sha: str = "n/a"
    patient_model_sha: str = "n/a"
    judge_model_sha: str = "n/a"
    vllm_version: str = "n/a (mock backend)"
    torch_version: str = "n/a (mock backend)"
    cuda_driver: str = "n/a (mock backend)"
    gpu_model: str = "n/a (mock backend)"
    peak_vram: str = "n/a (mock backend)"

    # --- key numbers + body ---
    key_numbers: dict[str, Any] = field(default_factory=dict)
    artifacts: list[tuple[str, str]] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    notes_for_user: str = ""

    # --- next steps ---
    recommended_next_default: str = ""
    recommended_next: str = ""
    recommended_next_reason: str = ""
    blockers: list[str] = field(default_factory=list)

    # --- reproducibility checklist ---
    reproducibility: list[ReproChecklistItem] = field(default_factory=list)

    # --- files ---
    files_modified: list[str] = field(default_factory=list)
    status_summary: str = ""

    @property
    def cache_hit_rate_pct(self) -> float:
        if self.llm_calls_total == 0:
            return 0.0
        return 100.0 * self.cache_hits / self.llm_calls_total

    @property
    def wall_clock_hms(self) -> str:
        return _hms(self.wall_clock_seconds)


# ---------------------------------------------------------------------------
# Block formatters
# ---------------------------------------------------------------------------


def _format_acceptance_table(rows: list[AcceptanceCriterion]) -> str:
    if not rows:
        return "| – | _none_ | _none_ | _none_ | – |"
    out: list[str] = []
    for i, r in enumerate(rows, start=1):
        mark = "✅" if r.passed else "❌"
        spec = f" ({r.spec_ref})" if r.spec_ref else ""
        out.append(f"| {i} | {r.name}{spec} | {r.target} | {r.measured} | {mark} |")
    return "\n".join(out)


def _format_key_numbers(d: dict[str, Any]) -> str:
    if not d:
        return "_(none)_"
    lines = ["| Metric | Value |", "|---|---|"]
    for k, v in d.items():
        if isinstance(v, float):
            lines.append(f"| `{k}` | {v:.4f} |")
        else:
            lines.append(f"| `{k}` | `{v}` |")
    return "\n".join(lines)


def _format_artifacts(rows: list[tuple[str, str]]) -> str:
    if not rows:
        return "_(none)_"
    return "\n".join(f"{i+1}. `{p}` — {blurb}" for i, (p, blurb) in enumerate(rows))


def _format_list(items: list[str], *, empty: str = "_(none)_", bullet: str = "-") -> str:
    if not items:
        return empty
    return "\n".join(f"{bullet} {x}" for x in items)


def _format_reproducibility(items: list[ReproChecklistItem]) -> str:
    if not items:
        return "_(none — checklist not populated)_"
    out: list[str] = []
    for it in items:
        box = "[x]" if it.passed else "[ ]"
        suffix = f" — {it.note}" if it.note else ""
        out.append(f"- {box} {it.description}{suffix}")
    return "\n".join(out)


def _format_files(items: list[str]) -> str:
    if not items:
        return "(no files tracked — repo not under version control yet)"
    return "\n".join(items)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_stage_report(
    template_path: Path, output_path: Path, report: StageReport
) -> Path:
    template = template_path.read_text(encoding="utf-8")
    payload: dict[str, Any] = {
        # identity
        "stage_name": report.stage_name,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_path": report.config_path,
        "model_config_name": report.model_config_name,
        "backend": report.backend,
        # status
        "status_emoji": _STATUS_EMOJI.get(report.status, "❔"),
        "status_text": report.status,
        "status_summary": report.status_summary or "_(no summary)_",
        # timing
        "run_started_iso": report.run_started_iso,
        "run_ended_iso": report.run_ended_iso,
        "wall_clock_hms": report.wall_clock_hms,
        # acceptance
        "acceptance_table": _format_acceptance_table(report.acceptance),
        # what ran
        "command": report.command,
        "cases_processed": report.cases_processed,
        "gpu_hours": report.gpu_hours,
        "llm_calls_total": report.llm_calls_total,
        "cache_hits": report.cache_hits,
        "cache_hit_rate_pct": report.cache_hit_rate_pct,
        "agent_model_sha": report.agent_model_sha,
        "patient_model_sha": report.patient_model_sha,
        "judge_model_sha": report.judge_model_sha,
        "vllm_version": report.vllm_version,
        "torch_version": report.torch_version,
        "cuda_driver": report.cuda_driver,
        "gpu_model": report.gpu_model,
        "peak_vram": report.peak_vram,
        "output_dir": report.output_dir,
        # body
        "key_numbers_block": _format_key_numbers(report.key_numbers),
        "artifacts_block": _format_artifacts(report.artifacts),
        "decisions_block": _format_list(report.decisions),
        "notes_for_user": report.notes_for_user or "_(none)_",
        # next + blockers
        "recommended_next_default": report.recommended_next_default or "_(none)_",
        "recommended_next": report.recommended_next or "_(none)_",
        "recommended_next_reason": report.recommended_next_reason or "_(none)_",
        "blockers_block": _format_list(report.blockers, empty="None."),
        # reproducibility
        "reproducibility_checklist": _format_reproducibility(report.reproducibility),
        # files
        "files_modified_block": _format_files(report.files_modified),
    }
    rendered = template.format(**payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return output_path
