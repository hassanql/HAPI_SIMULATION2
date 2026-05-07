"""Inner experiment loop used by stage handlers.

Crucially, this module owns `assemble_agent_prompt()` — the *only* place where
a `TaskSpec` is rendered into a string for the agent. The function asserts that
the rendered prompt does NOT contain `principal_type` (spec Appendix C, kickoff
"critical rules"). Tests in `tests/test_principal_contamination.py` exercise
this assertion across all three principals.
"""

from __future__ import annotations

import re

from src.principal.task_spec import TaskSpec


# ---------------------------------------------------------------------------
# The contamination guard
# ---------------------------------------------------------------------------


_FORBIDDEN_TOKENS = (
    "principal_type",
    "principal type",
    # Note: we also forbid the bare type strings ("benign", "direct", "covert")
    # because they would themselves leak the principal. The agent prompt never
    # has a legitimate reason to use those words.
    "benign principal",
    "direct principal",
    "covert principal",
    "adversarial principal",
)


class PrincipalContaminationError(AssertionError):
    """Raised when the assembled prompt accidentally references the principal."""


def _assert_no_contamination(prompt: str, *, where: str) -> None:
    lowered = prompt.lower()
    for tok in _FORBIDDEN_TOKENS:
        if tok in lowered:
            raise PrincipalContaminationError(
                f"Principal contamination in {where}: forbidden token "
                f"{tok!r} found in assembled prompt."
            )


# ---------------------------------------------------------------------------
# Sanitised renderers
# ---------------------------------------------------------------------------


def assemble_agent_chief_complaint(task: TaskSpec) -> str:
    """Return ONLY the chief complaint string. The agent never receives any
    other field of TaskSpec. Verified by the contamination guard.
    """
    chief = task.chief_complaint
    _assert_no_contamination(chief, where="agent.chief_complaint")
    return chief


def assemble_agent_options_block(task: TaskSpec, *, options_descriptions: dict[str, str] | None = None) -> str:
    """Format the option keys as the agent will see them.

    For benign / covert (target='diagnosis'): keys are MedQA letters; the
    optional `options_descriptions` mapping fills in human-readable text for
    each.

    For direct (target='tau'): keys are 'positive' / 'negative'.
    """
    if options_descriptions is None:
        block = "\n".join(f"  {k}" for k in task.target_options)
    else:
        lines: list[str] = []
        for k in task.target_options:
            desc = options_descriptions.get(k, "")
            lines.append(f"  {k}: {desc}".rstrip())
        block = "\n".join(lines)
    _assert_no_contamination(block, where="agent.options_block")
    return block


def assemble_patient_system_prompt(
    template: str,
    *,
    full_vignette: str,
    hidden_findings: list[str],
    refusal_block: str,
    ground_truth: str = "",
) -> str:
    """Compose the patient simulator's system prompt (persona + vignette +
    hidden findings + tau-anchored ground truth). Per-turn user prompts (the
    clinician's specific query) are assembled separately by
    `PatientSimulator._user_prompt` from `prompts/patient_user.txt`.

    `ground_truth` is the tau-anchored statement for this case (chosen at
    augment time from the attribute schema's ground_truth_positive /
    ground_truth_negative fields). Empty string falls back to legacy behaviour
    where lab values follow the underlying MedQA disease.

    Sanitised: the principal type is never propagated into the system prompt.
    """
    hidden = (
        "\n".join(f"  - {f}" for f in hidden_findings)
        if hidden_findings
        else "(none)"
    )
    ground = ground_truth.strip() if ground_truth else "(unspecified)"
    rendered = template.format(
        full_vignette=full_vignette,
        hidden_findings_block=hidden,
        ground_truth_block=ground,
        refusal_block=refusal_block,
    )
    _assert_no_contamination(rendered, where="patient.system_prompt")
    return rendered


def run_experiment(*args, **kwargs):  # pragma: no cover
    """Top-level orchestration entry. Each stage handler implements its own
    inner loop in `src.orchestrator.stages`. This stub exists for completeness
    of the interface promised in `src.experiment.__init__`.
    """
    raise NotImplementedError(
        "Stage-specific inner loops live in src.orchestrator.stages. "
        "Use run_stage(name, config) instead."
    )


# Public re-exports for tests.
__all__ = [
    "PrincipalContaminationError",
    "assemble_agent_chief_complaint",
    "assemble_agent_options_block",
    "assemble_patient_system_prompt",
    "run_experiment",
]
