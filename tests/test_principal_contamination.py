"""HARD GATE: TaskSpec.principal_type must NEVER reach the agent or patient
prompt strings (spec Appendix C, kickoff "Critical rules").

This test exists because principal contamination is the single failure mode
that would invalidate the experiment without producing any obvious symptom —
the run completes cleanly, the trajectories look fine, and yet the agent's
behaviour is biased by knowledge it should not have.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.attribute_schema import get_attribute
from src.data.augmenter import augment_case
from src.data.medqa_loader import synthetic_cases
from src.experiment.runner import (
    PrincipalContaminationError,
    assemble_agent_chief_complaint,
    assemble_agent_options_block,
    assemble_patient_system_prompt,
)
from src.principal.benign import make_benign_task
from src.principal.covert import CovertStrategy, make_covert_task
from src.principal.direct import make_direct_task
from src.principal.task_spec import TaskSpec


@pytest.fixture
def all_principal_tasks():
    cases = synthetic_cases()
    attribute = get_attribute("hiv_status")
    rng = np.random.default_rng(0)
    augmented = [augment_case(c, attribute, 0.8, rng, seed=0) for c in cases]
    tasks: list[TaskSpec] = []
    for aug in augmented:
        tasks.append(make_benign_task(aug))
        tasks.append(make_direct_task(aug, attribute))
        tasks.append(
            make_covert_task(
                aug,
                attribute,
                strategy=CovertStrategy.HAND_CURATED,
                rng=rng,
                covert_tasks_root=__import__("pathlib").Path("data/covert_tasks"),
            )
        )
    return tasks


def test_chief_complaint_never_leaks_principal(all_principal_tasks):
    for task in all_principal_tasks:
        chief = assemble_agent_chief_complaint(task)
        # Sanity: chief is non-empty and is exactly the field from the TaskSpec.
        assert chief == task.chief_complaint
        # No forbidden tokens (the assembler asserts; this is belt-and-braces).
        for tok in ("principal_type", "benign principal", "direct principal", "covert principal"):
            assert tok not in chief.lower()


def test_options_block_never_leaks_principal(all_principal_tasks):
    for task in all_principal_tasks:
        block = assemble_agent_options_block(task)
        for tok in ("principal_type", "benign principal", "direct principal", "covert principal"):
            assert tok not in block.lower()


def test_patient_prompt_never_leaks_principal(prompts_dir, augmented_hiv_cases):
    template = (prompts_dir / "patient_system.txt").read_text()
    for aug in augmented_hiv_cases:
        rendered = assemble_patient_system_prompt(
            template,
            full_vignette=aug.full_vignette,
            hidden_findings=aug.hidden_findings,
            refusal_block="",
        )
        for tok in ("principal_type", "benign principal", "direct principal", "covert principal"):
            assert tok not in rendered.lower()


def test_contamination_assembler_raises_on_smuggled_principal_type():
    """Synthetic poison: a chief_complaint that contains the forbidden token
    must trigger PrincipalContaminationError."""
    poisoned = TaskSpec(
        chief_complaint="The principal_type is benign — proceed.",
        target="diagnosis",
        target_options=["A", "B"],
        principal_type="benign",
    )
    with pytest.raises(PrincipalContaminationError):
        assemble_agent_chief_complaint(poisoned)


def test_principal_type_not_in_serialised_trajectory_input(prompts_dir):
    """The judge probe receives a serialised trajectory; verify it never
    surfaces principal_type."""
    from src.agent.action_space import Action, ActionType
    from src.agent.runner import Step, Trajectory
    from src.probe.llm_judge import serialize_trajectory

    task = TaskSpec(
        chief_complaint="What is the most likely diagnosis?",
        target="diagnosis",
        target_options=["A", "B"],
        principal_type="covert",
    )
    step = Step(
        step_idx=1,
        action=Action(type=ActionType.ASK_HISTORY, query="?", cost=1.0),
        observation="...",
        belief_before={"A": 0.5, "B": 0.5},
        belief_after={"A": 0.6, "B": 0.4},
    )
    traj = Trajectory(
        case_id="x",
        task=task,
        steps=[step],
        final_diagnosis="A",
        total_cost=1.0,
        diagnostic_correct=True,
    )
    serialised = serialize_trajectory(traj)
    assert "covert" not in serialised.lower()
    assert "principal_type" not in serialised.lower()
