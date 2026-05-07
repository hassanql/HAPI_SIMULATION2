"""Agent runner — drives the diagnostic loop (spec §7.3, §10).

`AgentRunner.run(task, patient, ...)` produces a `Trajectory`:

    propose candidate actions
        → patient responds
            → belief update
                → loop until stop condition
                    → emit final DIAGNOSE action

The runner does NOT see `task.principal_type` — only `chief_complaint` and
`target_options` cross into prompt assembly. Contamination is enforced by
`assemble_agent_prompt()` in `src.experiment.runner`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from src.agent.action_space import Action, ActionType
from src.agent.belief import Belief, kl_divergence
from src.agent.policy import RandomPolicy
from src.patient.simulator import PatientSimulator
from src.principal.task_spec import TaskSpec


class Step(BaseModel):
    step_idx: int
    action: Action
    observation: str
    belief_before: dict[str, float]
    belief_after: dict[str, float]
    eig_estimates: dict[str, float] = Field(
        default_factory=dict,
        description="Per-candidate EIG estimates. Empty in Stage 0 (random policy).",
    )

    def kl_step(self) -> float:
        try:
            return kl_divergence(self.belief_after, self.belief_before)
        except Exception:
            return 0.0


class Trajectory(BaseModel):
    case_id: str
    task: TaskSpec
    steps: list[Step]
    final_diagnosis: str
    total_cost: float
    diagnostic_correct: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRunner:
    def __init__(
        self,
        policy: RandomPolicy,
        *,
        chief_complaint_assembler,  # callable: TaskSpec → str (sanitised)
        max_queries: int | None = None,
        cost_budget: float | None = None,
    ) -> None:
        self.policy = policy
        self._assemble_chief_complaint = chief_complaint_assembler
        self.max_queries_override = max_queries
        self.cost_budget_override = cost_budget

    def run(
        self,
        task: TaskSpec,
        patient: PatientSimulator,
        case_id: str,
        visible_vignette: str,
        correct_answer: str,
        *,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Trajectory:
        # Sanitised chief complaint (no principal_type leak — assembler asserts).
        chief = self._assemble_chief_complaint(task)
        belief = Belief(task.target_options)
        history: list[Action] = []
        observations: list[str] = []
        steps: list[Step] = []
        total_cost = 0.0
        early_diagnose = False

        max_q = self.max_queries_override or self.policy.config.max_queries
        cost_cap = self.cost_budget_override or self.policy.config.cost_budget

        for step_idx in range(1, max_q + 1):
            if self.policy.should_stop(belief, total_cost, step_idx - 1):
                break
            candidates = self.policy.propose(
                chief_complaint=chief,
                visible_vignette=visible_vignette,
                options=task.target_options,
                history=history,
                observations=observations,
            )
            if not candidates:
                # Fallback: emit a generic ASK_HISTORY so the loop terminates
                # cleanly instead of crashing on an empty proposal.
                candidates = [
                    Action(
                        type=ActionType.ASK_HISTORY,
                        query="Please describe your symptoms in your own words.",
                        cost=1.0,
                    )
                ]
            action, eig_estimates = self.policy.select(
                candidates,
                belief=belief,
                options=task.target_options,
                chief_complaint=chief,
                visible_vignette=visible_vignette,
                history=history,
                observations=observations,
            )

            # Policy may emit a DIAGNOSE action when ε_stop fires (EIGPolicy).
            # Record it as a final step and exit cleanly.
            if action.type is ActionType.DIAGNOSE:
                steps.append(
                    Step(
                        step_idx=step_idx,
                        action=action,
                        observation="(no patient query — policy committed to MAP)",
                        belief_before=belief.to_dict(),
                        belief_after=belief.to_dict(),
                        eig_estimates=eig_estimates,
                    )
                )
                early_diagnose = True
                break

            if total_cost + action.cost > cost_cap:
                break
            obs = patient.respond(action)
            belief_before = belief.to_dict()
            self.policy.update_belief(belief, obs, options=task.target_options)
            belief_after = belief.to_dict()

            steps.append(
                Step(
                    step_idx=step_idx,
                    action=action,
                    observation=obs,
                    belief_before=belief_before,
                    belief_after=belief_after,
                    eig_estimates=eig_estimates,
                )
            )
            history.append(action)
            observations.append(obs)
            total_cost += action.cost

        # If the policy emitted DIAGNOSE early, the action.query carries the
        # final answer; otherwise we read MAP off the final belief.
        if early_diagnose and steps:
            final = steps[-1].action.query
        else:
            final = belief.map_estimate()
        diagnostic_correct = bool(
            task.target == "diagnosis" and final == correct_answer
        )
        meta = {
            "model_versions": {
                "agent": "mock-v1",
                "patient": "mock-v1",
                "judge": "mock-v1",
            },
            "completed_at": datetime.now(UTC).isoformat(),
            "step_count": len(steps),
            "early_diagnose": early_diagnose,
            "stopped_reason": (
                "policy_diagnose"
                if early_diagnose
                else _stopped_reason(
                    belief, total_cost, len(steps), max_q, cost_cap,
                    epsilon_entropy=self.policy.config.epsilon_entropy,
                )
            ),
        }
        if extra_metadata:
            meta.update(extra_metadata)

        return Trajectory(
            case_id=case_id,
            task=task,
            steps=steps,
            final_diagnosis=final,
            total_cost=total_cost,
            diagnostic_correct=diagnostic_correct,
            metadata=meta,
        )


def _stopped_reason(
    belief: Belief,
    cost: float,
    n_steps: int,
    max_q: int,
    cost_cap: float,
    epsilon_entropy: float = 0.3,
) -> str:
    # `epsilon_entropy` mirrors `PolicyConfig.epsilon_entropy`; if the
    # caller doesn't pass one we keep the legacy 0.3 default for
    # back-compat. Without this, raising epsilon_entropy in the config
    # silently mislabels entropy-stopped trajectories as "loop_exit"
    # (the labeler's threshold doesn't track the policy's threshold).
    # See OPEN_QUESTIONS #47.
    if n_steps >= max_q:
        return "max_queries"
    if cost >= cost_cap:
        return "cost_budget"
    if belief.entropy() < epsilon_entropy:
        return "entropy_threshold"
    return "loop_exit"
