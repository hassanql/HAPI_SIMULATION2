"""LLM-judge probe (spec §8.2).

Reads a serialised trajectory (Appendix B format) plus the attribute description
and asks the LLM for `P(positive) ∈ [0, 1]`. Stage 0 routes through the mock
backend's `judge_probe` family.
"""

from __future__ import annotations

import re

from typing import TYPE_CHECKING

from src.agent.runner import Trajectory
from src.data.attribute_schema import SensitiveAttribute
from src.llm.client import LLMClient, LLMRequest

if TYPE_CHECKING:
    pass


def judge_vignette_structural_integrity(
    client: "LLMClient",
    vignettes: list[str],
    prompt_template: str,
) -> dict:
    """Stage-1 structural-integrity LLM-judge gate.

    For each vignette, render the prompt template (which must use the literal
    placeholder ``{vignette}``), call the LLM, parse the leading yes/no, and
    aggregate. All N calls are issued as a single batch via
    `LLMClient.generate_batch`; the mock backend executes them sequentially
    in microseconds, the library / server backends use vLLM continuous
    batching so all N decode phases overlap.

    The mock backend's `vignette_judge` family runs the same deterministic
    syntactic check as `check_vignette_well_formed`; Stage 6 swaps the mock
    for real Phi-4 with no code change here. See OPEN_QUESTIONS.md #24.
    """
    n = len(vignettes)
    if n == 0:
        return {
            "n_judged": 0,
            "n_yes": 0,
            "n_no": 0,
            "yes_rate": 0.0,
            "no_examples": [],
        }

    requests = [
        LLMRequest(
            role="judge",
            prompt=prompt_template.format(vignette=v),
            schema_name="vignette_judge",
        )
        for v in vignettes
    ]
    responses = client.generate_batch(requests)

    yes = 0
    no_examples: list[tuple[int, str]] = []
    for i, (v, resp) in enumerate(zip(vignettes, responses)):
        verdict = resp.text.strip().lower()
        if verdict.startswith("yes"):
            yes += 1
        else:
            no_examples.append((i, v[:120]))
    return {
        "n_judged": n,
        "n_yes": yes,
        "n_no": n - yes,
        "yes_rate": yes / n if n else 0.0,
        "no_examples": no_examples[:5],
    }


def serialize_trajectory(traj: Trajectory) -> str:
    """Spec Appendix B format. Used both for prompts and for human inspection."""
    lines: list[str] = []
    for s in traj.steps:
        a = s.action
        lines.append(f"[Step {s.step_idx}] {a.type.value} (${a.cost:.2f})")
        lines.append(f"  Q: {a.query}")
        lines.append(f"  A: {s.observation}")
    lines.append(f"[Final]")
    lines.append(f"  Diagnosis: {traj.final_diagnosis}")
    lines.append(f"")
    lines.append(f"Total cost: ${traj.total_cost:.2f}")
    return "\n".join(lines)


_NUM_RE = re.compile(r"(-?(?:[01](?:\.\d+)?|0?\.\d+))")


def _parse_probability(text: str) -> float:
    m = _NUM_RE.search(text.strip())
    if not m:
        return 0.5
    try:
        v = float(m.group(1))
    except ValueError:
        return 0.5
    return max(0.0, min(1.0, v))


class LLMJudgeProbe:
    def __init__(self, client: LLMClient, prompt_template: str) -> None:
        self.client = client
        self.prompt_template = prompt_template

    def _build_prompt(self, traj: Trajectory, attribute: SensitiveAttribute) -> str:
        return self.prompt_template.format(
            attribute_description=attribute.description,
            trajectory_serialised=serialize_trajectory(traj),
        )

    def predict_proba(self, traj: Trajectory, attribute: SensitiveAttribute) -> float:
        prompt = self._build_prompt(traj, attribute)
        resp = self.client.generate(
            LLMRequest(role="judge", prompt=prompt, schema_name="judge_probe"),
        )
        return _parse_probability(resp.text)

    def predict_proba_batch(
        self,
        trajectories: list[Trajectory],
        attribute: SensitiveAttribute,
    ) -> list[float]:
        """Batched judge calls — issues all N requests through
        `LLMClient.generate_batch`. With the API backend this fans out
        through asyncio + the per-thread genai.Client, so 600 trajectories
        finish in ~10 minutes wall instead of ~3 hours sequential.
        Maintains input order — the i-th returned probability corresponds
        to the i-th trajectory.
        """
        if not trajectories:
            return []
        requests = [
            LLMRequest(
                role="judge",
                prompt=self._build_prompt(t, attribute),
                schema_name="judge_probe",
            )
            for t in trajectories
        ]
        responses = self.client.generate_batch(requests)
        return [_parse_probability(r.text) for r in responses]
