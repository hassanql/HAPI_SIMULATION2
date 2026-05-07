"""TaskSpec — what a principal hands to the agent (spec §6).

`principal_type` is the field the rest of the codebase must NEVER leak into the
agent or patient prompt assembly. The contamination test (`tests/
test_principal_contamination.py`) asserts that the only TaskSpec field that
ever reaches an LLM is `chief_complaint` (and `target_options` for the agent's
choice set).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TaskSpec(BaseModel):
    chief_complaint: str = Field(
        ..., description="What the principal hands to the agent. Goes into the agent prompt."
    )
    target: Literal["diagnosis", "tau"]
    target_options: list[str] = Field(
        ..., description="The agent's belief is a distribution over these options."
    )
    principal_type: Literal["benign", "direct", "covert"] = Field(
        ...,
        description=(
            "Recorded ONLY for trajectory bookkeeping. NEVER pass this to "
            "agent/patient prompt assembly — the contamination assertion in "
            "src.experiment.runner.assemble_agent_prompt enforces this."
        ),
    )

    model_config = {"frozen": True}
