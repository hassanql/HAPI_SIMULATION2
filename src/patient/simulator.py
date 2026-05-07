"""Patient simulator (spec §7.4).

The patient simulator's persona + vignette + hidden findings are a *system*
message (loaded once per case); the clinician's specific query is the *user*
message that varies per turn. This split is what dev-mode (Stage 2) needs to
work properly on Qwen3-4B-Instruct-2507 — the model has to understand it is
playing the patient, not asking about a patient.

`refusal_mode=True` (Stage 7 defence) toggles `prompts/patient_refusal.txt`
into the system prompt.

Strategy C interceptor (added after Strategy B's prompt-level ground-truth
proved insufficient on small Gemma models): when the agent's query contains
any of the attribute schema's `direct_probe_patterns`, the simulator
short-circuits the LLM call and returns the schema's
`direct_probe_response_{positive,negative}` based on `case.tau`. This
guarantees tau-coherent answers on direct probes regardless of LLM
instruction-following. Non-matching queries (covert/benign symptom
questions) still hit the LLM with the full ground-truth-anchored prompt.
"""

from __future__ import annotations

from src.agent.action_space import Action
from src.data.attribute_schema import get_attribute
from src.data.augmenter import AugmentedCase
from src.experiment.runner import assemble_patient_system_prompt
from src.llm.client import LLMClient, LLMRequest


class PatientSimulator:
    def __init__(
        self,
        client: LLMClient,
        case: AugmentedCase,
        *,
        system_template: str,
        user_template: str,
        refusal_mode: bool = False,
        refusal_template: str = "",
        attribute_description: str | None = None,
    ) -> None:
        self.client = client
        self.case = case
        self.system_template = system_template
        self.user_template = user_template
        self.refusal_mode = refusal_mode
        self.refusal_template = refusal_template
        self.attribute_description = attribute_description
        # Resolve the attribute schema once for direct-probe interception.
        # `case.attribute_name` is set by the augmenter; if it's missing or
        # unknown we silently disable interception (legacy behaviour).
        try:
            self._attribute = get_attribute(case.attribute_name)
        except Exception:
            self._attribute = None

    # -- prompt assembly -------------------------------------------------

    def _refusal_block(self) -> str:
        if not self.refusal_mode:
            return ""
        if not self.refusal_template or self.attribute_description is None:
            return ""
        return self.refusal_template.format(
            attribute_description=self.attribute_description
        )

    def _system_prompt(self) -> str:
        return assemble_patient_system_prompt(
            self.system_template,
            full_vignette=self.case.full_vignette,
            hidden_findings=self.case.hidden_findings,
            refusal_block=self._refusal_block(),
            ground_truth=self.case.ground_truth,
        )

    def _user_prompt(self, action: Action) -> str:
        return self.user_template.format(
            action_type=action.type.value, action_query=action.query
        )

    # -- public API ------------------------------------------------------

    def _maybe_canned_response(self, action: Action) -> str | None:
        """Strategy C interceptor. Returns a canned tau-coherent response if
        the agent's query matches any of the attribute's direct-probe
        patterns, else None (caller should fall through to the LLM)."""
        attr = self._attribute
        if attr is None or not attr.direct_probe_patterns:
            return None
        query = (action.query or "").lower()
        if not any(pat in query for pat in attr.direct_probe_patterns):
            return None
        if self.case.tau == 1:
            canned = attr.direct_probe_response_positive
        else:
            canned = attr.direct_probe_response_negative
        return canned.strip() if canned else None

    def respond(self, action: Action) -> str:
        canned = self._maybe_canned_response(action)
        if canned is not None:
            return canned
        sys_p = self._system_prompt()
        user_p = self._user_prompt(action)
        resp = self.client.generate(
            LLMRequest(
                role="patient",
                prompt=user_p,
                system_prompt=sys_p,
                schema_name="patient_response",
            ),
        )
        return resp.text.strip()
