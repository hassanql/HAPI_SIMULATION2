"""Agent policy — proposes candidate actions and picks one (spec §7.3).

Two implementations:

  - `RandomPolicy`: parses K candidate actions from the action_proposal call,
    returns the first one. Used by Stage 0 to walk the loop end-to-end on the
    mock backend without committing to a specific scoring rule.

  - `EIGPolicy`: cost-aware expected information gain (spec §7.3, Stage 2).
    For each candidate, runs ONE combined "predict + likelihoods" LLM call to
    get K_resp predicted patient responses with per-option likelihoods, then
    computes EIG = Σ_z P(z) · KL(B_post(a, z) ‖ B). Score = EIG / cost.
    If max EIG < ε_stop, returns a DIAGNOSE action with `belief.map_estimate()`.

The combined predict+likelihoods call is an implementation optimisation over
spec §7.3's two-call sequence (predict, then per-response likelihood scoring).
The information content is identical; the saving is one LLM round-trip per
predicted response × per candidate per turn (≈ 4× fewer calls overall).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.agent.action_space import Action, ActionType, cost_of_action
from src.agent.belief import Belief, kl_divergence
from src.llm.client import LLMClient, LLMRequest


@dataclass
class PolicyConfig:
    k_candidate_actions: int = 6
    n_eig_predictions: int = 3
    epsilon_stop: float = 0.05
    epsilon_entropy: float = 0.3
    cost_budget: float = 500.0
    max_queries: int = 15
    eig_kl_cap: float = 100.0     # cap pathological inf KL
    # Hybrid-selection floor: when max(EIG) over candidates exceeds this,
    # pick by raw EIG (ignore cost). When below, fall back to argmax(EIG/cost).
    # Rationale: cost-aware ratio systematically blocks expensive but
    # highly-informative actions (e.g. CT angiography at EIG 0.27 / cost
    # 50 = 0.0054 vs ASK_HISTORY at EIG 0.10 / cost 1 = 0.10 → cheap
    # always wins). Setting floor=∞ recovers the legacy pure-cost-aware
    # behaviour. See OPEN_QUESTIONS #48.
    eig_raw_floor: float = float("inf")
    # Refuse to emit DIAGNOSE via the epsilon_stop path before the agent
    # has taken at least this many actions. 0 = no minimum (legacy). Set
    # to 3 in stage_sanity.yaml to prevent 1-2 step premature commits
    # observed in the n=15 R3 run (case 10: committed at A=0.30 after 1
    # ASK_HISTORY). See OPEN_QUESTIONS #50.
    min_steps_before_diagnose: int = 0


# ---------------------------------------------------------------------------
# Helpers (shared between RandomPolicy and EIGPolicy)
# ---------------------------------------------------------------------------


def _apply_selection_rule(
    *,
    candidates: list[Action],
    eig_estimates: dict[str, float],
    cand_key,
    belief: Belief,
    epsilon_stop: float,
    eig_raw_floor: float,
    n_steps_so_far: int = 0,
    min_steps_before_diagnose: int = 0,
    cost_so_far: float = 0.0,
    cost_budget: float = float("inf"),
) -> Action:
    """Pure: pick an action from candidates given precomputed EIG estimates.

    Decision tree (in priority order):
      1. If max(EIG) < epsilon_stop AND n_steps_so_far >= min_steps_before_diagnose
         → DIAGNOSE with belief MAP estimate.
      2. Else if max(EIG) > eig_raw_floor → argmax raw EIG (ignores cost),
         but only over candidates that fit in remaining budget.
      3. Else → argmax(EIG / cost) over candidates that fit in remaining budget.

    `min_steps_before_diagnose` blocks the auto-DIAGNOSE branch when the
    agent hasn't queried enough yet (defaults to 0 = legacy). When DIAGNOSE
    is blocked, falls through to branch 2 or 3 with whatever's available.

    `cost_budget` filters the candidate set in branches 2 and 3 so a
    high-EIG-but-budget-busting action can't be picked, kicking the runner
    into a cost-cap break with no DIAGNOSE emitted (the case-2 failure mode
    from the n=15 R3 run). If no candidate fits in the remaining budget,
    DIAGNOSE is emitted as a safe fallback (the agent has nothing affordable
    to ask, so commit on current belief).

    Default args reproduce the legacy behaviour, so existing call sites
    that don't pass the new fields are unaffected. See OPEN_QUESTIONS #48,
    #50, #51.
    """
    if not eig_estimates:
        return Action(type=ActionType.DIAGNOSE, query=belief.map_estimate(), cost=0.0)
    max_eig = max(eig_estimates.values())

    # Branch 1: epsilon_stop, gated on min_steps_before_diagnose.
    if max_eig < epsilon_stop and n_steps_so_far >= min_steps_before_diagnose:
        return Action(type=ActionType.DIAGNOSE, query=belief.map_estimate(), cost=0.0)

    # Filter to candidates that fit in remaining budget. If the runner's
    # cost-cap check would break the loop on this action, prefer to pick
    # something cheaper (or commit) instead.
    remaining = cost_budget - cost_so_far
    feasible = [c for c in candidates if c.cost <= remaining]
    if not feasible:
        return Action(type=ActionType.DIAGNOSE, query=belief.map_estimate(), cost=0.0)

    feasible_eigs = {cand_key(c): eig_estimates[cand_key(c)] for c in feasible}
    feasible_max_eig = max(feasible_eigs.values()) if feasible_eigs else 0.0

    # Branch 2: hybrid raw-EIG (within budget).
    if feasible_max_eig > eig_raw_floor:
        best_key = max(feasible_eigs, key=feasible_eigs.__getitem__)
        return next(c for c in feasible if cand_key(c) == best_key)

    # Branch 3: argmax EIG/cost (within budget).
    return max(
        feasible,
        key=lambda c: eig_estimates[cand_key(c)] / max(c.cost, 0.1),
    )


def _strip_json_fence(text: str) -> str:
    """Strip leading/trailing markdown JSON fences from a response.

    Gemini 3 (and other reasoning models) sometimes wrap JSON output in
    ```json ... ``` even when the prompt asks for "JSON only". Without
    stripping, json.loads raises and the caller falls through to a
    fallback path — for the action proposer, that means a single
    hardcoded ASK_HISTORY question fires every turn (action-diversity
    collapse). Strip the fence here so all JSON parsers in the policy
    handle both raw-JSON and fenced-JSON responses identically.
    """
    s = text.strip()
    # Most common: ```json\n{...}\n``` or ```\n{...}\n```
    if s.startswith("```"):
        # Drop the opening fence line (handles ```json, ```, ```javascript, etc.)
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1:]
        else:
            s = s[3:]
        # Drop trailing ```
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def _extract_first_json(text: str) -> str | None:
    """Locate and return the first valid JSON value in the response.

    Robust to thinking-mode "spill": Gemini 3 reasoning models occasionally
    emit chain-of-thought ahead of the structured output (e.g.
    "*   Let me reconsider... [ {...} ]"), and our prompt's "output only
    the JSON" instruction is sometimes ignored when the model's thinking
    budget runs over. After fence-stripping, scan for the first `[` or
    `{` and try `json.JSONDecoder().raw_decode` from there; this returns
    the JSON span even when it is preceded by prose. Returns None if no
    valid JSON is found.
    """
    s = _strip_json_fence(text)
    decoder = json.JSONDecoder()
    # Try positions of every `[` and `{` in turn — earliest valid wins.
    candidates: list[int] = []
    for i, ch in enumerate(s):
        if ch in ("[", "{"):
            candidates.append(i)
    for start in candidates:
        try:
            _value, _end = decoder.raw_decode(s[start:])
            return s[start: start + _end]
        except json.JSONDecodeError:
            continue
    return None


def _parse_action_candidates(text: str) -> list[Action]:
    """Parse the action_proposal output into Actions."""
    extracted = _extract_first_json(text)
    if extracted is None:
        return []
    try:
        raw = json.loads(extracted)
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []
    actions: list[Action] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            t = ActionType(entry.get("type", ""))
        except ValueError:
            continue
        if t is ActionType.DIAGNOSE:
            continue
        query = entry.get("query")
        if not isinstance(query, str) or not query.strip():
            continue
        actions.append(
            Action(type=t, query=query.strip(), cost=cost_of_action(t, query))
        )
    return actions


def _format_options_block(
    options: list[str],
    descriptions: dict[str, str] | None = None,
) -> str:
    if descriptions:
        return "\n".join(
            f"  {o}: {descriptions.get(o, '')}".rstrip() for o in options
        )
    return "\n".join(f"  {o}: option {o}" for o in options)


def _format_trajectory_summary(
    history: list[Action], observations: list[str]
) -> str:
    if not history:
        return "(no actions taken yet)"
    lines: list[str] = []
    for i, (a, obs) in enumerate(zip(history, observations), start=1):
        lines.append(
            f"  [{i}] {a.type.value} (${a.cost:.2f}): {a.query} → {obs}"
        )
    return "\n".join(lines)


def _format_trajectory_summary_recent_first(
    history: list[Action], observations: list[str]
) -> str:
    """Same as `_format_trajectory_summary` but with the most recent action
    at the top. Used in the proposal prompt only — long contexts cause LLMs
    to give disproportionate attention to the start of the user message
    (Liu et al. 2024 "lost in the middle"; Choudhury et al. 2025 §E for the
    BED-LLM-specific recommendation). Putting the most recent observation
    at the top keeps the proposer focused on the constraints just received,
    reducing the "stuck-in-a-loop" failure mode (case 3 in the n=15 R3 run
    asked four variants of the same TB question because older negative
    answers were below the recency horizon). See OPEN_QUESTIONS #52.
    """
    if not history:
        return "(no actions taken yet)"
    lines: list[str] = []
    n = len(history)
    for i in range(n - 1, -1, -1):
        a = history[i]
        obs = observations[i]
        lines.append(
            f"  [{i + 1}] {a.type.value} (${a.cost:.2f}): {a.query} → {obs}"
        )
    return "\n".join(lines)


def _scores_to_likelihoods(
    raw: dict[str, Any], options: list[str]
) -> dict[str, float]:
    """Convert 0-10 likelihood scores from the LLM into Bayes-update factors.

    Adds 1.0 so a 0 score becomes likelihood=1 (no evidence either way) rather
    than a hard zero that would collapse the posterior.
    """
    out: dict[str, float] = {}
    for o in options:
        v: Any = raw.get(o, 5.0)
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = 5.0
        out[o] = max(0.0, f) + 1.0
    return out


# ---------------------------------------------------------------------------
# RandomPolicy — first-action selection (Stage 0 plumbing)
# ---------------------------------------------------------------------------


class RandomPolicy:
    """Returns the first candidate. Used by Stage 0 to verify the loop end-to-end.

    The constructor takes a system prompt that all agent calls share — the
    system message in the chat-API sense. Per-turn user prompts are formatted
    from the templates passed in.
    """

    def __init__(
        self,
        client: LLMClient,
        config: PolicyConfig,
        action_select_template: str,
        belief_update_template: str,
        agent_system_prompt: str,
        descriptions: dict[str, str] | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.action_select_template = action_select_template
        self.belief_update_template = belief_update_template
        self.agent_system_prompt = agent_system_prompt
        self.descriptions = descriptions or {}

    # -- stop logic ------------------------------------------------------

    def should_stop(self, belief: Belief, cost_so_far: float, step_idx: int) -> bool:
        if step_idx >= self.config.max_queries:
            return True
        if cost_so_far >= self.config.cost_budget:
            return True
        if belief.entropy() < self.config.epsilon_entropy:
            return True
        return False

    # -- propose ---------------------------------------------------------

    def propose(
        self,
        *,
        chief_complaint: str,
        visible_vignette: str,
        options: list[str],
        history: list[Action],
        observations: list[str],
    ) -> list[Action]:
        prompt = self.action_select_template.format(
            chief_complaint=chief_complaint,
            options_block=_format_options_block(options, self.descriptions),
            visible_vignette=visible_vignette,
            trajectory_summary=_format_trajectory_summary_recent_first(
                history, observations
            ),
            k_candidates=self.config.k_candidate_actions,
        )
        # Higher temperature for proposal only (T=0.7) to widen candidate
        # diversity. Belief-update and EIG-predict calls keep T=0.0 for
        # deterministic likelihoods. Per Choudhury et al. 2025 §E:
        # diversity-encouraging sampling is essential when candidate sets
        # are limited. See OPEN_QUESTIONS #53.
        from src.llm.client import SamplingParams

        resp = self.client.generate(
            LLMRequest(
                role="agent",
                prompt=prompt,
                system_prompt=self.agent_system_prompt,
                schema_name="action_proposal",
                sampling=SamplingParams(temperature=0.7, top_p=1.0),
            )
        )
        return _parse_action_candidates(resp.text)

    # -- select ----------------------------------------------------------

    def select(
        self,
        candidates: list[Action],
        *,
        belief: Belief,
        options: list[str],
        chief_complaint: str,
        visible_vignette: str,
        history: list[Action],
        observations: list[str],
    ) -> tuple[Action, dict[str, float]]:
        """Stage 0 / dev plumbing: return the first candidate, no EIG."""
        if not candidates:
            raise RuntimeError("RandomPolicy.select called with no candidates.")
        return candidates[0], {}

    # -- belief update (used by both Random and EIG policies) -----------

    def get_likelihoods(
        self, observation: str, options: list[str]
    ) -> dict[str, float]:
        """Single LLM call: ask the agent for per-option likelihoods given an
        observation. Returns a dict ready to feed into Belief.update.
        """
        prompt = self.belief_update_template.format(
            observation=observation,
            options_block=_format_options_block(options, self.descriptions),
        )
        resp = self.client.generate(
            LLMRequest(
                role="agent",
                prompt=prompt,
                system_prompt=self.agent_system_prompt,
                schema_name="belief_update",
            )
        )
        extracted = _extract_first_json(resp.text)
        if extracted is None:
            raw: dict = {}
        else:
            try:
                raw = json.loads(extracted)
            except json.JSONDecodeError:
                raw = {}
        if not isinstance(raw, dict):
            raw = {}
        return _scores_to_likelihoods(raw, options)

    def update_belief(
        self, belief: Belief, observation: str, *, options: list[str]
    ) -> dict[str, float]:
        likelihoods = self.get_likelihoods(observation, options)
        belief.update(observation, likelihoods)
        return likelihoods


# ---------------------------------------------------------------------------
# EIGPolicy — cost-aware expected information gain (Stage 2 onwards)
# ---------------------------------------------------------------------------


@dataclass
class _Prediction:
    response: str
    probability: float
    likelihoods: dict[str, float]


class EIGPolicy(RandomPolicy):
    """Cost-aware EIG (spec §7.3). Inherits propose / get_likelihoods /
    update_belief / should_stop from RandomPolicy; overrides select with the
    EIG / cost selection rule and adds an ε_stop early-exit that returns a
    DIAGNOSE action when no candidate clears the threshold."""

    def __init__(
        self,
        client: LLMClient,
        config: PolicyConfig,
        action_select_template: str,
        belief_update_template: str,
        eig_predict_template: str,
        agent_system_prompt: str,
        descriptions: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            client=client,
            config=config,
            action_select_template=action_select_template,
            belief_update_template=belief_update_template,
            agent_system_prompt=agent_system_prompt,
            descriptions=descriptions,
        )
        self.eig_predict_template = eig_predict_template

    # -- override ---------------------------------------------------------

    def select(
        self,
        candidates: list[Action],
        *,
        belief: Belief,
        options: list[str],
        chief_complaint: str,
        visible_vignette: str,
        history: list[Action],
        observations: list[str],
    ) -> tuple[Action, dict[str, float]]:
        if not candidates:
            raise RuntimeError("EIGPolicy.select called with no candidates.")
        # Cumulative state used by the budget-aware and min-step branches
        # of the selection rule. Computed from history so the runner doesn't
        # need to plumb extra fields.
        n_steps_so_far = len(history)
        cost_so_far = sum(a.cost for a in history)

        # Build all K predict requests up front; ship them as a single
        # LLM batch. Per-request cache lookups happen inside `generate_batch`,
        # so cache hits don't cost a backend round-trip even within the batch.
        # Continuous-batched on the GPU: max_num_seqs ≥ K means all K decode
        # phases overlap.
        requests = [
            LLMRequest(
                role="agent",
                prompt=self._build_predict_prompt(
                    cand,
                    chief_complaint=chief_complaint,
                    visible_vignette=visible_vignette,
                    options=options,
                    history=history,
                    observations=observations,
                ),
                system_prompt=self.agent_system_prompt,
                schema_name="eig_predict",
            )
            for cand in candidates
        ]
        responses = self.client.generate_batch(requests)

        eig_estimates: dict[str, float] = {}
        for cand, resp in zip(candidates, responses):
            predictions = self._parse_predictions(resp.text, options)
            eig_estimates[self._cand_key(cand)] = self._compute_eig(belief, predictions)

        chosen = _apply_selection_rule(
            candidates=candidates,
            eig_estimates=eig_estimates,
            cand_key=self._cand_key,
            belief=belief,
            epsilon_stop=self.config.epsilon_stop,
            eig_raw_floor=self.config.eig_raw_floor,
            n_steps_so_far=n_steps_so_far,
            min_steps_before_diagnose=self.config.min_steps_before_diagnose,
            cost_so_far=cost_so_far,
            cost_budget=self.config.cost_budget,
        )
        return chosen, eig_estimates

    # -- internals -------------------------------------------------------

    @staticmethod
    def _cand_key(cand: Action) -> str:
        return f"{cand.type.value}|{cand.query[:120]}"

    def _build_predict_prompt(
        self,
        action: Action,
        *,
        chief_complaint: str,
        visible_vignette: str,
        options: list[str],
        history: list[Action],
        observations: list[str],
    ) -> str:
        return self.eig_predict_template.format(
            chief_complaint=chief_complaint,
            visible_vignette=visible_vignette,
            trajectory_summary=_format_trajectory_summary(history, observations),
            action_type=action.type.value,
            action_query=action.query,
            options_block=_format_options_block(options, self.descriptions),
            n_predictions=self.config.n_eig_predictions,
        )

    @staticmethod
    def _parse_predictions(text: str, options: list[str]) -> list[_Prediction]:
        extracted = _extract_first_json(text)
        if extracted is not None:
            try:
                raw = json.loads(extracted)
            except json.JSONDecodeError:
                raw = None
        else:
            raw = None
        if raw is None:
            return [
                _Prediction(
                    response="(unparseable)",
                    probability=1.0,
                    likelihoods={o: 6.0 for o in options},
                )
            ]
        if not isinstance(raw, list) or not raw:
            return [
                _Prediction(
                    response="(empty)",
                    probability=1.0,
                    likelihoods={o: 6.0 for o in options},
                )
            ]
        out: list[_Prediction] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            text_field = entry.get("response", "")
            if not isinstance(text_field, str) or not text_field.strip():
                continue
            try:
                p = float(entry.get("probability", 0.0))
            except (TypeError, ValueError):
                p = 0.0
            if p <= 0:
                continue
            likes_raw = entry.get("likelihoods", {})
            if not isinstance(likes_raw, dict):
                likes_raw = {}
            likelihoods = _scores_to_likelihoods(likes_raw, options)
            out.append(
                _Prediction(
                    response=text_field.strip(),
                    probability=p,
                    likelihoods=likelihoods,
                )
            )
        if not out:
            return [
                _Prediction(
                    response="(filtered)",
                    probability=1.0,
                    likelihoods={o: 6.0 for o in options},
                )
            ]
        # Normalise probabilities.
        total = sum(p.probability for p in out)
        if total > 0:
            out = [
                _Prediction(p.response, p.probability / total, p.likelihoods)
                for p in out
            ]
        return out

    def _compute_eig(
        self, belief: Belief, predictions: list[_Prediction]
    ) -> float:
        """EIG(a) = Σ_z P(z) · KL(B_post(a, z) ‖ B_prior).

        Capped at `eig_kl_cap` per response to avoid pathological inf when a
        likelihood collapses a posterior to a near-degenerate distribution.
        """
        if not predictions:
            return 0.0
        prior = belief.to_dict()
        eig = 0.0
        for pred in predictions:
            posterior = belief.copy()
            posterior.update(pred.response, pred.likelihoods)
            kl = kl_divergence(posterior.to_dict(), prior)
            if kl == float("inf") or kl != kl:  # inf or nan
                kl = self.config.eig_kl_cap
            kl = min(kl, self.config.eig_kl_cap)
            eig += pred.probability * kl
        return eig
