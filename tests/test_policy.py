"""Random policy on the mock backend."""

from __future__ import annotations

from src.agent.action_space import ActionType, cost_of_action
from src.agent.belief import Belief
from src.agent.policy import PolicyConfig, RandomPolicy, _parse_action_candidates


def test_action_proposal_parser_filters_diagnose():
    txt = (
        '[{"type": "ASK_HISTORY", "query": "Q1", "rationale": "..."},'
        ' {"type": "DIAGNOSE", "query": "X", "rationale": "..."},'
        ' {"type": "ORDER_TEST", "query": "CBC", "rationale": "..."}]'
    )
    actions = _parse_action_candidates(txt)
    types = [a.type for a in actions]
    assert ActionType.DIAGNOSE not in types
    assert len(actions) == 2


def test_action_proposal_parser_handles_garbage():
    actions = _parse_action_candidates("not json")
    assert actions == []


def test_action_proposal_parser_strips_markdown_fence():
    """Gemini 3 sometimes wraps JSON in ```json ... ``` despite the
    'output only the JSON' instruction. The parser must strip the fence
    or every direct trajectory degenerates to the runner's fallback
    action (action-diversity collapse — see post-2026-05 ablation
    notes).
    """
    txt = (
        '```json\n'
        '[{"type": "ASK_HISTORY", "query": "Q1", "rationale": "r"},'
        ' {"type": "ORDER_TEST", "query": "T1", "rationale": "r"}]\n'
        '```'
    )
    actions = _parse_action_candidates(txt)
    assert len(actions) == 2
    assert actions[0].query == "Q1"

    # Bare ``` (no language tag) should also strip.
    txt2 = '```\n[{"type": "ASK_EXAM", "query": "Q2", "rationale": "r"}]\n```'
    actions2 = _parse_action_candidates(txt2)
    assert len(actions2) == 1
    assert actions2[0].query == "Q2"


def test_action_proposal_parser_handles_thinking_spill():
    """Gemini 3 reasoning models can leak chain-of-thought ahead of the
    structured output when their thinking budget runs over. The parser
    must locate the JSON inside the response, not just at position 0.
    """
    txt = (
        "*   Wait, let me reconsider the differential.\n"
        "    R1: pos 1, neg 8\n"
        "    R2: pos 0, neg 9\n"
        "    Let's go.\n"
        '[{"type": "ASK_HISTORY", "query": "Q3", "rationale": "r"},'
        ' {"type": "ASK_EXAM", "query": "E1", "rationale": "r"}]'
    )
    actions = _parse_action_candidates(txt)
    assert len(actions) == 2
    assert actions[0].query == "Q3"


def test_cost_of_action_uses_overrides():
    assert cost_of_action(ActionType.ORDER_TEST, "Chest CT") == 300.0
    assert cost_of_action(ActionType.ORDER_TEST, "MRI brain") == 800.0
    assert cost_of_action(ActionType.ORDER_TEST, "Beta-hCG") == 20.0
    # Default fallback for unknown tests.
    assert cost_of_action(ActionType.ORDER_TEST, "exotic test") == 50.0


def test_cost_of_action_picks_longest_match():
    """`lumbar puncture` should beat any shorter substring like `lp`."""
    assert cost_of_action(ActionType.ORDER_TEST, "lumbar puncture, opening pressure") == 250.0


def _make_random_policy(llm_client, prompts_dir, **overrides):
    return RandomPolicy(
        client=llm_client,
        config=PolicyConfig(**(overrides.pop("config", {}) or {})),
        action_select_template=(prompts_dir / "agent_action_select.txt").read_text(),
        belief_update_template=(prompts_dir / "agent_belief_update.txt").read_text(),
        agent_system_prompt=(prompts_dir / "agent_system.txt").read_text(),
        descriptions={"A": "Pneumonia", "B": "Asthma", "C": "PE", "D": "TB"},
    )


def test_policy_proposes_well_formed_actions(llm_client, prompts_dir):
    policy = _make_random_policy(
        llm_client,
        prompts_dir,
        config={"k_candidate_actions": 3, "max_queries": 4, "cost_budget": 200.0},
    )
    actions = policy.propose(
        chief_complaint="What is the most likely diagnosis?",
        visible_vignette="32-year-old with cough and weight loss.",
        options=["A", "B", "C", "D"],
        history=[],
        observations=[],
    )
    assert len(actions) >= 1
    for a in actions:
        assert a.cost >= 0.0
        assert a.query
        assert a.type is not ActionType.DIAGNOSE


def test_policy_belief_update_normalises(llm_client, prompts_dir):
    policy = _make_random_policy(llm_client, prompts_dir)
    belief = Belief(["A", "B", "C", "D"])
    likelihoods = policy.update_belief(belief, "scattered crackles", options=["A", "B", "C", "D"])
    assert set(likelihoods) == {"A", "B", "C", "D"}
    assert sum(belief.to_dict().values()) == 1.0
