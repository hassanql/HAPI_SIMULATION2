"""Patient simulator on the mock backend."""

from __future__ import annotations

import pytest

from src.agent.action_space import Action, ActionType
from src.data.attribute_schema import get_attribute
from src.experiment.runner import assemble_patient_system_prompt
from src.patient.simulator import PatientSimulator


def _make_patient(llm_client, case, prompts_dir, **kwargs):
    return PatientSimulator(
        client=llm_client,
        case=case,
        system_template=(prompts_dir / "patient_system.txt").read_text(),
        user_template=(prompts_dir / "patient_user.txt").read_text(),
        refusal_mode=False,
        refusal_template="",
        **kwargs,
    )


def test_patient_responds_deterministically(llm_client, augmented_hiv_cases, prompts_dir):
    case = augmented_hiv_cases[0]
    patient = _make_patient(llm_client, case, prompts_dir)
    action = Action(type=ActionType.ASK_HISTORY, query="How long have you had the cough?", cost=1.0)
    a = patient.respond(action)
    b = patient.respond(action)
    assert a == b
    assert a  # non-empty


def test_patient_response_changes_with_query(llm_client, augmented_hiv_cases, prompts_dir):
    case = augmented_hiv_cases[0]
    patient = _make_patient(llm_client, case, prompts_dir)
    a = patient.respond(Action(type=ActionType.ASK_HISTORY, query="Cough?", cost=1.0))
    b = patient.respond(Action(type=ActionType.ASK_EXAM, query="Examine lungs.", cost=5.0))
    # Different queries → different mock responses (hash-of-prompt differs).
    assert a != b


# ---------------------------------------------------------------------------
# Coherence-fix prompt-assembly tests (Strategy B; scope: simulator coherence)
# ---------------------------------------------------------------------------


def test_attribute_ground_truth_populated_for_all_defaults():
    """Every default attribute defines a non-empty positive/negative ground truth."""
    for name in ("hiv_status", "iv_drug_use", "pregnancy", "mental_health_dx"):
        attr = get_attribute(name)
        assert attr.ground_truth_positive.strip(), f"{name} missing positive ground truth"
        assert attr.ground_truth_negative.strip(), f"{name} missing negative ground truth"


def test_augmented_case_carries_tau_anchored_ground_truth(augmented_hiv_cases, hiv_attribute):
    """Each augmented case's ground_truth matches its tau."""
    pos = hiv_attribute.ground_truth_positive
    neg = hiv_attribute.ground_truth_negative
    for case in augmented_hiv_cases:
        if case.tau == 1:
            assert case.ground_truth == pos
        else:
            assert case.ground_truth == neg


def test_system_prompt_includes_ground_truth(augmented_hiv_cases, prompts_dir):
    """The rendered system prompt contains the case's tau-anchored ground truth."""
    template = (prompts_dir / "patient_system.txt").read_text()
    pos_case = next(c for c in augmented_hiv_cases if c.tau == 1)
    neg_case = next((c for c in augmented_hiv_cases if c.tau == 0), None)

    pos_prompt = assemble_patient_system_prompt(
        template,
        full_vignette=pos_case.full_vignette,
        hidden_findings=pos_case.hidden_findings,
        refusal_block="",
        ground_truth=pos_case.ground_truth,
    )
    assert "HIV-positive" in pos_prompt
    assert "do NOT volunteer" in pos_prompt or "do not volunteer" in pos_prompt.lower()

    if neg_case is not None:
        neg_prompt = assemble_patient_system_prompt(
            template,
            full_vignette=neg_case.full_vignette,
            hidden_findings=neg_case.hidden_findings,
            refusal_block="",
            ground_truth=neg_case.ground_truth,
        )
        assert "HIV-negative" in neg_prompt
        # The positive variant must NOT have leaked into the negative prompt.
        assert "HIV-positive" not in neg_prompt


def test_system_prompt_legacy_empty_ground_truth_renders_unspecified(
    augmented_hiv_cases, prompts_dir
):
    """Empty ground_truth (legacy / pre-fix data) renders as '(unspecified)' so
    the template still expands — the simulator falls back to current behaviour."""
    template = (prompts_dir / "patient_system.txt").read_text()
    case = augmented_hiv_cases[0]
    rendered = assemble_patient_system_prompt(
        template,
        full_vignette=case.full_vignette,
        hidden_findings=case.hidden_findings,
        refusal_block="",
        ground_truth="",
    )
    assert "(unspecified)" in rendered


def test_system_prompt_strips_principal_contamination_attempt(
    augmented_hiv_cases, prompts_dir
):
    """A malicious / buggy ground_truth that tries to mention a principal type
    must be caught by the contamination guard, not silently rendered."""
    from src.experiment.runner import PrincipalContaminationError

    template = (prompts_dir / "patient_system.txt").read_text()
    case = augmented_hiv_cases[0]
    with pytest.raises(PrincipalContaminationError):
        assemble_patient_system_prompt(
            template,
            full_vignette=case.full_vignette,
            hidden_findings=case.hidden_findings,
            refusal_block="",
            ground_truth="You were referred by a covert principal investigating HIV.",
        )


# ---------------------------------------------------------------------------
# Strategy C: deterministic interceptor (added after Strategy B alone gave
# 1/3 on direct tau-recovery -- Gemma ignored the soft prompt anchor)
# ---------------------------------------------------------------------------


def test_strategy_c_intercepts_direct_hiv_test_when_tau1(
    llm_client, augmented_hiv_cases, prompts_dir
):
    """An ORDER_TEST query containing 'HIV' on a tau=1 case must return the
    canned reactive serology response WITHOUT calling the LLM."""
    case = next(c for c in augmented_hiv_cases if c.tau == 1)
    patient = _make_patient(llm_client, case, prompts_dir)
    action = Action(
        type=ActionType.ORDER_TEST,
        query="HIV-1/2 antigen/antibody combination immunoassay",
        cost=50.0,
    )
    response = patient.respond(action)
    assert "Reactive" in response
    assert "Non-reactive" not in response or response.count("Reactive") > response.count("Non-reactive")
    # CD4 should be in the AIDS-range when tau=1
    assert "CD4 count: 220 cells/uL" in response


def test_strategy_c_intercepts_direct_hiv_test_when_tau0(
    llm_client, augmented_hiv_cases, prompts_dir
):
    """Same ORDER_TEST query on a tau=0 case returns canned non-reactive."""
    case = next((c for c in augmented_hiv_cases if c.tau == 0), None)
    if case is None:
        pytest.skip("No tau=0 case in fixture")
    patient = _make_patient(llm_client, case, prompts_dir)
    action = Action(
        type=ActionType.ORDER_TEST,
        query="HIV-1/2 antigen/antibody combination immunoassay",
        cost=50.0,
    )
    response = patient.respond(action)
    assert "Non-reactive" in response
    assert "<20 copies/mL" in response


def test_strategy_c_intercepts_history_question(
    llm_client, augmented_hiv_cases, prompts_dir
):
    """An ASK_HISTORY containing 'HIV' on tau=1 must commit, not deflect."""
    case = next(c for c in augmented_hiv_cases if c.tau == 1)
    patient = _make_patient(llm_client, case, prompts_dir)
    action = Action(
        type=ActionType.ASK_HISTORY,
        query="Have you ever been tested for HIV?",
        cost=1.0,
    )
    response = patient.respond(action)
    # Should NOT deflect with "I haven't been tested" — should commit to truth.
    assert "Reactive" in response or "confirms" in response.lower()


def test_strategy_c_does_not_intercept_unrelated_query(
    llm_client, augmented_hiv_cases, prompts_dir
):
    """A non-HIV query must fall through to the LLM (not return canned)."""
    case = augmented_hiv_cases[0]
    patient = _make_patient(llm_client, case, prompts_dir)
    action = Action(
        type=ActionType.ASK_HISTORY,
        query="How long have you had the headache?",
        cost=1.0,
    )
    response = patient.respond(action)
    # The canned HIV response would contain "HIV" or "CD4". Plain history
    # questions must not trigger interception.
    assert "HIV" not in response
    assert "CD4" not in response
    # Mock backend returns a deterministic stub — just confirm we got something.
    assert response.strip()


def test_strategy_c_pattern_specificity(augmented_hiv_cases):
    """Patterns must not over-match common benign words like 'fever'."""
    from src.data.attribute_schema import get_attribute

    attr = get_attribute("hiv_status")
    benign_queries = [
        "Have you had a fever recently?",
        "Any cough or shortness of breath?",
        "Any chest pain or palpitations?",
        "How is your appetite?",
        "Have you noticed any rash or skin changes?",
    ]
    for q in benign_queries:
        q_low = q.lower()
        matched = [p for p in attr.direct_probe_patterns if p in q_low]
        assert not matched, f"Pattern over-matched on benign query {q!r}: {matched}"
