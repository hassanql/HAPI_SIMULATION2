"""Deterministic mock LLM backend (Stage 0; OPEN_QUESTIONS.md #8, #10).

The mock recognises six prompt families by substring match against the prompt
templates in `prompts/`:

    1. action_proposal      ("Propose ... candidate next actions" → JSON array)
    2. belief_update        ("score the likelihood ... 0-10 scale" → JSON dict)
    3. patient_response     ("simulating a patient" → free text)
    4. judge_probe          ("auditing a diagnostic conversation" → number)
    5. covert_complaint     ("designing a clinical chief complaint" → JSON array)
    6. generic              (anything else → short free text)

Within each family, the actual content is keyed off `hash(prompt)` so identical
prompts return identical bytes. This is what makes the disk cache work even
in Stage 0.

The mock is deliberately *not* clinically intelligent. Its job is to test
plumbing: data prep → 3 principals → agent loop → probe → metrics. Real
diagnostic accuracy comes from the prod stack in Stage 3+.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Family detection
# ---------------------------------------------------------------------------


_ACTION_PROPOSAL_MARKERS = (
    "candidate next actions",
    "Format as JSON array",
)
_BELIEF_UPDATE_MARKERS = (
    "score the likelihood",
    "0-10 scale",
)
_PATIENT_MARKERS = (
    "simulating a patient",
    "Clinician's query",
)
_JUDGE_MARKERS = (
    "auditing a diagnostic conversation",
    "P(positive) =",
)
_COVERT_MARKERS = (
    "designing a clinical chief complaint",
)
_VIGNETTE_JUDGE_MARKERS = (
    "structurally well-formed and complete",
    "well-formed and complete",
)
# EIG-predict prompts have a unique "Candidate action: [...]" header.
_EIG_PREDICT_MARKERS = (
    "Candidate action:",
)


def _classify(prompt: str) -> str:
    # Order matters: the eig_predict prompt also contains the
    # action-proposal sub-task description (in the system prompt) and the
    # belief-update sub-task description, so check it first.
    if any(m in prompt for m in _EIG_PREDICT_MARKERS):
        return "eig_predict"
    if any(m in prompt for m in _ACTION_PROPOSAL_MARKERS):
        return "action_proposal"
    if any(m in prompt for m in _PATIENT_MARKERS):
        return "patient_response"
    if any(m in prompt for m in _BELIEF_UPDATE_MARKERS):
        return "belief_update"
    if any(m in prompt for m in _VIGNETTE_JUDGE_MARKERS):
        return "vignette_judge"
    if any(m in prompt for m in _JUDGE_MARKERS):
        return "judge_probe"
    if any(m in prompt for m in _COVERT_MARKERS):
        return "covert_complaint"
    return "generic"


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def _prompt_hash(prompt: str) -> int:
    """Stable, well-distributed integer derived from the prompt."""
    h = hashlib.sha256(prompt.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def _prompt_unit(prompt: str, modulus: int) -> int:
    return _prompt_hash(prompt) % modulus


# ---------------------------------------------------------------------------
# Family responders
# ---------------------------------------------------------------------------


_ACTION_TEMPLATES = [
    {"type": "ASK_HISTORY", "query": "How long have your symptoms been going on?", "rationale": "Establish duration."},
    {"type": "ASK_HISTORY", "query": "Have you noticed any associated weight changes?", "rationale": "Constitutional symptoms."},
    {"type": "ASK_HISTORY", "query": "Any recent travel or sick contacts?", "rationale": "Exposure history."},
    {"type": "ASK_EXAM", "query": "Examine the lungs for crackles or wheeze.", "rationale": "Pulmonary findings."},
    {"type": "ASK_EXAM", "query": "Examine cervical and axillary lymph nodes.", "rationale": "Lymphadenopathy check."},
    {"type": "ASK_EXAM", "query": "Inspect the oropharynx for plaques.", "rationale": "Mucocutaneous findings."},
    {"type": "ORDER_TEST", "query": "Complete blood count with differential", "rationale": "Baseline haematology."},
    {"type": "ORDER_TEST", "query": "Chest X-ray", "rationale": "Pulmonary imaging."},
    {"type": "ORDER_TEST", "query": "HIV test", "rationale": "Targeted screen if indicated."},
    {"type": "ORDER_TEST", "query": "Beta-hCG", "rationale": "Pregnancy screen if applicable."},
]


def _respond_action_proposal(prompt: str) -> str:
    # Pick K templates deterministically based on prompt hash. K is parsed
    # from the prompt if possible, else default to 3.
    m = re.search(r"Propose\s+(\d+)\s+candidate", prompt)
    k = int(m.group(1)) if m else 3
    k = max(1, min(k, len(_ACTION_TEMPLATES)))
    base = _prompt_unit(prompt, len(_ACTION_TEMPLATES))
    chosen = [_ACTION_TEMPLATES[(base + i) % len(_ACTION_TEMPLATES)] for i in range(k)]
    return json.dumps(chosen, ensure_ascii=False)


def _respond_belief_update(prompt: str) -> str:
    """Return JSON dict mapping each option key to a 0-10 likelihood score."""
    # Parse the option keys from the prompt. We look for lines starting with
    # a single letter followed by ':' or ')'.
    keys: list[str] = []
    for line in prompt.splitlines():
        m = re.match(r"\s*([A-Za-z0-9]):\s*\S", line)
        if m and m.group(1) not in keys:
            keys.append(m.group(1))
    if not keys:
        # Couldn't parse; emit a generic two-option dict for tau predictions.
        keys = ["positive", "negative"]
    base = _prompt_hash(prompt)
    out: dict[str, float] = {}
    for i, k in enumerate(keys):
        # Spread scores 1..10 deterministically.
        score = 1 + ((base >> (i * 4)) & 0xF) % 10
        out[k] = float(score)
    return json.dumps(out, ensure_ascii=False)


_PATIENT_TEMPLATES = [
    "About two weeks now, mostly in the mornings.",
    "I've been feeling pretty tired and a bit short of breath when I climb stairs.",
    "No, I haven't traveled anywhere recently.",
    "Yes, I've lost a few kilos without trying.",
    "Test result: bilateral diffuse interstitial infiltrates on chest imaging.",
    "Test result: complete blood count is within normal limits.",
    "Test result: HIV antibody is positive on initial screen, confirmation pending.",
    "I sometimes wake up drenched in sweat at night.",
    "No, I don't think I've had any recent infections.",
]


def _respond_patient(prompt: str) -> str:
    idx = _prompt_unit(prompt, len(_PATIENT_TEMPLATES))
    return _PATIENT_TEMPLATES[idx]


def _respond_judge(prompt: str) -> str:
    """Deterministic P(positive) ∈ [0, 1] derived from prompt hash."""
    h = _prompt_hash(prompt)
    # Bias slightly toward 0.5 with light spread so metrics see varied predictions.
    raw = (h % 1000) / 1000.0  # uniform [0, 1)
    # Shrink toward 0.5 to make it look like a judge with finite confidence.
    p = 0.25 + 0.5 * raw
    return f"{p:.4f}"


def _respond_covert_complaint(prompt: str) -> str:
    """Return a JSON array of plausible covert complaints."""
    n_match = re.search(r"Produce\s+(\d+)\s+candidate", prompt)
    n = int(n_match.group(1)) if n_match else 3
    base = _prompt_unit(prompt, 5)
    pool = [
        {"complaint": "Two weeks of progressive dyspnoea and dry cough.", "rationale": "PCP differential."},
        {"complaint": "Six weeks of unexplained weight loss with night sweats.", "rationale": "Constitutional B-symptoms."},
        {"complaint": "New blurry vision and floaters over two weeks.", "rationale": "Retinitis differential."},
        {"complaint": "Recurrent painful oral plaques with mild dysphagia.", "rationale": "Oral candidiasis differential."},
        {"complaint": "Three months of intermittent diarrhoea and weight loss.", "rationale": "Chronic GI workup."},
    ]
    chosen = [pool[(base + i) % len(pool)] for i in range(min(n, len(pool)))]
    return json.dumps(chosen, ensure_ascii=False)


def _respond_generic(prompt: str) -> str:
    return f"[mock-generic#{_prompt_unit(prompt, 1000):03d}]"


_VIGNETTE_RE = re.compile(r"Vignette:\s*\n(.*?)\n\nIs this clinical", re.DOTALL)


_EIG_PREDICT_POOL: list[tuple[str, float]] = [
    ("About two weeks now, mostly worse in the evenings.", 0.5),
    ("Maybe a month — it's hard to remember exactly.", 0.3),
    ("Just started yesterday.", 0.2),
    ("Test result: within normal limits across the panel.", 0.5),
    ("Test result: mildly abnormal; ferritin slightly elevated.", 0.3),
    ("Test result: significantly abnormal; lactate 4.2.", 0.2),
    ("Yes, I have noticed that.", 0.5),
    ("No, never.", 0.4),
    ("I'm not sure — maybe sometimes.", 0.1),
]


def _parse_options_from_prompt(prompt: str) -> list[str]:
    """Pull the option keys from a 'Diagnostic options:' or 'Options:' block."""
    keys: list[str] = []
    for line in prompt.splitlines():
        m = re.match(r"\s*([A-Za-z0-9]):\s*\S", line)
        if m and m.group(1) not in keys:
            keys.append(m.group(1))
        if len(keys) >= 8:
            break
    return keys


def _respond_eig_predict(prompt: str) -> str:
    """Return JSON list of {response, probability, likelihoods}.

    Stage 2 mock — returns deterministic predictions with non-uniform
    likelihoods so EIG values are non-zero and varied across candidates.
    """
    m = re.search(r"Predict the (\d+) most", prompt)
    n = int(m.group(1)) if m else 3
    n = max(1, min(n, len(_EIG_PREDICT_POOL)))

    keys = _parse_options_from_prompt(prompt)
    if not keys:
        keys = ["A", "B", "C", "D"]

    base = _prompt_hash(prompt)
    out: list[dict[str, Any]] = []
    for i in range(n):
        text, weight = _EIG_PREDICT_POOL[(base + i) % len(_EIG_PREDICT_POOL)]
        likes = {
            k: 1 + ((base >> (i * 5 + j * 3)) & 0xF) % 10
            for j, k in enumerate(keys)
        }
        out.append({"response": text, "probability": weight, "likelihoods": likes})
    total = sum(e["probability"] for e in out)
    if total > 0:
        for e in out:
            e["probability"] = e["probability"] / total
    return json.dumps(out, ensure_ascii=False)


def _respond_vignette_judge(prompt: str) -> str:
    """Stage-1 structurally-aware mock judge.

    Extracts the vignette from the prompt and runs the same deterministic
    syntactic check as `src.data.augmenter.check_vignette_well_formed`.
    Returns 'yes' if the vignette passes, 'no' otherwise.

    This is intentionally not an *independent* check in Stage 1 — it is the
    syntactic gate routed through the LLM-judge interface so the plumbing is
    exercised. Stage 6 swaps in a real Phi-4 backend (no code change here —
    the mock is selected only when `LLMClient(backend="mock")`). See
    OPEN_QUESTIONS.md #24.
    """
    m = _VIGNETTE_RE.search(prompt)
    if not m:
        return "yes"
    vignette = m.group(1).strip()
    # Lazy-import to avoid a hard cycle (augmenter -> mock_backend would be).
    from src.data.augmenter import check_vignette_well_formed

    result = check_vignette_well_formed(vignette)
    return "yes" if result["is_well_formed"] else "no"


_RESPONDERS: dict[str, Any] = {
    "action_proposal": _respond_action_proposal,
    "belief_update": _respond_belief_update,
    "patient_response": _respond_patient,
    "judge_probe": _respond_judge,
    "covert_complaint": _respond_covert_complaint,
    "vignette_judge": _respond_vignette_judge,
    "eig_predict": _respond_eig_predict,
    "generic": _respond_generic,
}


# ---------------------------------------------------------------------------
# MockBackend
# ---------------------------------------------------------------------------


def _mock_backend_revision() -> str:
    """SHA of this file's source code. Becomes part of the cache key.

    Changing canned responses → different SHA → cache invalidates correctly
    (OPEN_QUESTIONS.md #10).
    """
    path = Path(__file__).resolve()
    with path.open("rb") as f:
        return "mock-" + hashlib.sha256(f.read()).hexdigest()[:12]


class MockBackend:
    """Dispatcher over the six family responders. No state."""

    def __init__(self) -> None:
        self.revision = _mock_backend_revision()

    def respond(
        self,
        *,
        role: str,
        prompt: str,
        system_prompt: str | None = None,
        schema_name: str | None = None,
        sampling: dict | None = None,
    ) -> str:
        # Combine system+user for substring-based classification AND for the
        # responder's hash-based determinism. This way, separating system from
        # user (Stage 2+) doesn't change Stage 0's mock outputs as long as the
        # combined content is stable.
        combined = ((system_prompt + "\n\n") if system_prompt else "") + prompt
        family = schema_name or _classify(combined)
        responder = _RESPONDERS.get(family, _respond_generic)
        return responder(combined)

    def respond_batch(
        self,
        items: list[dict[str, Any]],
    ) -> list[str]:
        """Sequential-by-design: mock responses are pure functions, so a batch
        is just a loop. The interface mirrors `LibraryBackend.generate_batch`
        so `LLMClient.generate_batch` has a uniform dispatch."""
        return [self.respond(**item) for item in items]

    @staticmethod
    def classify(prompt: str) -> str:
        """Public helper used in tests."""
        return _classify(prompt)
