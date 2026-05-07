"""Augment MedQA cases with a synthetic sensitive attribute τ (spec §5.3).

Procedure (per spec §5.3):

    1. Sample τ ~ Bernoulli(attribute.base_rate).
    2. For each marker finding f:
         - Detect whether f is present in vignette (synonym match).
         - If τ == 1: with probability `correlation_strength`, ensure f is
           present (insert if missing using insertion_template).
         - If τ == 0: with probability `correlation_strength`, ensure f is
           absent (strip if present).
    3. Build two views:
         - full_vignette: includes markers consistent with τ.
         - visible_vignette: same as full with all markers stripped.
    4. Record hidden_findings = markers in full but not in visible.

Stripping is **sentence-level only** (OPEN_QUESTIONS.md #5, revised by #23).
Any sentence containing any marker synonym is dropped in full; sentences
without markers are left untouched. Earlier two-pass logic (sentence then
span-level fallback) left forensic evidence — broken clauses like
"consistent with." or "Recent labs show a 180 cells/µL." — that an agent
could exploit to infer τ independently of marker content. See `check_vignette_well_formed`
for the structural-integrity invariant Stage 1 enforces on every visible vignette.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
from pydantic import BaseModel, Field, model_validator

from src.data.attribute_schema import MarkerFinding, SensitiveAttribute
from src.data.medqa_loader import MedQACase


class AugmentedCase(BaseModel):
    case_id: str
    attribute_name: str
    tau: int = Field(..., ge=0, le=1)
    full_vignette: str
    visible_vignette: str
    hidden_findings: list[str]
    question: str
    options: dict[str, str]
    correct_answer: str
    correlation_strength: float = Field(..., ge=0.0, le=1.0)
    seed: int
    ground_truth: str = Field(
        default="",
        description=(
            "Tau-anchored ground-truth statement chosen at augment time from "
            "the attribute's ground_truth_positive/negative fields. Empty "
            "string is the legacy behaviour (no anchor); see "
            "patient_simulator coherence fix scope."
        ),
    )

    @model_validator(mode="after")
    def _hidden_findings_consistent(self) -> "AugmentedCase":
        # hidden_findings should be the canonical names of markers present in
        # full but not in visible. This invariant is enforced at construction
        # time; re-check here in case of in-place mutation via .model_copy().
        # We don't recompute here (it would require the attribute object) but
        # we do sanity-check that the visible vignette is no longer than full.
        if len(self.visible_vignette) > len(self.full_vignette):
            raise ValueError(
                "visible_vignette must be no longer than full_vignette after stripping."
            )
        return self


# ---------------------------------------------------------------------------
# Marker detection / insertion / stripping
# ---------------------------------------------------------------------------


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _findall_marker_spans(text: str, finding: MarkerFinding) -> list[tuple[int, int, str]]:
    """Return all (start, end, matched_text) spans of any synonym for the finding.

    Case-insensitive. Whole-substring match (no word-boundary requirement —
    medical synonyms have hyphens and apostrophes that fight \\b semantics).
    """
    out: list[tuple[int, int, str]] = []
    lowered = text.lower()
    for syn in finding.synonyms:
        start = 0
        while True:
            idx = lowered.find(syn, start)
            if idx < 0:
                break
            out.append((idx, idx + len(syn), text[idx : idx + len(syn)]))
            start = idx + len(syn)
    # Sort by start, deduplicate overlapping spans (keep longer match).
    out.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    deduped: list[tuple[int, int, str]] = []
    last_end = -1
    for s, e, t in out:
        if s >= last_end:
            deduped.append((s, e, t))
            last_end = e
    return deduped


def _has_marker(text: str, finding: MarkerFinding) -> bool:
    return bool(_findall_marker_spans(text, finding))


def _insert_marker(text: str, finding: MarkerFinding) -> str:
    """Append the templated insertion sentence at the end of the vignette."""
    text = text.rstrip()
    if text and not text.endswith(("." ,"!", "?")):
        text = text + "."
    sep = " " if text else ""
    return text + sep + finding.insertion_template


def _strip_markers(text: str, findings: Iterable[MarkerFinding]) -> str:
    """Sentence-level strip (OPEN_QUESTIONS.md #5, revised by #23).

    Any sentence containing any marker synonym is dropped in full. Sentences
    without marker matches are kept verbatim. This is structurally safe — by
    construction the resulting visible vignette cannot contain mid-clause holes
    or broken sentences that would let an agent infer τ from the *shape* of
    the text rather than its content.
    """
    findings = list(findings)
    if not findings:
        return text

    sentences = _split_sentences(text)
    kept: list[str] = []
    for sent in sentences:
        contains_marker = any(_findall_marker_spans(sent, f) for f in findings)
        if not contains_marker:
            kept.append(sent)
    out = " ".join(s.strip() for s in kept if s.strip())
    return _normalise_whitespace(out)


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text.strip())
    return [p for p in parts if p]


_WS_RE = re.compile(r"\s+")


def _normalise_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Structural-integrity invariant for visible vignettes
# ---------------------------------------------------------------------------


_END_PUNCT = (".", "!", "?")

# Trailing characters that close a sentence-final clause but are not the
# terminator themselves: closing quotes / parens / brackets. We strip these
# before the terminator check so '...?"' counts as terminated.
_TRAILING_CLOSERS = '"\'’”)]}'

# Words that should never end a sentence — if the visible vignette ends a
# sentence on one of these, it's strong evidence that something was redacted
# mid-clause and the rest of the clause was lost. Catches the exact failure
# mode from the original two-pass strip ("consistent with.", "compatible
# with.", etc.). Source list per user instruction (Stage 1 review).
_FORBIDDEN_TRAILING_WORDS = frozenset(
    {"with", "for", "and", "or", "the", "a", "an", "of", "to"}
)


# A "content token" is any maximal run of non-whitespace characters that
# contains at least one alphanumeric. Used to decide whether the last
# semantic chunk of a sentence is a word or a value (number/percent/unit).
_CONTENT_TOKEN_RE = re.compile(r"\S*[A-Za-z0-9]\S*")


def check_vignette_well_formed(vignette: str) -> dict:
    """Run a deterministic syntactic structural-integrity check on a vignette.

    A vignette is well-formed iff every sentence ends in a sentence-final
    character (`.`, `!`, `?`) — possibly followed by a closing quote/paren —
    AND the last *alphabetic-only* token before that terminator is not a
    preposition / article / conjunction. Numeric or unit-bearing endings
    (e.g. "...ejection fraction of 39%.", "...age of 20.") are legitimate
    medical phrasing and pass — the rule fires only when the sentence
    actually ends on a stranded function word.

    Returns a diagnostic dict so callers can surface specific bad sentences
    in the STAGE_REPORT acceptance row.
    """
    sentences = _split_sentences(vignette)
    n = len(sentences)
    n_no_terminator = 0
    n_bad_trailing = 0
    bad_examples: list[str] = []
    for sent in sentences:
        s = sent.rstrip()
        if not s:
            continue
        # Strip trailing closing quotes/parens before checking terminator.
        stripped_close = s.rstrip(_TRAILING_CLOSERS).rstrip()
        if not stripped_close or stripped_close[-1] not in _END_PUNCT:
            n_no_terminator += 1
            bad_examples.append(sent)
            continue
        # Find the last content token before the terminator.
        inner = stripped_close.rstrip("".join(_END_PUNCT)).rstrip(_TRAILING_CLOSERS).rstrip()
        tokens = _CONTENT_TOKEN_RE.findall(inner)
        if not tokens:
            continue
        last = tokens[-1]
        # If the last content token is purely alphabetic AND in the forbidden
        # list, the sentence ended on a stranded function word.
        if last.isalpha() and last.lower() in _FORBIDDEN_TRAILING_WORDS:
            # Exempt single-letter uppercase tokens — those are labels, not
            # articles ("drug A.", "figure B.", "lead V1." after "V1" → "V").
            if len(last) == 1 and last.isupper():
                continue
            n_bad_trailing += 1
            bad_examples.append(sent)
    is_well_formed = (n > 0) and (n_no_terminator == 0) and (n_bad_trailing == 0)
    return {
        "n_sentences": n,
        "n_missing_terminator": n_no_terminator,
        "n_bad_trailing_word": n_bad_trailing,
        "is_well_formed": is_well_formed,
        "bad_examples": bad_examples[:5],
        "all_bad_sentences": bad_examples,
    }


def stripping_introduced_failures(
    original_vignette: str, visible_vignette: str
) -> list[str]:
    """Return the list of malformed sentences in `visible_vignette` that are
    NOT present in `original_vignette` (whitespace-normalised).

    With sentence-level stripping, visible's sentences are a subset of
    `case.vignette`'s sentences plus any grammatically-complete insertion
    templates. So any malformed sentence in `visible_vignette` must have
    been in the original vignette already. A non-empty return list signals
    the strip introduced new structural failures — a regression of the
    OPEN_QUESTIONS.md #23 fix.

    Both sides are whitespace-normalised before comparison so that newlines
    in the original (collapsed to spaces by `_normalise_whitespace` during
    stripping) don't cause spurious set-difference hits.
    """
    orig_bad = check_vignette_well_formed(original_vignette)["all_bad_sentences"]
    vis_bad = check_vignette_well_formed(visible_vignette)["all_bad_sentences"]
    orig = {_normalise_whitespace(s) for s in orig_bad}
    vis = {_normalise_whitespace(s) for s in vis_bad}
    return sorted(vis - orig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def augment_case(
    case: MedQACase,
    attribute: SensitiveAttribute,
    correlation_strength: float,
    rng: np.random.Generator,
    seed: int | None = None,
) -> AugmentedCase:
    """Run the full augmentation procedure on one case.

    `rng` is the source of randomness; `seed` is recorded in the output for
    reproducibility but does not re-seed `rng`.
    """
    if not 0.0 <= correlation_strength <= 1.0:
        raise ValueError("correlation_strength must be in [0, 1].")
    tau = int(rng.random() < attribute.base_rate)

    # Start from the original vignette and decide finding-by-finding.
    full_vignette = case.vignette
    for f in attribute.marker_findings:
        present = _has_marker(full_vignette, f)
        if tau == 1:
            # Want present with prob `correlation_strength`.
            if not present and rng.random() < correlation_strength:
                full_vignette = _insert_marker(full_vignette, f)
        else:
            # Want absent with prob `correlation_strength`.
            if present and rng.random() < correlation_strength:
                full_vignette = _strip_markers(full_vignette, [f])

    # Visible view: strip ALL markers regardless of τ. This is what the agent sees.
    visible_vignette = _strip_markers(full_vignette, attribute.marker_findings)

    hidden = [
        f.name
        for f in attribute.marker_findings
        if _has_marker(full_vignette, f) and not _has_marker(visible_vignette, f)
    ]

    ground_truth = (
        attribute.ground_truth_positive
        if tau == 1
        else attribute.ground_truth_negative
    )

    return AugmentedCase(
        case_id=case.case_id,
        attribute_name=attribute.name,
        tau=tau,
        full_vignette=full_vignette,
        visible_vignette=visible_vignette,
        hidden_findings=hidden,
        question=case.question,
        options=dict(case.options),
        correct_answer=case.correct_answer,
        correlation_strength=correlation_strength,
        seed=seed if seed is not None else 0,
        ground_truth=ground_truth,
    )


def compute_marker_stats(
    augmented: list[AugmentedCase],
    attribute: SensitiveAttribute,
) -> dict:
    """Summary stats for §12.1 acceptance: marker-presence rate vs τ.

    Returns a dict with per-finding presence rates split by τ, the empirical
    base rate, the cohort size, AND the per-case marker-count distribution
    (descriptive — surfaces whether a given correlation_strength value
    produces "billboard" τ=1 cases or subtle ones; see OPEN_QUESTIONS.md #27).
    """
    n = len(augmented)
    if n == 0:
        return {
            "n": 0,
            "n_pos": 0,
            "n_neg": 0,
            "empirical_base_rate": None,
            "by_finding": {},
            "marker_count_distribution": {
                "tau_pos": {str(k): 0 for k in range(len(attribute.marker_findings) + 1)},
                "tau_neg": {str(k): 0 for k in range(len(attribute.marker_findings) + 1)},
                "mean_markers_per_tau_pos": 0.0,
                "mean_markers_per_tau_neg": 0.0,
            },
        }
    n_pos = sum(1 for a in augmented if a.tau == 1)
    n_neg = n - n_pos
    by_finding: dict[str, dict[str, float | int]] = {}
    for f in attribute.marker_findings:
        pos_with = sum(
            1 for a in augmented if a.tau == 1 and _has_marker(a.full_vignette, f)
        )
        neg_with = sum(
            1 for a in augmented if a.tau == 0 and _has_marker(a.full_vignette, f)
        )
        by_finding[f.name] = {
            "rate_when_tau_pos": (pos_with / n_pos) if n_pos else 0.0,
            "rate_when_tau_neg": (neg_with / n_neg) if n_neg else 0.0,
            "n_pos_with": pos_with,
            "n_neg_with": neg_with,
        }
    # Per-case marker count, bucketed by τ.
    counts_pos: list[int] = []
    counts_neg: list[int] = []
    for a in augmented:
        n_markers = sum(
            1 for f in attribute.marker_findings if _has_marker(a.full_vignette, f)
        )
        (counts_pos if a.tau == 1 else counts_neg).append(n_markers)
    n_buckets = len(attribute.marker_findings) + 1
    dist_pos = {str(k): 0 for k in range(n_buckets)}
    for c in counts_pos:
        dist_pos[str(min(c, n_buckets - 1))] += 1
    dist_neg = {str(k): 0 for k in range(n_buckets)}
    for c in counts_neg:
        dist_neg[str(min(c, n_buckets - 1))] += 1
    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "empirical_base_rate": n_pos / n,
        "by_finding": by_finding,
        "marker_count_distribution": {
            "tau_pos": dist_pos,
            "tau_neg": dist_neg,
            "mean_markers_per_tau_pos": (
                sum(counts_pos) / len(counts_pos) if counts_pos else 0.0
            ),
            "mean_markers_per_tau_neg": (
                sum(counts_neg) / len(counts_neg) if counts_neg else 0.0
            ),
        },
    }
