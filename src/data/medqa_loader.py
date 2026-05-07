"""MedQA loading and filtering.

Stage 0 shipped only the schema and a synthetic 5-case fixture (used in tests).
Stage 1 adds the real HuggingFace loader. The spec (§5.1) prescribes
`bigbio/med_qa` with config `med_qa_en_source`, but `datasets >= 4.x` dropped
support for the dataset script that BigBIO uses. We fall back to
`GBaker/MedQA-USMLE-4-options` (parquet, ~11 K total rows, same MedQA-USMLE
source) and document the deviation in OPEN_QUESTIONS.md.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field, model_validator


class MedQACase(BaseModel):
    """One MedQA-style multiple-choice clinical question.

    case_id is a stable SHA-256 prefix of the question text (see
    OPEN_QUESTIONS.md #2). Same question text → same id, even across runs.
    """

    case_id: str
    question: str
    vignette: str
    options: dict[str, str]
    correct_answer: str
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _correct_answer_in_options(self) -> "MedQACase":
        if self.correct_answer not in self.options:
            raise ValueError(
                f"correct_answer {self.correct_answer!r} not in options "
                f"{sorted(self.options)}"
            )
        return self


def make_case_id(question: str) -> str:
    """SHA-256 prefix of the question text (16 hex chars)."""
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


DEFAULT_MEDQA_SOURCE = "GBaker/MedQA-USMLE-4-options"


def load_medqa(
    split: str = "train",
    *,
    source: str = DEFAULT_MEDQA_SOURCE,
) -> list[MedQACase]:
    """Load MedQA from HuggingFace and parse to `MedQACase` objects.

    Spec §5.1 originally prescribed `bigbio/med_qa` with config
    `med_qa_en_source`, but newer `datasets` libraries refuse to load
    script-based datasets. The fallback `GBaker/MedQA-USMLE-4-options` is
    parquet-formatted, has the same MedQA-USMLE source content, and produces
    `MedQACase` objects with identical semantics for our purposes.

    Failures (network, schema mismatch, empty result) raise `RuntimeError`
    with an actionable message — `run_data_stage` catches and surfaces it
    in `STAGE_REPORT.md` blockers.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "The `datasets` package is required to load MedQA. "
            "It is part of the base dependency set (`pip install -e .`); "
            f"install it before running this stage. Underlying: {e}"
        ) from e

    try:
        ds = load_dataset(source, split=split)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load MedQA from HuggingFace ({source}, split={split!r}). "
            f"Common causes: network unavailable, HF rate-limit (set $HF_TOKEN), "
            f"upstream schema change. Underlying error: {type(e).__name__}: {e}"
        ) from e

    cases: list[MedQACase] = []
    for idx, row in enumerate(ds):
        case = _parse_medqa_row(row, idx, source=source)
        if case is not None:
            cases.append(case)

    if not cases:
        raise RuntimeError(
            f"MedQA download succeeded but produced 0 parseable rows from {source}. "
            f"This usually means the upstream schema changed."
        )
    return cases


# Loader version. Bumped when the splitter or any other parsing step changes
# in a way that affects MedQACase output. The Stage 1 handler embeds this in
# the filtered-cache filename so a stale cache is automatically ignored after
# a logic change. See OPEN_QUESTIONS.md #34.
LOADER_VERSION = "v2"


# Fallback regex (v1): matches the LAST `?`-terminated chunk that has no
# preceding `.`/`!`/`?` between it and the end. This was the original
# splitter, but it has a fatal bug on USMLE cases with embedded lab values
# like "Ca2+: 8.5 mg/dL" — the `.` inside `8.5` is mis-read as a sentence
# boundary, splitting mid-token. ~ 5.8 % of MedQA train cases hit this. The
# fix is to find the question STEM directly, not walk back from the `?`.
# See OPEN_QUESTIONS.md #33.
_QUESTION_RE = re.compile(r"([^.?!]*\?)\s*$")


# Known USMLE-style question stems (44). The splitter finds the LAST
# occurrence of any of these in the row and splits there. The list is ordered
# longest-first within families so a more-specific stem wins over a less-
# specific one when both match (string-position match short-circuits naturally
# via `rfind`).
_QUESTION_STEMS: tuple[str, ...] = (
    # "Which of the following ..." family — most USMLE cases.
    "Which of the following is the most likely",
    "Which of the following is most likely",
    "Which of the following is the next best step",
    "Which of the following is the next step",
    "Which of the following is the most appropriate",
    "Which of the following is the best",
    "Which of the following best describes",
    "Which of the following best explains",
    "Which of the following best",
    "Which of the following findings",
    "Which of the following statements",
    "Which of the following structures",
    "Which of the following mechanisms",
    "Which of the following organisms",
    "Which of the following medications",
    "Which of the following laboratory",
    "Which of the following pharmacologic",
    "Which of the following processes",
    "Which of the following changes",
    "Which of the following is associated",
    "Which of the following indicates",
    "Which of the following will",
    "Which of the following would",
    "Which of the following should",
    "Which of the following is",
    "Which of the following are",
    "Which of these",
    # "What is the ..." family.
    "What is the most likely diagnosis",
    "What is the most likely cause",
    "What is the most likely etiology",
    "What is the most likely mechanism",
    "What is the most likely organism",
    "What is the most likely complication",
    "What is the most likely",
    "What is the next best step",
    "What is the most appropriate next step",
    "What is the most appropriate",
    "What is the underlying",
    "What is the best",
    "What is this patient's",
    "What is this patient",
    # Other common stems.
    "What additional",
    "What would be",
    "What should be",
    "How would you",
)


def _split_vignette_question(full_text: str) -> tuple[str, str]:
    """Return (vignette, question).

    Strategy (v2 — see OPEN_QUESTIONS.md #33):
      1. Find the LAST occurrence of any known question stem in the text.
         If found and the suffix ends in `?`, split there. Stem-based splits
         are immune to embedded `.` inside lab values, decimals, and dosages.
      2. Fall back to the v1 regex (`([^.?!]*\\?)\\s*$`) for residual cases
         (~ 0.27 % of MedQA after the v2 lookup) whose questions don't begin
         with any of the 44 known stems.
      3. Fall back to `(full, full)` if the row contains no `?` at all
         (rare; usually means upstream-malformed MedQA).
    """
    text = full_text.strip()
    if "?" not in text:
        return text, text

    # 1. Stem-based split.
    best_idx = -1
    for stem in _QUESTION_STEMS:
        idx = text.rfind(stem)
        if idx > best_idx:
            best_idx = idx
    if best_idx > 0:
        vignette = text[:best_idx].rstrip()
        question = text[best_idx:].strip()
        if vignette and question.endswith("?"):
            return vignette, question

    # 2. Regex fallback for non-stem questions ending in `?`.
    m = _QUESTION_RE.search(text)
    if m:
        question = m.group(1).strip()
        vignette = text[: m.start()].rstrip()
        if vignette:
            return vignette, question
        return text, question

    # 3. No clean split possible.
    return text, text


def _parse_medqa_row(row: dict, idx: int, *, source: str) -> MedQACase | None:
    """Parse a row from `GBaker/MedQA-USMLE-4-options` (or compatible).

    Schema:
        question:  str (combined vignette + question)
        options:   dict[str, str]   (keys A/B/C/D)
        answer:    str              (the option text, not the letter)
        answer_idx: str             (the letter, e.g. "D")
        meta_info: str
        metamap_phrases: list[str]
    """
    full = row.get("question")
    if not isinstance(full, str) or not full.strip():
        return None

    options_raw = row.get("options")
    if isinstance(options_raw, dict):
        options = {str(k).strip(): str(v).strip() for k, v in options_raw.items()}
    elif isinstance(options_raw, list):
        # Older bigbio shape: [{"key": "A", "value": "..."}, ...]
        options = {}
        for o in options_raw:
            if isinstance(o, dict) and "key" in o and "value" in o:
                options[str(o["key"]).strip()] = str(o["value"]).strip()
    else:
        return None
    if not options:
        return None

    answer_idx = row.get("answer_idx")
    if isinstance(answer_idx, str) and answer_idx.strip() in options:
        correct_answer = answer_idx.strip()
    else:
        # Fall back to matching by value.
        ans_value = (row.get("answer") or "").strip()
        correct_answer = next(
            (k for k, v in options.items() if v == ans_value), None
        )
        if correct_answer is None:
            return None

    vignette, question = _split_vignette_question(full)

    return MedQACase(
        case_id=make_case_id(f"{source}::{full}"),
        question=question,
        vignette=vignette,
        options=options,
        correct_answer=correct_answer,
        metadata={
            "source": source,
            "raw_idx": idx,
            "meta_info": row.get("meta_info"),
        },
    )


def filter_cases(
    cases: list[MedQACase],
    min_vignette_chars: int = 600,
    max_vignette_chars: int = 2000,
    max_options: int = 5,
) -> list[MedQACase]:
    """Keep only cases with rich vignettes and tractable option counts (spec §5.1).

    Also de-duplicates by `case_id`. Real MedQA datasets occasionally contain
    duplicate rows (same vignette+question+options); without dedup the
    `case_ids_unique` invariant in Stage 1 fails. See OPEN_QUESTIONS.md #22.
    """
    seen: set[str] = set()
    out: list[MedQACase] = []
    for c in cases:
        n = len(c.vignette)
        if n < min_vignette_chars or n > max_vignette_chars:
            continue
        if len(c.options) > max_options:
            continue
        if c.case_id in seen:
            continue
        seen.add(c.case_id)
        out.append(c)
    return out


def write_cases_jsonl(cases: list[MedQACase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(c.model_dump_json() + "\n")


def read_cases_jsonl(path: Path) -> list[MedQACase]:
    out: list[MedQACase] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(MedQACase.model_validate(json.loads(line)))
    return out


# ---------------------------------------------------------------------------
# Synthetic fixture (5 cases). Used by Stage 0 bootstrap and by tests.
# Hand-built so augmentation/stripping is exercised on realistic prose
# without hitting the HF download (OPEN_QUESTIONS.md #1).
# ---------------------------------------------------------------------------


_SYNTHETIC = [
    {
        "vignette": (
            "A 32-year-old man presents to the emergency department with a two-week "
            "history of progressive shortness of breath and a dry, non-productive "
            "cough. He reports low-grade fevers in the evenings and increasing fatigue. "
            "He has lost approximately 4 kg over the past two months without trying. "
            "He denies chest pain, haemoptysis, or recent travel. Vital signs show a "
            "temperature of 38.1 °C, heart rate 102 bpm, respiratory rate 24, blood "
            "pressure 118/76, and oxygen saturation of 91 % on room air. Examination "
            "reveals diffuse non-tender cervical and axillary lymphadenopathy and "
            "scattered fine crackles on lung auscultation. Chest imaging shows "
            "bilateral diffuse interstitial infiltrates."
        ),
        "question": "What is the most likely diagnosis?",
        "options": {
            "A": "Bacterial community-acquired pneumonia",
            "B": "Pneumocystis jirovecii pneumonia",
            "C": "Pulmonary tuberculosis",
            "D": "Idiopathic pulmonary fibrosis",
        },
        "correct_answer": "B",
    },
    {
        "vignette": (
            "A 45-year-old woman comes to the clinic with a six-week history of "
            "fatigue, frequent headaches, and intermittent palpitations. She reports "
            "feeling cold in warm rooms and notes that her hair has been thinning. "
            "She denies recent infections, weight loss, or travel. Examination is "
            "notable for dry skin, bradycardia at 56 bpm, and a diffusely enlarged, "
            "non-tender thyroid gland. Reflexes are slow to relax. Laboratory studies "
            "show TSH 14.5 mIU/L (elevated) and free T4 0.6 ng/dL (low)."
        ),
        "question": "Which of the following is the most likely diagnosis?",
        "options": {
            "A": "Graves disease",
            "B": "Subacute thyroiditis",
            "C": "Hashimoto thyroiditis",
            "D": "Toxic multinodular goitre",
        },
        "correct_answer": "C",
    },
    {
        "vignette": (
            "A 27-year-old woman is brought to the emergency department with severe "
            "right-lower-quadrant abdominal pain that started six hours ago. The pain "
            "is sharp, constant, and worse with movement. She also reports light "
            "vaginal spotting. Her last menstrual period was approximately seven weeks "
            "ago. She is sexually active with a single partner and is not on "
            "contraception. Vital signs are temperature 37.2 °C, heart rate 110 bpm, "
            "blood pressure 96/58. On examination there is right adnexal tenderness "
            "without rebound. A urine pregnancy test is positive."
        ),
        "question": "What is the most likely diagnosis?",
        "options": {
            "A": "Ovarian torsion",
            "B": "Acute appendicitis",
            "C": "Ectopic pregnancy",
            "D": "Ruptured ovarian cyst",
        },
        "correct_answer": "C",
    },
    {
        "vignette": (
            "A 58-year-old man with a history of hypertension and type 2 diabetes "
            "presents with sudden-onset severe substernal chest pressure radiating to "
            "the left arm, associated with diaphoresis and shortness of breath. The "
            "pain began 45 minutes ago while he was watching television. He has no "
            "prior history of similar episodes. Vital signs show heart rate 96 bpm, "
            "blood pressure 152/94, oxygen saturation 96 % on room air. ECG reveals "
            "ST-segment elevation in leads II, III, and aVF. Initial troponin I is "
            "elevated at 1.4 ng/mL."
        ),
        "question": "Which artery is most likely occluded?",
        "options": {
            "A": "Left anterior descending",
            "B": "Right coronary",
            "C": "Left circumflex",
            "D": "Posterior descending",
        },
        "correct_answer": "B",
    },
    {
        "vignette": (
            "A 65-year-old woman with a 40-pack-year smoking history presents with "
            "three months of progressive shortness of breath on exertion and a chronic "
            "morning cough productive of clear sputum. She reports needing to use two "
            "pillows at night to breathe comfortably. She denies haemoptysis or chest "
            "pain. Examination shows pursed-lip breathing, decreased breath sounds "
            "bilaterally, and a prolonged expiratory phase. Spirometry reveals an "
            "FEV1/FVC ratio of 0.58 and an FEV1 that is 55 % of predicted, with "
            "minimal bronchodilator response."
        ),
        "question": "What is the most likely diagnosis?",
        "options": {
            "A": "Asthma",
            "B": "Chronic obstructive pulmonary disease",
            "C": "Idiopathic pulmonary fibrosis",
            "D": "Congestive heart failure",
        },
        "correct_answer": "B",
    },
]


def synthetic_cases() -> list[MedQACase]:
    """Return a hermetic 5-case fixture for Stage 0 / unit tests.

    See OPEN_QUESTIONS.md #1 for why we don't hit the real HF download in Stage 0.
    """
    cases: list[MedQACase] = []
    for i, raw in enumerate(_SYNTHETIC):
        cases.append(
            MedQACase(
                case_id=make_case_id(raw["question"] + f"::{i}"),
                question=raw["question"],
                vignette=raw["vignette"],
                options=dict(raw["options"]),
                correct_answer=raw["correct_answer"],
                metadata={"source": "synthetic", "fixture_index": i},
            )
        )
    return cases
