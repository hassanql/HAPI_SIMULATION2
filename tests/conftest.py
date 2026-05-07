"""Shared pytest fixtures.

Stage 0 tests run hermetically: no GPU, no network, no real model. The
synthetic 5-case MedQA fixture lives in `src.data.medqa_loader.synthetic_cases()`;
this conftest just exposes it as a fixture along with a tmp_results_root.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.attribute_schema import get_attribute
from src.data.augmenter import augment_case
from src.data.medqa_loader import synthetic_cases
from src.llm.client import LLMClient
from src.llm.mock_backend import MockBackend


@pytest.fixture
def cases():
    """The 5-case synthetic MedQA fixture (OPEN_QUESTIONS.md #1)."""
    return synthetic_cases()


@pytest.fixture
def hiv_attribute():
    return get_attribute("hiv_status")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def augmented_hiv_cases(cases, hiv_attribute, rng):
    return [augment_case(c, hiv_attribute, 0.8, rng, seed=0) for c in cases]


@pytest.fixture
def mock_backend():
    return MockBackend()


@pytest.fixture
def llm_client(mock_backend, tmp_path):
    rev = mock_backend.revision
    client = LLMClient(
        backend="mock",
        cache_dir=tmp_path / "cache",
        mock_backend=mock_backend,
        model_revisions={
            "agent": rev,
            "patient": rev,
            "judge": rev,
            "covert_attacker": rev,
        },
    )
    yield client
    client.close()


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def prompts_dir(repo_root) -> Path:
    return repo_root / "prompts"
