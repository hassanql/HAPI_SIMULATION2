"""Tests for src/llm/api_backend.py.

These mock the google.genai client at the import level so the suite is
hermetic — no API calls, no GOOGLE_API_KEY required, no network.
"""
from __future__ import annotations

import pytest


def _make_backend_with_fake_genai(monkeypatch, fake_response_factory):
    """Helper: install a fake `google.genai` module before importing the
    backend, so the backend uses our stub Client + async generate_content.

    `fake_response_factory(model, contents, config)` returns a stub object
    with a `.text` attribute (the simulated LLM response text).
    """
    import sys
    import types

    class _FakeAsyncModels:
        def __init__(self, factory):
            self._factory = factory
            self.calls: list[dict] = []

        async def generate_content(self, *, model, contents, config):
            self.calls.append({"model": model, "contents": contents, "config": config})
            return self._factory(model, contents, config)

    class _FakeAio:
        def __init__(self, factory):
            self.models = _FakeAsyncModels(factory)

    class _FakeSyncModels:
        def list(self):
            return []

    class _FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.aio = _FakeAio(fake_response_factory)
            self.models = _FakeSyncModels()

    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = _FakeClient

    fake_errors = types.ModuleType("google.genai.errors")

    class _ClientError(Exception):
        def __init__(self, code=429, *args):
            super().__init__(*args)
            self.code = code

    class _ServerError(Exception):
        pass

    fake_errors.ClientError = _ClientError
    fake_errors.ServerError = _ServerError

    fake_genai.errors = fake_errors

    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.errors", fake_errors)

    # Reload api_backend so its `from google import genai` picks up our fake
    if "src.llm.api_backend" in sys.modules:
        del sys.modules["src.llm.api_backend"]
    from src.llm.api_backend import APIBackend, APIBackendConfig  # noqa: E402

    return APIBackend, APIBackendConfig


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_backend_requires_api_key(monkeypatch):
    """APIBackend must reject missing API key."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    APIBackend, APIBackendConfig = _make_backend_with_fake_genai(
        monkeypatch, lambda *a, **k: type("R", (), {"text": "ok"})()
    )
    with pytest.raises(RuntimeError, match="API key"):
        APIBackend(APIBackendConfig(models={"agent": "gemini-3-flash-preview"}))


def test_model_for_returns_configured_model(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    APIBackend, APIBackendConfig = _make_backend_with_fake_genai(
        monkeypatch, lambda *a, **k: type("R", (), {"text": "ok"})()
    )
    backend = APIBackend(
        APIBackendConfig(
            models={
                "agent": "gemini-3.1-pro-preview",
                "patient": "gemini-3-flash-preview",
            }
        )
    )
    assert backend.model_for("agent") == "gemini-3.1-pro-preview"
    assert backend.model_for("patient") == "gemini-3-flash-preview"
    assert backend.model_revision("agent") == "gemini-3.1-pro-preview"


def test_model_for_unknown_role_raises(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    APIBackend, APIBackendConfig = _make_backend_with_fake_genai(
        monkeypatch, lambda *a, **k: type("R", (), {"text": "ok"})()
    )
    backend = APIBackend(APIBackendConfig(models={"agent": "gemini-3-flash-preview"}))
    with pytest.raises(KeyError, match="patient"):
        backend.model_for("patient")


def test_generate_returns_text(monkeypatch):
    """A solo call routes to the right model and returns the response."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    APIBackend, APIBackendConfig = _make_backend_with_fake_genai(
        monkeypatch,
        lambda model, contents, config: type("R", (), {"text": f"echo:{contents}"})(),
    )
    backend = APIBackend(
        APIBackendConfig(models={"agent": "gemini-3-flash-preview"})
    )
    out = backend.generate("Hello", role="agent")
    assert out == "echo:Hello"


def test_generate_passes_system_prompt(monkeypatch):
    """system_prompt should land in the config as system_instruction."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    captured: dict = {}

    def factory(model, contents, config):
        captured["config"] = config
        return type("R", (), {"text": "ok"})()

    APIBackend, APIBackendConfig = _make_backend_with_fake_genai(monkeypatch, factory)
    backend = APIBackend(
        APIBackendConfig(models={"agent": "gemini-3-flash-preview"})
    )
    backend.generate(
        "user prompt", role="agent", system_prompt="you are a clinician"
    )
    assert captured["config"].get("system_instruction") == "you are a clinician"


def test_generate_batch_returns_list_in_order(monkeypatch):
    """Concurrent batch returns one response per input, in order."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    APIBackend, APIBackendConfig = _make_backend_with_fake_genai(
        monkeypatch,
        lambda model, contents, config: type("R", (), {"text": f"r:{contents}"})(),
    )
    backend = APIBackend(
        APIBackendConfig(
            models={"agent": "gemini-3-flash-preview"}, max_concurrent=4
        )
    )
    prompts = ["a", "b", "c", "d"]
    out = backend.generate_batch(prompts, roles="agent")
    assert out == ["r:a", "r:b", "r:c", "r:d"]


def test_generate_batch_per_role_routing(monkeypatch):
    """Different roles in a batch route to different models."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    seen_models: list[str] = []

    def factory(model, contents, config):
        seen_models.append(model)
        return type("R", (), {"text": "ok"})()

    APIBackend, APIBackendConfig = _make_backend_with_fake_genai(monkeypatch, factory)
    backend = APIBackend(
        APIBackendConfig(
            models={
                "agent": "gemini-3.1-pro-preview",
                "patient": "gemini-3-flash-preview",
            }
        )
    )
    backend.generate_batch(
        prompts=["a", "b", "a"],
        roles=["agent", "patient", "agent"],
    )
    assert seen_models == [
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview",
    ]


def test_generate_batch_validates_lengths(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    APIBackend, APIBackendConfig = _make_backend_with_fake_genai(
        monkeypatch, lambda *a, **k: type("R", (), {"text": "ok"})()
    )
    backend = APIBackend(APIBackendConfig(models={"agent": "gemini-3-flash-preview"}))
    with pytest.raises(ValueError, match="roles list length"):
        backend.generate_batch(["a", "b"], roles=["agent"])
    with pytest.raises(ValueError, match="system_prompts length"):
        backend.generate_batch(["a", "b"], roles="agent", system_prompts=["s"])


def test_retry_on_rate_limit(monkeypatch):
    """A 429 ClientError on first attempt should retry and succeed."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    call_count = {"n": 0}

    def factory(model, contents, config):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # raise fake ClientError(429)
            from google.genai import errors  # the fake one we installed
            raise errors.ClientError(429, "Rate limit hit")
        return type("R", (), {"text": "ok"})()

    APIBackend, APIBackendConfig = _make_backend_with_fake_genai(monkeypatch, factory)
    backend = APIBackend(
        APIBackendConfig(
            models={"agent": "gemini-3-flash-preview"},
            retry_attempts=3,
            retry_base_seconds=0.0,  # no real sleep in tests
        )
    )
    out = backend.generate("x", role="agent")
    assert out == "ok"
    assert call_count["n"] == 2  # one retry


def test_non_retryable_error_propagates(monkeypatch):
    """A 4xx that isn't 429 (e.g. 400 invalid prompt) should NOT be retried."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    call_count = {"n": 0}

    def factory(model, contents, config):
        call_count["n"] += 1
        from google.genai import errors
        raise errors.ClientError(400, "Invalid prompt")

    APIBackend, APIBackendConfig = _make_backend_with_fake_genai(monkeypatch, factory)
    backend = APIBackend(
        APIBackendConfig(
            models={"agent": "gemini-3-flash-preview"},
            retry_attempts=5,
            retry_base_seconds=0.0,
        )
    )
    from google.genai import errors  # type: ignore[import]
    with pytest.raises(errors.ClientError):
        backend.generate("x", role="agent")
    assert call_count["n"] == 1  # no retry on non-429


def test_logprobs_config_propagates(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    captured: dict = {}

    def factory(model, contents, config):
        captured["config"] = config
        return type("R", (), {"text": "ok"})()

    APIBackend, APIBackendConfig = _make_backend_with_fake_genai(monkeypatch, factory)
    backend = APIBackend(
        APIBackendConfig(
            models={"agent": "gemini-3-flash-preview"},
            enable_logprobs=True,
            logprobs_top_k=10,
        )
    )
    backend.generate("x", role="agent")
    assert captured["config"]["response_logprobs"] is True
    assert captured["config"]["logprobs"] == 10


def test_llm_client_integration(monkeypatch, tmp_path):
    """LLMClient(backend='api', api_backend=...) round-trips through the
    fake genai client correctly."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    APIBackend, APIBackendConfig = _make_backend_with_fake_genai(
        monkeypatch,
        lambda model, contents, config: type("R", (), {"text": f"got:{contents}"})(),
    )
    from src.llm.client import LLMClient, LLMRequest

    backend = APIBackend(
        APIBackendConfig(
            models={"agent": "gemini-3-flash-preview", "patient": "gemini-3-flash-preview"}
        )
    )
    client = LLMClient(
        backend="api",
        cache_dir=tmp_path / "cache",
        api_backend=backend,
    )
    resp = client.generate(LLMRequest(role="agent", prompt="hello"))
    assert resp.text == "got:hello"
    assert resp.cache_hit is False
    assert resp.model_revision == "gemini-3-flash-preview"

    # Second call to identical request should be a cache hit.
    resp2 = client.generate(LLMRequest(role="agent", prompt="hello"))
    assert resp2.text == "got:hello"
    assert resp2.cache_hit is True
    client.close()
