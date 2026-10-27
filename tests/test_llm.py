"""Tests for LLM client selection, token counting and cost."""

import pytest
from pydantic import BaseModel

from scheme_assistant.llm import (
    INVALID_REPLY_ATTEMPTS,
    ClaudeClient,
    GeminiClient,
    LLMError,
    TokenUsage,
    cost_usd,
    make_llm,
)


def test_token_usage_adds_up():
    usage = TokenUsage()
    usage.add(input_tokens=100, output_tokens=20)
    usage.add(input_tokens=50, output_tokens=5)
    assert (usage.calls, usage.input_tokens, usage.output_tokens) == (2, 150, 25)


def test_cost_uses_input_and_output_prices():
    usage = TokenUsage()
    usage.add(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_usd("claude-haiku-5-5", usage) == pytest.approx(0.10 + 0.50)


def test_cost_is_none_for_unknown_model():
    assert cost_usd("some-new-model", TokenUsage()) is None


@pytest.fixture
def fake_keys(monkeypatch):
    # Clients only read the keys when created; no request is sent in these tests.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


@pytest.mark.usefixtures("fake_keys")
def test_make_llm_picks_claude_for_claude_models():
    llm = make_llm("claude-sonnet-5-5")
    assert isinstance(llm, ClaudeClient)
    assert llm.model == "claude-sonnet-5-5"


@pytest.mark.usefixtures("fake_keys")
@pytest.mark.parametrize("model", ["gemini-3.8-flash", "gemma-4-31b-it"])
def test_make_llm_picks_gemini_for_google_models(model: str):
    assert isinstance(make_llm(model), GeminiClient)


class FakeGeminiResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_metadata = type(
            "Usage",
            (),
            {"prompt_token_count": 10, "candidates_token_count": 2, "thoughts_token_count": 0},
        )()


class FakeGeminiModels:
    """Returns the given reply texts in order, one per call."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)

    def generate_content(self, model, contents, config):
        return FakeGeminiResponse(self.texts.pop(0))


class Reply(BaseModel):
    answer: str


def gemini_with_replies(texts: list[str]) -> GeminiClient:
    client = GeminiClient("gemma-4-31b-it")
    client._client = type("FakeSDK", (), {"models": FakeGeminiModels(texts)})()
    return client


@pytest.mark.usefixtures("fake_keys")
def test_gemini_retries_broken_json():
    client = gemini_with_replies(['{"answer": "ye', '{"answer": "yes"}'])
    assert client.generate("system", "user", Reply) == Reply(answer="yes")
    assert client.usage.calls == 2
    assert client.usage.invalid_replies == 1


@pytest.mark.usefixtures("fake_keys")
def test_gemini_gives_up_after_repeated_broken_json():
    client = gemini_with_replies(["not json"] * INVALID_REPLY_ATTEMPTS)
    with pytest.raises(LLMError):
        client.generate("system", "user", Reply)
