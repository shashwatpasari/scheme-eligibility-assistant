"""A small, provider-neutral interface for structured LLM calls.

Pipeline code depends only on the `LLMClient` protocol. `make_llm()` picks the client
from the model name, so any model can be tried with `--model <name>`:
    claude-*   Anthropic (ANTHROPIC_API_KEY)
    otherwise  Google Gemini API, including the open Gemma models (GEMINI_API_KEY)

API keys are read from a `.env` file at the project root (never committed).

Every client counts the tokens it uses, so evaluations can report cost.
"""

import threading
from dataclasses import dataclass
from typing import Protocol, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from scheme_assistant.hf_dataset import PROJECT_ROOT

DEFAULT_MODEL = "gemini-3.8-flash"  # most accurate free-tier model in the comparison
RETRY_ATTEMPTS = 5  # retries cover rate limits (HTTP 429) and brief outages
INVALID_REPLY_ATTEMPTS = 3  # some models (e.g. Gemma) occasionally return broken JSON
CLAUDE_MAX_TOKENS = 16000  # room for thinking plus the short JSON reply
CLAUDE_EFFORT = "medium"  # set explicitly: the default differs between Claude models

# Paid-tier list prices in US$ per 1M tokens (input, output), checked 2026-10-27.
# Gemini Flash is at its introductory price until 2026-12-31. Gemma is free only.
PRICES_PER_MILLION = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemma-4-31b-it": (0.0, 0.0),
    "gemma-4-26b-a4b-it": (0.0, 0.0),
}

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class LLMError(RuntimeError):
    """The model returned no usable reply (e.g. it declined the request)."""


@dataclass
class TokenUsage:
    """Running token totals for one client. Safe to update from several threads."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0  # includes thinking tokens, which are billed as output
    invalid_replies: int = 0  # replies that did not match the schema and were retried

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def add(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    def add_invalid_reply(self) -> None:
        with self._lock:
            self.invalid_replies += 1


def cost_usd(model: str, usage: TokenUsage) -> float | None:
    """Price of the tokens used, or None if the model's price is unknown."""
    if model not in PRICES_PER_MILLION:
        return None
    input_price, output_price = PRICES_PER_MILLION[model]
    return (usage.input_tokens * input_price + usage.output_tokens * output_price) / 1_000_000


class LLMClient(Protocol):
    """Anything that can turn a prompt into a validated Pydantic object."""

    model: str
    usage: TokenUsage

    def generate(
        self, system_prompt: str, user_prompt: str, response_model: type[ResponseT]
    ) -> ResponseT: ...


class GeminiClient:
    """Calls a Gemini or Gemma model and parses its JSON reply into a Pydantic model."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        # Imported lazily so that tests and non-LLM commands don't need the SDK loaded.
        from google import genai
        from google.genai import types

        load_dotenv(PROJECT_ROOT / ".env")
        self.model = model
        self.usage = TokenUsage()
        self._types = types
        self._client = genai.Client(
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=RETRY_ATTEMPTS)
            )
        )

    def generate(
        self, system_prompt: str, user_prompt: str, response_model: type[ResponseT]
    ) -> ResponseT:
        """Ask the model for a reply that matches `response_model`.

        Args:
            system_prompt: Instructions describing the task.
            user_prompt: The input to apply the instructions to.
            response_model: Pydantic model the reply must conform to.

        Returns:
            The reply, validated as an instance of `response_model`.

        Raises:
            LLMError: If no reply matches the schema after INVALID_REPLY_ATTEMPTS tries.
        """
        config = self._types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=response_model,
        )
        for _ in range(INVALID_REPLY_ATTEMPTS):
            response = self._client.models.generate_content(
                model=self.model, contents=user_prompt, config=config
            )
            metadata = response.usage_metadata
            self.usage.add(
                input_tokens=metadata.prompt_token_count or 0,
                output_tokens=(metadata.candidates_token_count or 0)
                + (metadata.thoughts_token_count or 0),
            )
            try:
                return response_model.model_validate_json(response.text or "")
            except ValidationError:
                self.usage.add_invalid_reply()
        raise LLMError(f"{self.model} gave no valid reply in {INVALID_REPLY_ATTEMPTS} tries")


class ClaudeClient:
    """Calls a Claude model with structured output parsed into a Pydantic model."""

    def __init__(self, model: str) -> None:
        import anthropic  # imported lazily, like the Gemini SDK

        load_dotenv(PROJECT_ROOT / ".env")
        self.model = model
        self.usage = TokenUsage()
        self._client = anthropic.Anthropic(max_retries=RETRY_ATTEMPTS)

    def generate(
        self, system_prompt: str, user_prompt: str, response_model: type[ResponseT]
    ) -> ResponseT:
        """Ask the model for a reply that matches `response_model`.

        See GeminiClient.generate for the arguments.

        Raises:
            LLMError: If the model declines the request or returns no parsed reply.
        """
        response = self._client.messages.parse(
            model=self.model,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            output_format=response_model,
            output_config={"effort": CLAUDE_EFFORT},
        )
        self.usage.add(response.usage.input_tokens, response.usage.output_tokens)
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise LLMError(f"{self.model} returned no reply (stop reason: {response.stop_reason})")
        return response.parsed_output


def make_llm(model: str = DEFAULT_MODEL) -> LLMClient:
    """Create the right client for a model name."""
    if model.startswith("claude-"):
        return ClaudeClient(model)
    return GeminiClient(model)
