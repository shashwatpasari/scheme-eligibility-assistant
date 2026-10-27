"""A small, provider-neutral interface for structured LLM calls.

Pipeline code depends only on the `LLMClient` protocol, so the provider can be
swapped (Gemini now, Claude later) without touching the callers.

The API key is read from the GEMINI_API_KEY variable, which can be set in a
`.env` file at the project root (never committed).
"""

from typing import Protocol, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

from scheme_assistant.hf_dataset import PROJECT_ROOT

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
RETRY_ATTEMPTS = 5  # retries cover free-tier rate limits (HTTP 429) and brief outages

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class LLMClient(Protocol):
    """Anything that can turn a prompt into a validated Pydantic object."""

    model: str

    def generate(
        self, system_prompt: str, user_prompt: str, response_model: type[ResponseT]
    ) -> ResponseT: ...


class GeminiClient:
    """Calls a Gemini model and parses its JSON reply into a Pydantic model."""

    def __init__(self, model: str = DEFAULT_GEMINI_MODEL) -> None:
        # Imported lazily so that tests and non-LLM commands don't need the SDK loaded.
        from google import genai
        from google.genai import types

        load_dotenv(PROJECT_ROOT / ".env")
        self.model = model
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
            pydantic.ValidationError: If the reply does not match the schema.
        """
        config = self._types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=response_model,
        )
        response = self._client.models.generate_content(
            model=self.model, contents=user_prompt, config=config
        )
        return response_model.model_validate_json(response.text)
