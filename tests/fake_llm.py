"""Stand-in for a real LLM client, used in tests."""

from pydantic import BaseModel


class FakeLLM:
    """Returns a fixed reply and records the prompts it was given."""

    model = "fake-llm"

    def __init__(self, reply: BaseModel) -> None:
        self.reply = reply
        self.prompts: list[tuple[str, str]] = []  # (system prompt, user prompt)

    def generate(self, system_prompt, user_prompt, response_model):
        self.prompts.append((system_prompt, user_prompt))
        return response_model.model_validate(self.reply.model_dump())
