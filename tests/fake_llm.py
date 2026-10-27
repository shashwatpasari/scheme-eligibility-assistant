"""Stand-in for a real LLM client, used in tests."""

from pydantic import BaseModel


class FakeLLM:
    """Returns fixed replies and records the prompts it was given.

    One reply is given per response type; each call returns the reply whose type
    matches the requested response model.
    """

    model = "fake-llm"

    def __init__(self, *replies: BaseModel) -> None:
        self.replies = {type(reply): reply for reply in replies}
        self.prompts: list[tuple[str, str]] = []  # (system prompt, user prompt)

    def generate(self, system_prompt, user_prompt, response_model):
        self.prompts.append((system_prompt, user_prompt))
        return self.replies[response_model].model_copy()
