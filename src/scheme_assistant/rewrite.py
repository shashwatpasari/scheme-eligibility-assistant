"""Query rewriting: turn a user's question into a clean English search query.

Users write in English, Hindi or Hinglish and describe their situation in their own
words. Before searching, an LLM rewrites the question into:
    search_query  An English query that uses the wording schemes tend to use.
    state         The state or union territory, if the user mentions one or a city in it.
    facts         Short facts about the person (age, occupation, ...) for the eligibility step.

Usage:
    uv run python -m scheme_assistant.rewrite "Main Leh mein rehne wali vidhwa hoon"
"""

import argparse

from pydantic import BaseModel

from scheme_assistant.llm import DEFAULT_GEMINI_MODEL, GeminiClient, LLMClient

# State and UT names exactly as they appear in the scheme data.
STATES = [
    "Andaman and Nicobar Islands",
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chandigarh",
    "Chhattisgarh",
    "Dadra & Nagar Haveli and Daman & Diu",
    "Delhi",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jammu and Kashmir",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Ladakh",
    "Lakshadweep",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Puducherry",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
]

SYSTEM_PROMPT = f"""\
You help Indian citizens find government schemes. Rewrite the user's message for a \
search engine over scheme descriptions.

Return:
- search_query: the message as one clear English sentence. Translate Hindi or \
Hinglish. Keep every specific detail: scheme and card names, places, ages, amounts, \
occupation, caste or tribe, disability, family situation, and what the user wants. \
Do not add generic words the user did not say, do not answer the question, and do not \
name schemes the user did not mention.
- state: the state or union territory the user lives in or comes from, only if they \
name it or a place in it. Use exactly one of: {", ".join(STATES)}. If it is unclear \
(e.g. they study in one state but ask about their unnamed home state), return null. \
Never guess.
- facts: short English facts about the user, one per item, e.g. "age 55", "widow", \
"no income", "lives in Leh". Empty if none are given.
"""


class QueryRewrite(BaseModel):
    """The LLM's structured reading of a user question."""

    search_query: str
    state: str | None
    facts: list[str]


def normalise_state(name: str | None) -> str | None:
    """Map a state name to its spelling in the scheme data, or None if unknown."""
    if name is None:
        return None
    for state in STATES:
        if state.lower() == name.strip().lower():
            return state
    return None


def rewrite_query(llm: LLMClient, query: str) -> QueryRewrite:
    """Rewrite one user question with the LLM.

    Args:
        llm: Client used to call the model.
        query: The question as the user typed it.

    Returns:
        The rewrite, with `state` set to a known state name or None.
    """
    rewrite = llm.generate(SYSTEM_PROMPT, query, QueryRewrite)
    rewrite.state = normalise_state(rewrite.state)
    return rewrite


def main() -> int:
    parser = argparse.ArgumentParser(description="Rewrite a question for scheme search.")
    parser.add_argument("query", help="the user's question")
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL, help="Gemini model name")
    args = parser.parse_args()

    rewrite = rewrite_query(GeminiClient(args.model), args.query)
    print(rewrite.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
