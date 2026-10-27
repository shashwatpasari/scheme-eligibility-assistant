"""Shared data models for the scheme eligibility assistant."""

from datetime import date
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class SchemeLevel(StrEnum):
    CENTRAL = "central"
    STATE = "state"


class Scheme(BaseModel):
    """A government scheme, stored as text sections from its source."""

    model_config = ConfigDict(extra="forbid")  # catch typos in JSON field names

    id: str = Field(pattern="^[a-z0-9-]+$")  # e.g. "pm-kisan"; also the file name
    name: str
    level: SchemeLevel
    state: str | None = None  # only for state schemes
    ministry: str | None = None

    # Sections kept as the original text from the source.
    details: str
    eligibility: str
    benefits: str
    exclusions: str | None = None
    documents: list[str]
    how_to_apply: str

    source_url: HttpUrl
    fetched_on: date

    @model_validator(mode="after")
    def state_matches_level(self) -> Self:
        if self.level == SchemeLevel.STATE and not self.state:
            raise ValueError("state schemes must name their state")
        if self.level == SchemeLevel.CENTRAL and self.state:
            raise ValueError("central schemes must not have a state")
        return self


class RetrievalQuery(BaseModel):
    """A test question for scheme search, with the schemes it must find."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^q\d{3}$")  # e.g. "q001"
    query: str = Field(min_length=3)
    user_state: str | None = None  # where the user lives; None if they don't say
    style: Literal["persona", "hinglish", "hindi", "keyword"]
    relevant: list[str] = Field(min_length=1)  # scheme ids that should be found


Verdict = Literal["eligible", "not_eligible", "need_more_info"]

# basic            One clearly stated fact decides the verdict.
# boundary         A value at or near a limit (e.g. age exactly 40, "mid-fifties").
# indirect         A fact must be worked out (e.g. a daily wage to a monthly income).
# hinglish         The description is written in Hinglish.
# multi_condition  Several conditions or an exception in the rules must be combined.
CaseKind = Literal["basic", "boundary", "indirect", "hinglish", "multi_condition"]


class EligibilityCase(BaseModel):
    """A test case for the eligibility check: one user, one scheme, the correct verdict."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^e\d{3}$")  # e.g. "e001"
    kind: CaseKind
    scheme_id: str
    user_state: str | None = None
    user_description: str = Field(min_length=10)
    verdict: Verdict
    note: str  # why this verdict is correct, for reviewers
