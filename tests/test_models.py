"""Tests for the shared data models."""

import pytest
from pydantic import ValidationError

from scheme_assistant.models import Scheme, SchemeLevel


def make_scheme(**overrides: object) -> Scheme:
    """Build a valid scheme; pass keyword arguments to change single fields."""
    data = {
        "id": "pm-kisan",
        "name": "Pradhan Mantri Kisan Samman Nidhi",
        "level": "central",
        "details": "Income support for land-holding farmer families.",
        "benefits": "Rs 6,000 per year in three instalments.",
        "eligibility": "Farmer families that own cultivable land.",
        "documents": ["Aadhaar card", "Land records"],
        "how_to_apply": "Register on pmkisan.gov.in.",
        "source_url": "https://pmkisan.gov.in",
        "fetched_on": "2026-10-19",
    }
    data.update(overrides)
    return Scheme.model_validate(data)


def test_valid_scheme_loads():
    scheme = make_scheme()
    assert scheme.level is SchemeLevel.CENTRAL
    assert str(scheme.fetched_on) == "2026-10-19"
    assert scheme.exclusions is None


def test_state_scheme_with_state_loads():
    scheme = make_scheme(id="kalia", level="state", state="Odisha")
    assert scheme.state == "Odisha"


def test_state_scheme_without_state_is_rejected():
    with pytest.raises(ValidationError, match="must name their state"):
        make_scheme(level="state")


def test_central_scheme_with_state_is_rejected():
    with pytest.raises(ValidationError, match="must not have a state"):
        make_scheme(state="Bihar")


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        make_scheme(benifits="typo")


@pytest.mark.parametrize("bad_id", ["PM Kisan", "pm_kisan", "PM-KISAN", ""])
def test_bad_id_is_rejected(bad_id: str):
    with pytest.raises(ValidationError):
        make_scheme(id=bad_id)


def test_bad_url_is_rejected():
    with pytest.raises(ValidationError):
        make_scheme(source_url="pmkisan.gov.in")
