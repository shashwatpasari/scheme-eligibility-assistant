"""Tests for turning dataset rows into Scheme objects."""

from datetime import date

import pytest
from pydantic import ValidationError

from scheme_assistant.build_schemes import clean_text, row_to_scheme, split_documents
from scheme_assistant.hf_dataset import EXPECTED_COLUMNS, Row
from scheme_assistant.models import SchemeLevel


def make_row(**overrides: str | None) -> Row:
    """Build a complete dataset row; pass keyword arguments to change single columns."""
    row: Row = {column: "text" for column in EXPECTED_COLUMNS}
    row.update(
        slug="pm-kisan",
        state="Central",
        ministry="Ministry Of Agriculture and Farmers Welfare",
        documents_required="Aadhaar Card.\nLandholding papers.",
        official_url="https://www.myscheme.gov.in/schemes/pm-kisan",
        scraped_at="2026-07-05 10:10:29.658+00",
    )
    row.update(overrides)
    return row


def test_clean_text_removes_html_and_bold():
    assert (
        clean_text("farmers&#39; families<br>**Note:** apply") == "farmers' families\nNote: apply"
    )


def test_clean_text_handles_double_escaped_html():
    assert (
        clean_text("borrower&amp;#39;s loan &amp;amp; repayment") == "borrower's loan & repayment"
    )


def test_clean_text_simplifies_headings_quotes_and_numbered_lists():
    raw = "#### Age\n\n1. Minimum 18 years.\n1. Maximum 40 years.\n> **Rural**"
    assert clean_text(raw) == "Age\n\n- Minimum 18 years.\n- Maximum 40 years.\nRural"


def test_clean_text_splits_glued_bullets():
    assert clean_text("- **A:** one.- **B:** two") == "- A: one.\n- B: two"


def test_clean_text_tidies_spaces_and_blank_lines():
    assert clean_text("  a  savings   account \n\n\n\nnext  ") == "a savings account\n\nnext"


@pytest.mark.parametrize("empty", [None, "", "  <br>  "])
def test_clean_text_returns_none_for_empty(empty: str | None):
    assert clean_text(empty) is None


def test_split_documents_strips_numbers_and_punctuation():
    raw = "1.\tAadhaar Card.\n2.\tProof of Address,\n\nBank account details;"
    assert split_documents(raw) == ["Aadhaar Card", "Proof of Address", "Bank account details"]


def test_split_documents_of_nothing_is_empty_list():
    assert split_documents(None) == []


def test_central_row_becomes_central_scheme():
    scheme = row_to_scheme(make_row())
    assert scheme.level is SchemeLevel.CENTRAL
    assert scheme.state is None
    assert scheme.documents == ["Aadhaar Card", "Landholding papers"]
    assert scheme.fetched_on == date(2026, 7, 5)
    assert scheme.exclusions is None


def test_state_row_uses_department_when_ministry_missing():
    row = make_row(state="Odisha", ministry=None, department="Agriculture Department")
    scheme = row_to_scheme(row)
    assert scheme.level is SchemeLevel.STATE
    assert scheme.state == "Odisha"
    assert scheme.ministry == "Agriculture Department"


def test_blank_row_is_rejected():
    with pytest.raises(ValidationError):
        row_to_scheme(make_row(benefits=None, eligibility_text=None))
