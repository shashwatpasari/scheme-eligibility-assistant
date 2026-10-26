"""Tests for the Hugging Face dataset checks (no network needed)."""

from pathlib import Path

from scheme_assistant.hf_dataset import EXPECTED_COLUMNS, Finding, Row, check_rows, load_rows


def make_row(slug: str = "pm-kisan", **overrides: str | None) -> Row:
    """Build a clean row; pass keyword arguments to change single columns."""
    row: Row = {column: "text" for column in EXPECTED_COLUMNS}
    row.update(
        slug=slug,
        state="Central",
        official_url=f"https://www.myscheme.gov.in/schemes/{slug}",
        scraped_at="2026-07-05 10:10:29+00",
    )
    row.update(overrides)
    return row


def check(rows: list[Row], **kwargs) -> list[Finding]:
    kwargs.setdefault("planned_slugs", ())
    kwargs.setdefault("fact_checks", {})
    return check_rows(rows, **kwargs)


def messages(findings: list[Finding], level: str) -> list[str]:
    return [f.message for f in findings if f.level == level]


def test_clean_rows_have_no_errors():
    assert messages(check([make_row(), make_row("apy")]), "error") == []


def test_empty_file_is_an_error():
    assert messages(check([]), "error") == ["the file has no rows"]


def test_missing_columns_is_an_error():
    assert messages(check([{"slug": "pm-kisan"}]), "error")[0].startswith("missing columns")


def test_duplicate_slug_is_an_error():
    errors = messages(check([make_row(), make_row()]), "error")
    assert errors == ["duplicate slug: 'pm-kisan'"]


def test_slug_must_fit_scheme_id():
    errors = messages(check([make_row("PM Kisan")]), "error")
    assert "slug doesn't fit Scheme.id: 'PM Kisan'" in errors


def test_url_must_match_slug():
    errors = messages(check([make_row(official_url="https://example.com")]), "error")
    assert errors == ["pm-kisan: official_url doesn't match its slug"]


def test_missing_planned_scheme_is_an_error():
    errors = messages(check([make_row()], planned_slugs=("pm-kisan", "apy")), "error")
    assert errors == ["planned scheme missing: apy"]


def test_fact_check_catches_wrong_text():
    rows = [make_row(benefits="Rs 2,000 per year")]
    errors = messages(check(rows, fact_checks={"pm-kisan": ("6,000",)}), "error")
    assert errors == ["fact check: pm-kisan text lacks ['6,000']"]


def test_fact_check_passes_on_right_text():
    rows = [make_row(benefits="Rs 6,000 per year")]
    assert messages(check(rows, fact_checks={"pm-kisan": ("6,000",)}), "error") == []


def test_markup_and_blank_eligibility_are_warnings():
    rows = [make_row(benefits="a<br>b"), make_row("vpby", eligibility_text=None)]
    findings = check(rows)
    warnings = " ".join(messages(findings, "warning"))
    assert "1 schemes contain HTML/Markdown leftovers" in warnings
    assert "no eligibility text: vpby" in warnings
    assert messages(findings, "error") == []


def test_load_rows_turns_null_and_empty_into_none(tmp_path: Path):
    path = tmp_path / "schemes.csv"
    path.write_text('slug,benefits,ministry\nvpby,null,\napy,"Pension, monthly",x\n')
    rows = load_rows(path)
    assert rows[0] == {"slug": "vpby", "benefits": None, "ministry": None}
    assert rows[1]["benefits"] == "Pension, monthly"
