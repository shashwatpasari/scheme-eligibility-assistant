"""Download and check the smartduketech scheme dataset from Hugging Face.

Run with: uv run python -m scheme_assistant.hf_dataset
"""

import csv
import hashlib
import re
import shutil
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

REPO_ID = "smartduketech/indian-government-schemes-2025"
# Pinned to one commit so the data can't change under us; the checksum proves it didn't.
REVISION = "a1bbf7b20778f4e82ed35e09b187bf4fd86a1a29"
URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/Schemes.csv"
SHA256 = "31605736a693fb638cbb722d8a36f95ee46ddd13ce7ecc970a401ee4c3fd739e"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = PROJECT_ROOT / "data" / "raw" / "hf" / "schemes.csv"

EXPECTED_COLUMNS = (
    "slug",
    "name",
    "description",
    "ministry",
    "department",
    "state",
    "benefits",
    "eligibility_text",
    "application_process",
    "documents_required",
    "official_url",
    "scraped_at",
)
TEXT_COLUMNS = (
    "description",
    "benefits",
    "eligibility_text",
    "application_process",
    "documents_required",
)

# The first schemes we plan to use (myScheme slugs).
PLANNED_SLUGS = (
    "pm-kisan",
    "apy",
    "pmjjby",
    "pmsby",
    "pmmvy",
    "pmuy",
    "pm-svanidhi",
    "pmv",
    "ab-pmjay",
    "pm-sym",
    "pmkmdy",
    "sui",
    "pmmy",
)

# Known scheme rules; each phrase must appear somewhere in that scheme's text.
FACT_CHECKS: dict[str, tuple[str, ...]] = {
    "pm-kisan": ("6,000",),  # Rs 6,000 per year
    "apy": ("18 years", "40 years"),  # joining age 18-40
    "pmjjby": ("18 to 50",),
    "pmsby": ("18 years", "70 years"),
    "pm-sym": ("18 and 40", "15,000"),  # age 18-40, income up to Rs 15,000/month
}

MYSCHEME_URL = "https://www.myscheme.gov.in/schemes/{slug}"
SLUG_PATTERN = re.compile(r"[a-z0-9-]+")  # same rule as Scheme.id
MARKUP_PATTERN = re.compile(r"<br\s*/?>|&#39;|&amp;|&quot;|\*\*")

Row = dict[str, str | None]


@dataclass(frozen=True)
class Finding:
    level: Literal["error", "warning", "info"]
    message: str


def download(path: Path = DEFAULT_PATH) -> Path:
    """Download the CSV, unless a verified copy is already at `path`."""
    if path.exists() and _sha256(path) == SHA256:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")  # never leave a half-written CSV behind
    with urllib.request.urlopen(URL, timeout=120) as response, partial.open("wb") as f:
        shutil.copyfileobj(response, f)

    if (actual := _sha256(partial)) != SHA256:
        partial.unlink()
        raise RuntimeError(f"checksum mismatch for {URL}: got {actual}")
    partial.replace(path)
    return path


def _sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_rows(path: Path = DEFAULT_PATH) -> list[Row]:
    """Read the CSV, turning empty cells and the text "null" into None."""
    with path.open(encoding="utf-8", newline="") as f:
        return [
            {key: None if value.strip() in ("", "null") else value for key, value in row.items()}
            for row in csv.DictReader(f)
        ]


def check_rows(
    rows: list[Row],
    planned_slugs: tuple[str, ...] = PLANNED_SLUGS,
    fact_checks: dict[str, tuple[str, ...]] = FACT_CHECKS,
) -> list[Finding]:
    """Check the data is usable for this project. Any error means: don't use it yet."""
    if not rows:
        return [Finding("error", "the file has no rows")]
    missing_columns = [c for c in EXPECTED_COLUMNS if c not in rows[0]]
    if missing_columns:
        return [Finding("error", f"missing columns: {', '.join(missing_columns)}")]

    findings = [Finding("info", f"{len(rows)} schemes")]
    by_slug: dict[str, Row] = {}

    for row in rows:
        slug = row["slug"] or ""
        if slug in by_slug:
            findings.append(Finding("error", f"duplicate slug: {slug!r}"))
        by_slug[slug] = row
        if not SLUG_PATTERN.fullmatch(slug):
            findings.append(Finding("error", f"slug doesn't fit Scheme.id: {slug!r}"))
        if row["official_url"] != MYSCHEME_URL.format(slug=slug):
            findings.append(Finding("error", f"{slug}: official_url doesn't match its slug"))

    states = Counter(row["state"] for row in rows)
    central = states.pop("Central", 0)
    findings.append(
        Finding(
            "info",
            f"{central} central schemes, {sum(states.values())} state schemes "
            f"across {len(states)} states/UTs",
        )
    )
    dates = sorted(row["scraped_at"] for row in rows if row["scraped_at"])
    if dates:
        findings.append(
            Finding("info", f"scraped from myScheme {dates[0][:10]} to {dates[-1][:10]}")
        )

    blank = sorted(slug for slug, row in by_slug.items() if not row["eligibility_text"])
    if blank:
        findings.append(
            Finding("warning", f"{len(blank)} schemes have no eligibility text: {', '.join(blank)}")
        )

    with_markup = sum(
        1 for row in rows if any(row[c] and MARKUP_PATTERN.search(row[c]) for c in TEXT_COLUMNS)
    )
    if with_markup:
        findings.append(
            Finding(
                "warning",
                f"{with_markup} schemes contain HTML/Markdown leftovers "
                "(<br>, &#39;, **); clean before use",
            )
        )

    mentions_exclusions = sum(
        1 for row in rows if "exclu" in (row["eligibility_text"] or "").lower()
    )
    findings.append(
        Finding(
            "warning",
            "no exclusions column; only "
            f"{mentions_exclusions} schemes mention exclusions inside eligibility_text",
        )
    )

    for slug in planned_slugs:
        if slug not in by_slug:
            findings.append(Finding("error", f"planned scheme missing: {slug}"))

    for slug, phrases in fact_checks.items():
        row = by_slug.get(slug)
        if row is None:
            findings.append(Finding("error", f"fact check: {slug} not in the data"))
            continue
        text = " ".join(row[c] or "" for c in TEXT_COLUMNS)
        missing = [p for p in phrases if p not in text]
        if missing:
            findings.append(Finding("error", f"fact check: {slug} text lacks {missing}"))

    return findings


def main() -> int:
    path = download()
    print(f"Data: {path.relative_to(PROJECT_ROOT)} ({REPO_ID} @ {REVISION[:12]})\n")

    findings = check_rows(load_rows(path))
    for finding in findings:
        print(f"[{finding.level.upper()}] {finding.message}")

    errors = sum(f.level == "error" for f in findings)
    checked = len(FACT_CHECKS)
    print(f"\n{len(PLANNED_SLUGS)} planned schemes checked, {checked} fact checks run.")
    print("Result: OK" if not errors else f"Result: {errors} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
