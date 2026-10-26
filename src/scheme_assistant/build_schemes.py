"""Turn Hugging Face dataset rows into clean Scheme JSON files in data/schemes/.

Run with: uv run python -m scheme_assistant.build_schemes
"""

import html
import re
from datetime import date

from pydantic import ValidationError

from scheme_assistant.hf_dataset import PROJECT_ROOT, Row, download, load_rows
from scheme_assistant.models import Scheme

SCHEMES_DIR = PROJECT_ROOT / "data" / "schemes"


def clean_text(text: str | None) -> str | None:
    """Remove HTML and Markdown leftovers, keeping line breaks and bullet lists."""
    if text is None:
        return None
    while (unescaped := html.unescape(text)) != text:  # some text is escaped twice
        text = unescaped
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"(?<=\S)- \*\*", "\n- **", text)  # split bullets glued onto one line
    text = text.replace("**", "")
    text = re.sub(r"^[ \t]*(?:#+|>)[ \t]*", "", text, flags=re.MULTILINE)  # headings, quotes
    text = re.sub(r"^[ \t]*\d+\.[ \t]+", "- ", text, flags=re.MULTILINE)  # "1." items -> bullets
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() or None


def split_documents(text: str | None) -> list[str]:
    """One document per line, without bullets or trailing punctuation."""
    lines = (clean_text(text) or "").splitlines()
    items = (line.removeprefix("- ").strip().rstrip(",;.").strip() for line in lines)
    return [item for item in items if item]


def row_to_scheme(row: Row) -> Scheme:
    """Build a Scheme from one dataset row; raises ValidationError if fields are missing."""
    is_central = row["state"] == "Central"
    scraped_at = row["scraped_at"] or ""
    return Scheme.model_validate(
        {
            "id": row["slug"],
            "name": clean_text(row["name"]),
            "level": "central" if is_central else "state",
            "state": None if is_central else row["state"],
            "ministry": row["ministry"] or row["department"],  # state rows only have department
            "details": clean_text(row["description"]),
            "benefits": clean_text(row["benefits"]),
            "eligibility": clean_text(row["eligibility_text"]),
            "exclusions": None,  # this dataset has no exclusions column
            "documents": split_documents(row["documents_required"]),
            "how_to_apply": clean_text(row["application_process"]),
            "source_url": row["official_url"],
            "fetched_on": date.fromisoformat(scraped_at[:10]) if scraped_at else None,
        }
    )


def main() -> int:
    rows = load_rows(download())
    SCHEMES_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped: list[str] = []
    no_documents = 0
    for row in rows:
        try:
            scheme = row_to_scheme(row)
        except ValidationError:
            skipped.append(row["slug"] or "?")  # e.g. rows with no text at all
            continue
        path = SCHEMES_DIR / f"{scheme.id}.json"
        path.write_text(scheme.model_dump_json(indent=2) + "\n", encoding="utf-8")
        written += 1
        no_documents += not scheme.documents

    print(f"Wrote {written} schemes to {SCHEMES_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"Skipped {len(skipped)} rows that don't fit the Scheme model: {', '.join(skipped)}")
    print(f"{no_documents} schemes list no documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
