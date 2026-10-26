"""Checks that every scheme data file is valid."""

from pathlib import Path

import pytest

from scheme_assistant.models import Scheme

SCHEMES_DIR = Path(__file__).parent.parent / "data" / "schemes"
SCHEME_FILES = sorted(SCHEMES_DIR.glob("*.json"))


def test_scheme_files_exist():
    assert SCHEME_FILES, (
        f"no scheme files in {SCHEMES_DIR}; run: uv run python -m scheme_assistant.build_schemes"
    )


@pytest.mark.parametrize("path", SCHEME_FILES, ids=lambda p: p.name)
def test_scheme_file_is_valid(path: Path):
    scheme = Scheme.model_validate_json(path.read_text(encoding="utf-8"))
    assert scheme.id == path.stem  # file name must match the scheme id
