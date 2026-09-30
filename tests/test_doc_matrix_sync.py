"""The published support matrices have to say what the catalog actually claims.

Both README.md and docs/DOCUMENTATION.md restate TASK_FRAMEWORK_MAP as tables. They drifted
repeatedly (rows for engines with no code path, a Forecast rename), so the tables are now
compared against the catalog on every pull request instead of being trusted.
"""
import re
from pathlib import Path

import pytest

from src.task_catalog import FRAMEWORK_IMPORTS, TASK_FRAMEWORK_MAP

SUPPORTED = {"✅", "⚠️", "✅ β"}
UNSUPPORTED = {"❌", "—"}


def _rows(header_and_body: str):
    lines = [line for line in header_and_body.strip().splitlines() if line.strip().startswith("|")]
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    return header, [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in lines[2:]
    ]


def _readme_table():
    text = Path("README.md").read_text(encoding="utf-8")
    start = text.index("| Data Category | Task Type |")
    end = text.index("\n\n", start)
    header, rows = _rows(text[start:end])
    return header, [(row[0], row[1], row[2:]) for row in rows]


def _documentation_tables():
    text = Path("docs/DOCUMENTATION.md").read_text(encoding="utf-8")
    tables = []
    for match in re.finditer(r"^### (Tabular|Sequential|Text|Computer Vision|Multimodal)\n\n(\| Task \|[^\n]*(?:\n\|[^\n]*)*)",
                             text, re.MULTILINE):
        category = match.group(1)
        header, rows = _rows(match.group(2))
        tables.append((category, header, [(row[0], row[1:]) for row in rows]))
    return tables


def _claimed_engines(header, cells):
    engines = header[header.index("AutoGluon"):]
    assert len(engines) == len(cells), f"{engines} vs {cells}"
    for cell in cells:
        assert cell in SUPPORTED | UNSUPPORTED, f"unknown matrix mark {cell!r}"
    return engines, {engine for engine, cell in zip(engines, cells) if cell in SUPPORTED}


@pytest.mark.parametrize("category,header,rows", [(c, h, r) for c, h, r in _documentation_tables()])
def test_documentation_matrix_matches_the_catalog(category, header, rows):
    assert [task for task, _ in rows] == [t for (c, t) in TASK_FRAMEWORK_MAP if c == category]
    for task, cells in rows:
        _, claimed = _claimed_engines(header, cells)
        assert claimed == set(TASK_FRAMEWORK_MAP[(category, task)]), f"{category} / {task}"


def test_readme_matrix_matches_the_catalog():
    header, rows = _readme_table()
    assert {(category, task) for category, task, _ in rows} == set(TASK_FRAMEWORK_MAP)
    for category, task, cells in rows:
        # ⚠️ marks the same engine as supported, only less tested.
        _, claimed = _claimed_engines(header, cells)
        assert claimed == set(TASK_FRAMEWORK_MAP[(category, task)]), f"{category} / {task}"


def test_matrices_cover_every_engine_the_catalog_can_offer():
    """Columns are the engines some row actually offers, so an engine that no row can run
    (TPOT, currently pinned out) does not get a column of empty promises."""
    readme_header, _ = _readme_table()
    offered = {engine for engines in TASK_FRAMEWORK_MAP.values() for engine in engines}
    assert set(readme_header[readme_header.index("AutoGluon"):]) == offered
