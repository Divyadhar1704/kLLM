from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.ingestion.chunker import ChunkerConfig, chunk_document, is_heading
from app.rag.ingestion.metadata import (
    classify_doc_type,
    detect_as_of,
    detect_crops,
    make_title,
    source_org_for,
)
from app.rag.ingestion.models import Block, DocumentRecord, Page

# ---------------------------------------------------------------- metadata


@pytest.mark.parametrize(
    ("filename", "head", "expected"),
    [
        ("list_of_pesticides_which_are_banned_refused.pdf", "", "pesticide_banned_list"),
        ("2._chemical_mup_fungicide_as_on_31.03.2026_0.pdf", "", "pesticide_use_guide"),
        ("list_pf_pesticide_formulations_registered.pdf", "", "pesticide_registry"),
        ("approved_pesticides_for_desert_locust_control.pdf", "", "pesticide_registry"),
        ("Tomato.pdf", "AESA based IPM \u2013 Tomato Department of Agriculture", "ipm_package"),
        ("cowpea_brown_rust_control_measures.pdf", "", "advisory"),
        ("notes.txt", "random text", "general"),
    ],
)
def test_classify_doc_type(filename: str, head: str, expected: str) -> None:
    assert classify_doc_type(filename, head) == expected


def test_source_org() -> None:
    assert "CIB&RC" in (source_org_for("pesticide_use_guide") or "")
    assert "IPM" in (source_org_for("ipm_package") or "")
    assert source_org_for("general") is None


@pytest.mark.parametrize(
    ("filename", "head", "expected"),
    [
        ("2._chemical_mup_fungicide_as_on_31.03.2026_0.pdf", "", "2026-03-31"),
        ("x.pdf", "LIST OF PESTICIDES (Updated on 31.07.2026)", "2026-07-31"),
        ("x.pdf", "As on: 5-1-2025", "2025-01-05"),
        ("x.pdf", "no date here", None),
        ("x.pdf", "as on 31.02.2026", None),
    ],
)
def test_detect_as_of(filename: str, head: str, expected: str | None) -> None:
    assert detect_as_of(filename, head) == expected


def test_detect_crops_aliases() -> None:
    text = "Pod borer on pigeonpea and Paddy; also bhindi and egg plant. Not 'tomatoey'."
    assert detect_crops(text) == ["brinjal", "okra", "redgram", "rice"]


def test_make_title() -> None:
    assert make_title(Path("CucurbitaceousVegetable.pdf"), "ipm_package", None) == (
        "AESA based IPM - Cucurbitaceous Vegetable"
    )
    assert make_title(Path("2._chemical_mup_fungicide_as_on_31.03.2026_0.pdf"), "x", None) == (
        "Chemical Major Uses of Pesticides fungicide as on 31.03.2026"
    )
    assert make_title(Path("a.pdf"), "x", "Package of Practices for Rice") == (
        "Package of Practices for Rice"
    )
    assert make_title(Path("notes_v2.pdf"), "x", "Microsoft Word - doc1") == "Notes v2"


# ---------------------------------------------------------------- chunker


def record(**overrides: object) -> DocumentRecord:
    params: dict[str, object] = {
        "doc_id": "doc1",
        "sha256": "a" * 64,
        "source_path": "raw/test.pdf",
        "title": "AESA based IPM - Tomato",
        "doc_type": "ipm_package",
        "crops": ["tomato"],
    }
    params.update(overrides)
    return DocumentRecord(**params)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("III. MANAGEMENT OF PESTS", True),
        ("2.1 Nursery management", True),
        ("DISEASES", True),
        ("This is a normal sentence that ends with a period.", False),
        ("• Field sanitation", False),
        ("4. http://en.wikipedia.org/wiki/Helicoverpa", False),
        ("12", False),
    ],
)
def test_is_heading(line: str, expected: bool) -> None:
    assert is_heading(line) is expected


def test_chunks_have_context_header_pages_and_crops() -> None:
    pages = [
        Page(1, [Block("text", "III. DISEASE MANAGEMENT\nEarly blight appears on older\nleaves.")]),
        Page(2, [Block("table_row", "Mancozeb 75% WP | Crop: Potato; Dose: 1.5 kg")]),
    ]
    chunks = chunk_document(pages, record(), ChunkerConfig(min_chars=10))
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text.startswith("AESA based IPM - Tomato > III. DISEASE MANAGEMENT\n")
    assert "Early blight appears on older leaves." in chunk.text  # wrapped line rejoined
    assert (chunk.page_start, chunk.page_end) == (1, 2)
    assert chunk.crops == ["potato", "tomato"]
    assert chunk.token_estimate > 0


def test_chunks_respect_size_and_never_split_rows() -> None:
    rows = [Block("table_row", f"Product {i} | Crop: Rice; Dose: {i} g") for i in range(200)]
    chunks = chunk_document([Page(1, rows)], record(), ChunkerConfig(target_chars=600))
    assert len(chunks) > 5
    assert all(c.char_count <= 600 + 200 for c in chunks)
    body = "\n".join(c.text.split("\n", 1)[1] for c in chunks)
    for i in (0, 57, 199):
        assert f"Product {i} | Crop: Rice; Dose: {i} g" in body


def test_prose_overlap_between_chunks() -> None:
    paragraphs = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(12))
    chunks = chunk_document(
        [Page(1, [Block("text", paragraphs)])],
        record(),
        ChunkerConfig(target_chars=700, overlap_chars=300),
    )
    assert len(chunks) >= 3
    first_tail = chunks[0].text.strip().split("\n")[-1]
    assert first_tail in chunks[1].text


def test_long_unit_split() -> None:
    text = " ".join(f"Sentence number {i} about pests." for i in range(400))
    chunks = chunk_document([Page(1, [Block("text", text)])], record(), ChunkerConfig())
    assert all(c.char_count <= ChunkerConfig().max_chars + 100 for c in chunks)


def test_chunk_ids_deterministic_and_unique() -> None:
    pages = [Page(1, [Block("table_row", f"Row {i} " + "x" * 100) for i in range(60)])]
    first = chunk_document(pages, record())
    second = chunk_document(pages, record())
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)


def test_tiny_chunks_dropped() -> None:
    assert chunk_document([Page(1, [Block("text", "ok")])], record()) == []
