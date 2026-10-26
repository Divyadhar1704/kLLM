"""Runs the pipeline on the real curated corpus in documents/raw (not in git). Skipped if absent."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.ingestion.pipeline import load_chunks, run_ingestion

RAW = Path(__file__).resolve().parents[2] / "documents" / "raw"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not any(RAW.rglob("*.pdf")), reason="documents/raw has no PDFs"),
]


@pytest.fixture(scope="module")
def chunks(tmp_path_factory: pytest.TempPathFactory) -> list:  # type: ignore[type-arg]
    out = tmp_path_factory.mktemp("processed")
    report = run_ingestion([RAW], out)
    assert report.failed == {}
    return load_chunks(out / "chunks.jsonl")


def test_mup_rows_keep_product_crop_and_headers(chunks: list) -> None:  # type: ignore[type-arg]
    rows = [
        line
        for c in chunks
        if c.doc_type == "pesticide_use_guide"
        for line in c.text.split("\n")
        if line.startswith("Acetamiprid 20% SP |") and "Crop: Cotton" in line
    ]
    assert rows, "Acetamiprid rows not found"
    assert any("Waiting Period (days): 15" in r for r in rows)


def test_dose_ranges_not_corrupted(chunks: list) -> None:  # type: ignore[type-arg]
    tomato = "\n".join(c.text for c in chunks if c.title == "AESA based IPM - Tomato")
    assert "600800" not in tomato
    assert "600-800 g" in tomato


def test_banned_list_present(chunks: list) -> None:  # type: ignore[type-arg]
    banned = [c for c in chunks if c.doc_type == "pesticide_banned_list"]
    assert banned
    assert any("Phorate" in c.text for c in banned)
