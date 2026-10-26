from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from app.rag.ingestion import pipeline
from app.rag.ingestion.extractors import ExtractionError, extract, extract_docx
from app.rag.ingestion.pesticides import normalize_name, parse_banned_list
from app.rag.ingestion.pipeline import discover, load_chunks, run_ingestion

BANNED_TEXT = """LIST OF PESTICIDES WHICH ARE BANNED,
(Updated on 31.07.2026)
I. PESTICIDES / FORMULATIONS BANNED IN INDIA
A.
Pesticides Banned for manufacture, import and use.
1. Alachlor ( S.O. 3951 (E), dated 08.08.2018)
2. Dibromochloropropane (DBCP) ( S.O. 569 (E) dated 25th July 1989)
3. Endosulfron ( ad -Interim order of the Supreme Court of India in the Writ
Petition (Civil) No. 213 of 2011)
4. Dichlorovos ( S.O. 3951 (E), dated 08.08.2018)
B.
Pesticide / Pesticide formulations banned for use but continued to manufacture for
export
1. Captafol 80% Powder ( S.O. 679 (E) dated 17th July 2001)
C.
Pesticides Withdrawn (Withdrawal may become inoperative (S.O 915(E) dated 15 th
Jun,2006)
1. Dalapon
* Regulation to be continued in the extant manner for non-insecticidal uses.
II. PESTICIDES REFUSED REGISTRATION
S. No. Name of Pesticides
1. Azinphos Methyl
III. PESTICIDES RESTRICTED FOR USE IN THE COUNTRY
S.
No. Name of Pesticides Details of Restrictions
1.
Captafol
The use of Captafol as foliar spray is banned.
2.
Malathion
Malathion is banned for use on Brinjal, Tomato and Grape.
3. Mancozeb Mancozeb is banned for use on Guava, Jowar and Tapioca.
4.
Trifluralin
i. The Registration and all uses except use in wheat shall be prohibited.
ii. A cautionary statement has to be incorporated in the label.
"""


def test_parse_banned_list() -> None:
    records = parse_banned_list(BANNED_TEXT)
    by_name = {(r.name, r.status): r for r in records}
    assert [r.name for r in records] == [
        "Alachlor",
        "Dibromochloropropane",
        "Endosulfron",
        "Dichlorovos",
        "Captafol 80% Powder",
        "Dalapon",
        "Azinphos Methyl",
        "Captafol",
        "Malathion",
        "Mancozeb",
        "Trifluralin",
    ]
    assert by_name[("Dibromochloropropane", "banned")].aliases == ["DBCP"]
    assert "Endosulfan" in by_name[("Endosulfron", "banned")].aliases
    assert "Dichlorvos" in by_name[("Dichlorovos", "banned")].aliases
    assert ("Captafol 80% Powder", "banned_except_export") in by_name
    assert ("Dalapon", "withdrawn") in by_name
    assert ("Azinphos Methyl", "refused_registration") in by_name
    malathion = by_name[("Malathion", "restricted")]
    assert malathion.details == "Malathion is banned for use on Brinjal, Tomato and Grape."
    assert by_name[("Captafol", "restricted")].details == (
        "The use of Captafol as foliar spray is banned."
    )
    trifluralin = by_name[("Trifluralin", "restricted")]
    assert trifluralin.details is not None and "except use in wheat" in trifluralin.details
    assert "cautionary" in trifluralin.details
    # Banned entries do not carry the legal order text as "details".
    assert by_name[("Alachlor", "banned")].details is None


def test_normalize_name() -> None:
    assert normalize_name("Toxaphene(Camphechlor)") == "toxaphene camphechlor"
    assert normalize_name("2,4, 5-T") == "2 4 5 t"


# ---------------------------------------------------------------- extractors


def make_docx(path: Path) -> Path:
    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    xml = (
        f"<w:document {ns}><w:body>"
        "<w:p><w:r><w:t>Paddy blast management</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>Use resistant varieties and balanced nitrogen.</w:t></w:r></w:p>"
        "<w:tbl>"
        "<w:tr><w:tc><w:p><w:r><w:t>Crop</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>Disease</w:t></w:r></w:p></w:tc></w:tr>"
        "<w:tr><w:tc><w:p><w:r><w:t>Rice</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>Blast 2026</w:t></w:r></w:p></w:tc></w:tr>"
        "</w:tbl>"
        "</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return path


def test_extract_docx(tmp_path: Path) -> None:
    document = extract_docx(make_docx(tmp_path / "advisory.docx"))
    blocks = document.pages[0].blocks
    assert blocks[0].kind == "text"
    assert "balanced nitrogen" in blocks[0].text
    assert blocks[1].kind == "table_row"
    assert blocks[1].text == "Crop: Rice; Disease: Blast 2026"


def test_extract_invalid_docx(tmp_path: Path) -> None:
    bad = tmp_path / "bad.docx"
    bad.write_bytes(b"not a zip")
    with pytest.raises(ExtractionError):
        extract(bad)


def test_extract_unsupported(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError):
        extract(tmp_path / "photo.jpg")


# ---------------------------------------------------------------- pipeline


def make_corpus(root: Path) -> Path:
    raw = root / "raw"
    (raw / "advisories").mkdir(parents=True)
    (raw / "advisories" / "rice_blast_advisory.txt").write_text(
        "RICE BLAST\n\n" + "Blast is a fungal disease of paddy. " * 30, encoding="utf-8"
    )
    make_docx(raw / "advisories" / "paddy.docx")
    (raw / "images").mkdir()
    (raw / "images" / "leaf.jpg").write_bytes(b"\xff\xd8")
    (raw / "dataset" / "labels").mkdir(parents=True)
    (raw / "dataset" / "labels" / "0001.txt").write_text("0 0.5 0.5 0.1 0.1")
    return raw


def test_discover_skips_images_and_labels(tmp_path: Path) -> None:
    files, skipped = discover([make_corpus(tmp_path)])
    names = sorted(p.name for p, _ in files)
    assert names == ["paddy.docx", "rice_blast_advisory.txt"]
    assert skipped[".jpg"] == 1
    assert skipped["excluded"] == 1


def test_run_ingestion_writes_outputs_and_uses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = make_corpus(tmp_path)
    out = tmp_path / "processed"

    report = run_ingestion([raw], out, workers=1)
    assert report.failed == {}
    assert report.processed == 2
    chunks = load_chunks(out / "chunks.jsonl")
    assert len(chunks) == report.total_chunks >= 2
    advisory = next(c for c in chunks if "rice_blast" in c.source_path)
    assert advisory.doc_type == "advisory"
    assert advisory.crops == ["rice"]
    assert advisory.source_path == "raw/advisories/rice_blast_advisory.txt"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["documents"]) == 2
    assert manifest["skipped"][".jpg"] == 1

    # Second run: unchanged files come from the cache; the extractor is never called.
    def fail(*_: object, **__: object) -> None:
        raise AssertionError("should use cache")

    monkeypatch.setattr(pipeline, "process_document", fail)
    second = run_ingestion([raw], out, workers=1)
    assert second.cached == 2
    assert second.processed == 0
    assert [c.chunk_id for c in load_chunks(out / "chunks.jsonl")] == [c.chunk_id for c in chunks]


def test_code_change_invalidates_cache_and_prunes_old_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = make_corpus(tmp_path)
    out = tmp_path / "processed"
    run_ingestion([raw], out, workers=1)
    old_entries = set((out / "cache").glob("*.json"))
    assert len(old_entries) == 2

    monkeypatch.setattr(pipeline, "CACHE_KEY", "v2-changedcode")
    report = run_ingestion([raw], out, workers=1)
    assert report.processed == 2
    assert report.cached == 0
    new_entries = set((out / "cache").glob("*.json"))
    assert len(new_entries) == 2
    assert new_entries.isdisjoint(old_entries)


def test_one_bad_file_does_not_stop_the_run(tmp_path: Path) -> None:
    raw = make_corpus(tmp_path)
    (raw / "advisories" / "broken.docx").write_bytes(b"corrupt")
    report = run_ingestion([raw], tmp_path / "out", workers=1)
    assert report.processed == 2
    assert len(report.failed) == 1
    assert "broken.docx" in next(iter(report.failed))


def test_duplicate_files_ingested_once(tmp_path: Path) -> None:
    raw = make_corpus(tmp_path)
    content = (raw / "advisories" / "rice_blast_advisory.txt").read_bytes()
    (raw / "advisories" / "copy_of_advisory.txt").write_bytes(content)
    report = run_ingestion([raw], tmp_path / "out", workers=1)
    assert report.skipped["duplicate"] == 1
    assert len(report.documents) == 2
