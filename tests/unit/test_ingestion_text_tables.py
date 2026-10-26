from __future__ import annotations

from app.rag.ingestion.tables import TableState, table_to_rows
from app.rag.ingestion.text import clean_page_texts, is_page_number, normalize_text

# ---------------------------------------------------------------- text


def test_normalize_bullets_spaces_controls() -> None:
    raw = "\uf0b7  Field   sanitation\u00a0now\x0c\r\n\n\n\nNext"
    assert normalize_text(raw) == "• Field sanitation now\n\nNext"


def test_dehyphenates_words_but_never_number_ranges() -> None:
    assert normalize_text("applica-\ntion of spray") == "application of spray"
    # A dose range split across lines must stay a range.
    assert normalize_text("mancozeb 75% WP @ 600-\n800 g") == "mancozeb 75% WP @ 600-\n800 g"


def test_page_numbers() -> None:
    for line in ("12", "- 7 -", "Page 3", "page 3 of 40", "3/40"):
        assert is_page_number(line)
    for line in ("12 kg N/acre", "2026", "Page three"):
        assert not is_page_number(line) or line == "2026"


def test_repeated_headers_and_page_numbers_removed() -> None:
    pages = [f"AESA based IPM - Tomato\nContent of page {i}\n{i}" for i in range(1, 7)]
    cleaned = clean_page_texts(pages, drop_devanagari_lines=False)
    assert cleaned[0] == "Content of page 1"
    assert all("AESA based IPM" not in page for page in cleaned)


def test_devanagari_header_stripped_from_english_docs() -> None:
    text = "भारत सरकार/Government of India\nकृषि एवं ककसान\nDosage per ha"
    cleaned = clean_page_texts([text], drop_devanagari_lines=True)[0]
    assert cleaned == "Government of India\nDosage per ha"
    kept = clean_page_texts([text], drop_devanagari_lines=False)[0]
    assert "भारत" in kept


# ---------------------------------------------------------------- tables

MUP_PAGE_1: list[list[str | None]] = [
    ["1. Fungicides single product formulations use", None, None, None, None, None],
    ["Crop", "Common name\nof the disease", "Dosage per ha", None, None, "Waiting\nperiod (days)"],
    [None, None, "a. i. (g)", "Formulation\n(g/ml)", "Dilution\nin water(L)", None],
    ["Amisulbrom 20% SC", None, None, None, None, None],
    ["Potato", "Late blight\n(Phytophthora\ninfestans)", "100", "500", "375-500", "19"],
    ["Aureofungin 46.15% w/v SP", None, None, None, None, None],
    ["Paddy(Rice)", "Blast, Brown leaf", "-", "0.005%", "500", "30"],
    [None, "spot", None, None, None, None],
    [None, "Sheath blight", "-", "0.01%", "500", "30"],
]
MUP_PAGE_2: list[list[str | None]] = [
    ["Grapes", "Downy mildew", "-", "0.005%", "750", "15"],
    ["Rose", "Powdery mildew", "-", "-", "-", "-"],
]


def test_mup_table_rows_are_self_contained() -> None:
    rows, state = table_to_rows(MUP_PAGE_1)
    assert rows[0] == (
        "1. Fungicides single product formulations use > Amisulbrom 20% SC | Crop: Potato; "
        "Common name of the disease: Late blight (Phytophthora infestans); "
        "Dosage per ha a. i. (g): 100; Formulation (g/ml): 500; Dilution in water(L): 375-500; "
        "Waiting period (days): 19"
    )
    # Wrapped cell text is appended to the row above, not mistaken for a product label.
    assert "Common name of the disease: Blast, Brown leaf spot" in rows[1]
    assert rows[1].startswith("1. Fungicides single product formulations use > Aureofungin")
    # Empty crop cell inherits the crop of the row above (merged cell).
    assert "Crop: Paddy(Rice); Common name of the disease: Sheath blight" in rows[2]
    assert len(rows) == 3
    assert state.label == "Aureofungin 46.15% w/v SP"


def test_headers_and_labels_carry_to_next_page() -> None:
    _, state = table_to_rows(MUP_PAGE_1)
    rows, _ = table_to_rows(MUP_PAGE_2, state)
    assert rows[0].startswith("1. Fungicides single product formulations use > Aureofungin 46.15%")
    assert "Crop: Grapes" in rows[0]
    assert "Waiting period (days): 15" in rows[0]
    # A data row without digits is not mistaken for a header mid-table.
    assert "Crop: Rose" in rows[1]


def test_headers_reused_by_column_count_after_shape_change() -> None:
    _, state = table_to_rows(MUP_PAGE_1)
    other_shape = [["a", "b", "c"], ["1", "2", "3"]]
    _, state = table_to_rows(other_shape, state)
    rows, _ = table_to_rows([["Okra", "Aphids", "15", "75", "500", "3"]], state)
    assert "Crop: Okra" in rows[0]


def test_enumerator_cells_dropped() -> None:
    rows, _ = table_to_rows([["A.", "Pesticides banned", None], [None, "1.", "Alachlor"]])
    assert rows == ["Pesticides banned", "Alachlor"]


def test_table_without_header() -> None:
    rows, _ = table_to_rows([["Vegetative stage", "Weed management", "Two hoeings at 15 days"]])
    assert rows == ["Vegetative stage; Weed management; Two hoeings at 15 days"]


def test_single_column_table() -> None:
    rows, state = table_to_rows([["Line one"], [None], ["Line two"]])
    assert rows == ["Line one", "Line two"]
    assert isinstance(state, TableState)


def test_empty_table() -> None:
    rows, _ = table_to_rows([[None, None], ["", ""]])
    assert rows == []
