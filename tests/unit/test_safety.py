from __future__ import annotations

import pytest

from app.services.language import SUPPORTED
from app.services.safety import WARNINGS, PesticideSafetyChecker, Restriction

RESTRICTIONS = [
    Restriction("Phorate", (), "banned", None),
    Restriction("Endosulfron", ("Endosulfan",), "banned", None),
    Restriction("Dichlorovos", ("DDVP", "Dichlorvos"), "banned", None),
    Restriction("Benzene Hexachloride", ("BHC", "HCH"), "banned", None),
    Restriction("Captafol 80% Powder", (), "banned_except_export", None),
    Restriction("Thiodemeton / Disulfoton", (), "refused_registration", None),
    Restriction("Dalapon", (), "withdrawn", None),
    Restriction(
        "Malathion",
        (),
        "restricted",
        "Malathion is banned for use on Sorghum, Pea, Bhindi, Brinjal, Tomato and Grape.",
    ),
    Restriction(
        "Monocrotophos", (), "restricted", "Monocrotophos is banned for use on vegetables."
    ),
    Restriction(
        "Mancozeb", (), "restricted", "Mancozeb is banned for use on Guava, Jowar and Tapioca."
    ),
    Restriction(
        "Aluminium Phosphide",
        (),
        "restricted",
        "Operations may be undertaken only by Govt. organizations.",
    ),
]


@pytest.fixture
def checker() -> PesticideSafetyChecker:
    return PesticideSafetyChecker(RESTRICTIONS)


def names(findings: list) -> list[tuple[str, str]]:  # type: ignore[type-arg]
    return [(f.kind, f.name) for f in findings]


def test_banned_names_and_aliases(checker: PesticideSafetyChecker) -> None:
    answer = "Apply phorate 10 G or endosulfan. DDVP also works. Disulfoton is old."
    assert names(checker.find(answer, [])) == [
        ("banned", "Phorate"),
        ("banned", "Endosulfron"),
        ("banned", "Dichlorovos"),
        ("banned", "Thiodemeton / Disulfoton"),
    ]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("धान में फोरेट का प्रयोग करें।", [("banned", "Phorate")]),  # Devanagari only
        ("వరిలో ఫోరేట్‌ను వాడండి.", [("banned", "Phorate")]),  # Telugu, with case suffix
        ("ಭತ್ತಕ್ಕೆ ಫೋರೇಟ್ ಬಳಸಿ.", [("banned", "Phorate")]),  # Kannada
        ("ఎండోసల్ఫాన్ పిచికారీ చేయండి.", [("banned", "Endosulfron")]),
    ],
)
def test_banned_names_in_indian_scripts(
    checker: PesticideSafetyChecker, answer: str, expected: list[tuple[str, str]]
) -> None:
    assert names(checker.find(answer, [])) == expected


def test_crop_restriction_in_indian_script(checker: PesticideSafetyChecker) -> None:
    answer = "टमाटर पर मैलाथियान का छिड़काव करें।"
    assert names(checker.find(answer, ["tomato"])) == [("crop", "Malathion")]
    assert checker.find(answer, ["cotton"]) == []


def test_short_codes_need_exact_case(checker: PesticideSafetyChecker) -> None:
    assert names(checker.find("Use BHC dust", [])) == [("banned", "Benzene Hexachloride")]
    assert checker.find("bhc is mentioned in lowercase", []) == []
    assert checker.find("SUBHCASE", []) == []


def test_no_partial_word_matches(checker: PesticideSafetyChecker) -> None:
    assert checker.find("phorates", []) == []
    assert checker.find("Captafol as seed dresser", []) == []  # only the 80% powder is banned


def test_crop_specific_ban_only_for_that_crop(checker: PesticideSafetyChecker) -> None:
    answer = "Spray malathion 50 EC."
    assert names(checker.find(answer, ["tomato"])) == [("crop", "Malathion")]
    assert checker.find(answer, ["cotton"]) == []
    assert checker.find(answer, []) == []


def test_vegetables_expand(checker: PesticideSafetyChecker) -> None:
    answer = "Monocrotophos 36 SL can be sprayed."
    assert names(checker.find(answer, ["okra"])) == [("crop", "Monocrotophos")]
    assert checker.find(answer, ["rice"]) == []


def test_common_restricted_products_do_not_warn(checker: PesticideSafetyChecker) -> None:
    # Mancozeb on tomato is approved; Aluminium phosphide restriction is not crop-based.
    assert checker.find("Spray mancozeb 75% WP.", ["tomato"]) == []
    assert checker.find("Mancozeb on jowar", ["sorghum"]) != []
    assert checker.find("Aluminium phosphide tablets", ["rice"]) == []


@pytest.mark.parametrize("language", SUPPORTED)
def test_warnings_in_every_language(checker: PesticideSafetyChecker, language: str) -> None:
    findings = checker.find("phorate and malathion", ["tomato"])
    messages = checker.warnings(findings, language)  # type: ignore[arg-type]
    assert len(messages) == 2
    assert "Phorate" in messages[0] and messages[0].startswith("⚠️")
    assert "Malathion" in messages[1]
    assert set(WARNINGS["banned"]) == set(SUPPORTED)


def test_size(checker: PesticideSafetyChecker) -> None:
    assert checker.size == 7 + 3  # 7 banned-type + 3 crop-specific (aluminium phosphide skipped)
