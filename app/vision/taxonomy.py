# ruff: noqa: E501  (data table: one condition per line reads better than wrapped tuples)
"""Fixed list of crops and conditions the image diagnosis may return.

The vision model must choose from these ids (structured output enum), so it cannot invent
disease names. Each condition carries the English terms used in the official documents, so the
diagnosis can drive retrieval (Phase 17). Crop ids match the RAG crop vocabulary.

Kinds: disease | pest | virus | deficiency | abiotic | healthy | unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["disease", "pest", "virus", "deficiency", "abiotic", "healthy", "unknown"]

UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Condition:
    id: str
    crop: str  # crop id, or "any" for conditions that apply to every crop
    name: str
    kind: Kind
    search_terms: tuple[str, ...] = ()


CROPS: dict[str, str] = {
    "tomato": "Tomato",
    "potato": "Potato",
    "chilli": "Chilli / Capsicum",
    "brinjal": "Brinjal (Eggplant)",
    "okra": "Okra (Bhindi)",
    "rice": "Rice (Paddy)",
    "wheat": "Wheat",
    "maize": "Maize",
    "cotton": "Cotton",
    "groundnut": "Groundnut",
    "redgram": "Redgram (Pigeon pea)",
    "cowpea": "Cowpea",
    "bean": "Bean",
    "soybean": "Soybean",
    "cucumber": "Cucumber",
    "pumpkin": "Pumpkin / Squash",
    "gourds": "Gourds",
    "ginger": "Ginger",
    "grape": "Grape",
    "apple": "Apple",
    "citrus": "Citrus",
    "mango": "Mango",
    "banana": "Banana",
}

_RAW: tuple[tuple[str, str, str, Kind, tuple[str, ...]], ...] = (
    # tomato
    ("tomato_early_blight", "tomato", "Early blight", "disease", ("early blight", "alternaria")),
    ("tomato_late_blight", "tomato", "Late blight", "disease", ("late blight", "phytophthora")),
    ("tomato_septoria_leaf_spot", "tomato", "Septoria leaf spot", "disease", ("septoria", "leaf spot")),
    ("tomato_bacterial_spot", "tomato", "Bacterial spot", "disease", ("bacterial spot", "bacterial leaf spot")),
    ("tomato_leaf_mold", "tomato", "Leaf mold", "disease", ("leaf mould", "leaf mold")),
    ("tomato_target_spot", "tomato", "Target spot", "disease", ("target spot", "corynespora")),
    ("tomato_leaf_curl_virus", "tomato", "Leaf curl virus (TYLCV)", "virus", ("leaf curl", "whitefly")),
    ("tomato_mosaic_virus", "tomato", "Mosaic virus", "virus", ("mosaic", "virus")),
    ("tomato_spider_mites", "tomato", "Spider mites", "pest", ("mite", "spider mite")),
    ("tomato_fruit_borer", "tomato", "Fruit borer damage", "pest", ("fruit borer", "helicoverpa")),
    ("tomato_healthy", "tomato", "Healthy", "healthy", ()),
    # potato
    ("potato_early_blight", "potato", "Early blight", "disease", ("early blight", "alternaria")),
    ("potato_late_blight", "potato", "Late blight", "disease", ("late blight", "phytophthora")),
    ("potato_mosaic_virus", "potato", "Viral mosaic", "virus", ("mosaic", "virus")),
    ("potato_leafroll_virus", "potato", "Leaf roll virus (PLRV)", "virus", ("leaf roll", "aphid")),
    ("potato_healthy", "potato", "Healthy", "healthy", ()),
    # chilli
    ("chilli_leaf_curl_virus", "chilli", "Leaf curl virus", "virus", ("leaf curl", "thrips", "whitefly")),
    ("chilli_anthracnose", "chilli", "Anthracnose / fruit rot", "disease", ("anthracnose", "fruit rot", "die back")),
    ("chilli_cercospora_leaf_spot", "chilli", "Cercospora leaf spot", "disease", ("cercospora", "leaf spot")),
    ("chilli_bacterial_spot", "chilli", "Bacterial leaf spot", "disease", ("bacterial leaf spot", "bacterial spot")),
    ("chilli_powdery_mildew", "chilli", "Powdery mildew", "disease", ("powdery mildew",)),
    ("chilli_thrips_damage", "chilli", "Thrips damage", "pest", ("thrips",)),
    ("chilli_healthy", "chilli", "Healthy", "healthy", ()),
    # brinjal
    ("brinjal_shoot_fruit_borer", "brinjal", "Shoot and fruit borer", "pest", ("shoot and fruit borer", "leucinodes")),
    ("brinjal_little_leaf", "brinjal", "Little leaf", "disease", ("little leaf", "phytoplasma")),
    ("brinjal_phomopsis_blight", "brinjal", "Phomopsis blight", "disease", ("phomopsis",)),
    ("brinjal_cercospora_leaf_spot", "brinjal", "Cercospora leaf spot", "disease", ("cercospora", "leaf spot")),
    ("brinjal_bacterial_wilt", "brinjal", "Bacterial wilt", "disease", ("bacterial wilt", "wilt")),
    ("brinjal_healthy", "brinjal", "Healthy", "healthy", ()),
    # okra
    ("okra_yellow_vein_mosaic", "okra", "Yellow vein mosaic", "virus", ("yellow vein mosaic", "whitefly")),
    ("okra_powdery_mildew", "okra", "Powdery mildew", "disease", ("powdery mildew",)),
    ("okra_healthy", "okra", "Healthy", "healthy", ()),
    # rice
    ("rice_blast", "rice", "Blast", "disease", ("blast", "pyricularia", "tricyclazole")),
    ("rice_brown_spot", "rice", "Brown spot", "disease", ("brown spot", "helminthosporium")),
    ("rice_bacterial_leaf_blight", "rice", "Bacterial leaf blight", "disease", ("bacterial leaf blight", "xanthomonas")),
    ("rice_sheath_blight", "rice", "Sheath blight", "disease", ("sheath blight", "rhizoctonia")),
    ("rice_stem_borer", "rice", "Stem borer damage (dead heart)", "pest", ("stem borer", "dead heart")),
    ("rice_healthy", "rice", "Healthy", "healthy", ()),
    # wheat
    ("wheat_yellow_rust", "wheat", "Yellow (stripe) rust", "disease", ("yellow rust", "stripe rust")),
    ("wheat_brown_rust", "wheat", "Brown (leaf) rust", "disease", ("brown rust", "leaf rust")),
    ("wheat_powdery_mildew", "wheat", "Powdery mildew", "disease", ("powdery mildew",)),
    ("wheat_healthy", "wheat", "Healthy", "healthy", ()),
    # maize
    ("maize_fall_armyworm", "maize", "Fall armyworm damage", "pest", ("fall army worm", "spodoptera frugiperda")),
    ("maize_northern_leaf_blight", "maize", "Northern (turcicum) leaf blight", "disease", ("leaf blight", "turcicum")),
    ("maize_common_rust", "maize", "Common rust", "disease", ("rust",)),
    ("maize_gray_leaf_spot", "maize", "Gray leaf spot (Cercospora)", "disease", ("leaf spot", "cercospora")),
    ("maize_healthy", "maize", "Healthy", "healthy", ()),
    # cotton
    ("cotton_leaf_curl_virus", "cotton", "Leaf curl virus", "virus", ("leaf curl", "whitefly")),
    ("cotton_bacterial_blight", "cotton", "Bacterial blight", "disease", ("bacterial blight",)),
    ("cotton_whitefly", "cotton", "Whitefly infestation", "pest", ("whitefly", "white fly")),
    ("cotton_pink_bollworm", "cotton", "Pink bollworm damage", "pest", ("pink bollworm",)),
    ("cotton_healthy", "cotton", "Healthy", "healthy", ()),
    # groundnut
    ("groundnut_tikka_leaf_spot", "groundnut", "Tikka leaf spot", "disease", ("tikka", "leaf spot")),
    ("groundnut_rust", "groundnut", "Rust", "disease", ("rust",)),
    ("groundnut_healthy", "groundnut", "Healthy", "healthy", ()),
    # redgram
    ("redgram_pod_borer", "redgram", "Pod borer damage", "pest", ("pod borer", "helicoverpa", "maruca")),
    ("redgram_wilt", "redgram", "Fusarium wilt", "disease", ("wilt", "fusarium")),
    ("redgram_sterility_mosaic", "redgram", "Sterility mosaic", "virus", ("sterility mosaic",)),
    ("redgram_healthy", "redgram", "Healthy", "healthy", ()),
    # cowpea / bean / soybean
    ("cowpea_rust", "cowpea", "Rust", "disease", ("rust", "uromyces")),
    ("cowpea_mosaic_virus", "cowpea", "Mosaic virus", "virus", ("mosaic",)),
    ("cowpea_healthy", "cowpea", "Healthy", "healthy", ()),
    ("bean_rust", "bean", "Rust", "disease", ("rust",)),
    ("bean_blight", "bean", "Blight", "disease", ("blight",)),
    ("bean_mosaic_virus", "bean", "Mosaic virus", "virus", ("mosaic",)),
    ("bean_healthy", "bean", "Healthy", "healthy", ()),
    ("soybean_rust", "soybean", "Rust", "disease", ("rust",)),
    ("soybean_healthy", "soybean", "Healthy", "healthy", ()),
    # cucurbits
    ("cucumber_anthracnose", "cucumber", "Anthracnose", "disease", ("anthracnose",)),
    ("cucumber_downy_mildew", "cucumber", "Downy mildew", "disease", ("downy mildew",)),
    ("cucumber_powdery_mildew", "cucumber", "Powdery mildew", "disease", ("powdery mildew",)),
    ("cucumber_bacterial_wilt", "cucumber", "Bacterial wilt", "disease", ("bacterial wilt", "wilt")),
    ("cucumber_mosaic_virus", "cucumber", "Mosaic virus", "virus", ("mosaic",)),
    ("cucumber_healthy", "cucumber", "Healthy", "healthy", ()),
    ("pumpkin_bacterial_leaf_spot", "pumpkin", "Bacterial leaf spot", "disease", ("bacterial leaf spot", "leaf spot")),
    ("pumpkin_downy_mildew", "pumpkin", "Downy mildew", "disease", ("downy mildew",)),
    ("pumpkin_powdery_mildew", "pumpkin", "Powdery mildew", "disease", ("powdery mildew",)),
    ("pumpkin_mosaic_virus", "pumpkin", "Mosaic virus", "virus", ("mosaic",)),
    ("pumpkin_healthy", "pumpkin", "Healthy", "healthy", ()),
    ("gourds_powdery_mildew", "gourds", "Powdery mildew", "disease", ("powdery mildew",)),
    ("gourds_downy_mildew", "gourds", "Downy mildew", "disease", ("downy mildew",)),
    ("gourds_fruit_fly", "gourds", "Fruit fly damage", "pest", ("fruit fly",)),
    ("gourds_healthy", "gourds", "Healthy", "healthy", ()),
    # ginger
    ("ginger_leaf_blight", "ginger", "Leaf blight / leaf spot", "disease", ("leaf spot", "leaf blight")),
    ("ginger_rhizome_rot", "ginger", "Rhizome (soft) rot", "disease", ("rhizome rot", "soft rot")),
    ("ginger_pest_damage", "ginger", "Insect pest damage", "pest", ("shoot borer", "pest")),
    ("ginger_healthy", "ginger", "Healthy", "healthy", ()),
    # fruit crops
    ("grape_black_rot", "grape", "Black rot", "disease", ("black rot",)),
    ("grape_downy_mildew", "grape", "Downy mildew", "disease", ("downy mildew",)),
    ("grape_powdery_mildew", "grape", "Powdery mildew", "disease", ("powdery mildew",)),
    ("grape_leaf_blight", "grape", "Leaf blight (Isariopsis)", "disease", ("leaf blight", "leaf spot")),
    ("grape_esca", "grape", "Esca (black measles)", "disease", ("esca",)),
    ("grape_healthy", "grape", "Healthy", "healthy", ()),
    ("apple_scab", "apple", "Scab", "disease", ("scab",)),
    ("apple_black_rot", "apple", "Black rot", "disease", ("black rot",)),
    ("apple_cedar_rust", "apple", "Cedar apple rust", "disease", ("rust",)),
    ("apple_healthy", "apple", "Healthy", "healthy", ()),
    ("citrus_greening", "citrus", "Citrus greening (HLB)", "disease", ("greening", "psylla")),
    ("citrus_canker", "citrus", "Canker", "disease", ("canker",)),
    ("citrus_healthy", "citrus", "Healthy", "healthy", ()),
    ("mango_anthracnose", "mango", "Anthracnose", "disease", ("anthracnose",)),
    ("mango_powdery_mildew", "mango", "Powdery mildew", "disease", ("powdery mildew",)),
    ("mango_healthy", "mango", "Healthy", "healthy", ()),
    ("banana_sigatoka", "banana", "Sigatoka leaf spot", "disease", ("sigatoka", "leaf spot")),
    ("banana_healthy", "banana", "Healthy", "healthy", ()),
    # any crop
    ("nutrient_deficiency", "any", "Nutrient deficiency", "deficiency", ("deficiency", "nutrient")),
    ("water_stress", "any", "Water stress / drying", "abiotic", ("irrigation", "drought")),
    (UNKNOWN, "any", "Unknown / cannot tell", "unknown", ()),
)  # fmt: skip

CONDITIONS: dict[str, Condition] = {
    cid: Condition(id=cid, crop=crop, name=name, kind=kind, search_terms=terms)
    for cid, crop, name, kind, terms in _RAW
}

CROP_IDS: tuple[str, ...] = (*CROPS, UNKNOWN)
CONDITION_IDS: tuple[str, ...] = tuple(CONDITIONS)


def condition(condition_id: str) -> Condition:
    return CONDITIONS.get(condition_id, CONDITIONS[UNKNOWN])


def crop_name(crop_id: str) -> str:
    return CROPS.get(crop_id, "Unknown crop")
