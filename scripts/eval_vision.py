"""Image diagnosis accuracy on labelled datasets (local folders, not in git).

Each sampled photo goes through the full pipeline (validation, quality gate, Gemini vision,
calibration). Reported separately for lab photos (PlantVillage: plain background) and field
photos (real farm conditions), because lab accuracy overstates real-world accuracy.

Metrics:
    crop accuracy · condition top-1 · condition top-3 · confident precision (accuracy when the
    system says "confident") · coverage (share answered confidently)
"Lenient" also accepts the same condition on a sibling crop (pumpkin vs gourds, bean vs cowpea).

Usage:
    python -m scripts.eval_vision --per-class 2
    python -m scripts.eval_vision --per-class 1 --only field
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.services.llm.errors import LLMError
from app.vision.diagnosis import VisionDiagnoser

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(r"C:\Users\krish\Downloads\krishimitra")
PV = DATA / "plantvillage-dataset" / "plantvillage dataset" / "color"
PUMPKIN = DATA / "pumpkin" / "Pumpkin Leaf Diseases Dataset From Bangladesh" / "Original Dataset"
GINGER = DATA / "ginger" / "Ginger-Leaf-Dataset" / "Ginger_Leaf_Dataset" / "combined"
BEAN = (
    DATA / "cowpea" / "Image Dataset for Bean and Cowpea Plant Leaf Disease Detection and "
    "Freshness Assessment from Bangladesh" / "Bean"
)

# (source, folder, expected condition id)
LABELS: list[tuple[str, Path, str]] = [
    ("lab", PV / "Tomato___Early_blight", "tomato_early_blight"),
    ("lab", PV / "Tomato___Late_blight", "tomato_late_blight"),
    ("lab", PV / "Tomato___Bacterial_spot", "tomato_bacterial_spot"),
    ("lab", PV / "Tomato___Septoria_leaf_spot", "tomato_septoria_leaf_spot"),
    ("lab", PV / "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "tomato_leaf_curl_virus"),
    ("lab", PV / "Tomato___healthy", "tomato_healthy"),
    ("lab", PV / "Potato___Early_blight", "potato_early_blight"),
    ("lab", PV / "Potato___Late_blight", "potato_late_blight"),
    ("lab", PV / "Potato___healthy", "potato_healthy"),
    ("lab", PV / "Corn_(maize)___Common_rust_", "maize_common_rust"),
    ("lab", PV / "Corn_(maize)___Northern_Leaf_Blight", "maize_northern_leaf_blight"),
    ("lab", PV / "Pepper,_bell___Bacterial_spot", "chilli_bacterial_spot"),
    ("lab", PV / "Grape___Black_rot", "grape_black_rot"),
    ("lab", PV / "Grape___healthy", "grape_healthy"),
    ("field", PUMPKIN / "Downy Mildew", "pumpkin_downy_mildew"),
    ("field", PUMPKIN / "Powdery_Mildew", "pumpkin_powdery_mildew"),
    ("field", PUMPKIN / "Mosaic Disease", "pumpkin_mosaic_virus"),
    ("field", PUMPKIN / "Bacterial Leaf Spot", "pumpkin_bacterial_leaf_spot"),
    ("field", PUMPKIN / "Healthy Leaf", "pumpkin_healthy"),
    ("field", GINGER / "Leaf-blight", "ginger_leaf_blight"),
    ("field", GINGER / "Damage-Pest", "ginger_pest_damage"),
    ("field", GINGER / "Healthy", "ginger_healthy"),
    ("field", DATA / "cowpea" / "VIGSI_UROMVI_brown rust", "cowpea_rust"),
    ("field", DATA / "cowpea" / "VIGSI_healthy leaves", "cowpea_healthy"),
    ("field", BEAN / "Rust", "bean_rust"),
    ("field", BEAN / "Mosaic Virus", "bean_mosaic_virus"),
    ("field", BEAN / "Fresh Leaf", "bean_healthy"),
    ("field", DATA / "cucumber" / "cucumber_disease" / "Anthracnose", "cucumber_anthracnose"),
    ("field", DATA / "cucumber" / "cucumber_disease" / "Bacterial Wilt", "cucumber_bacterial_wilt"),
]

SIBLINGS = [{"pumpkin", "cucumber", "gourds"}, {"bean", "cowpea"}, {"chilli"}]


def crop_of(condition_id: str) -> str:
    return condition_id.split("_", 1)[0]


def lenient_match(predicted: str, expected: str) -> bool:
    if predicted == expected:
        return True
    p_crop, _, p_rest = predicted.partition("_")
    e_crop, _, e_rest = expected.partition("_")
    same_family = any(p_crop in family and e_crop in family for family in SIBLINGS)
    return same_family and p_rest == e_rest


async def main(per_class: int, only: str | None, delay: float, crop_hint: bool) -> int:
    settings = get_settings()
    diagnoser = VisionDiagnoser.from_settings(settings)
    rng = random.Random(7)  # noqa: S311 - reproducible sampling
    rows: list[dict[str, Any]] = []
    for source, folder, expected in LABELS:
        if only and source != only:
            continue
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not files:
            print(f"  (no images in {folder})")
            continue
        for path in rng.sample(files, min(per_class, len(files))):
            start = time.perf_counter()
            try:
                d = await diagnoser.diagnose(
                    path.read_bytes(), crop=crop_of(expected) if crop_hint else None
                )
            except LLMError as exc:
                print(f"  ! {expected:30} {exc.code}")
                await asyncio.sleep(delay * 3)
                continue
            seconds = time.perf_counter() - start
            top3 = [d.condition.id, *(a.condition.id for a in d.alternatives)]
            row = {
                "source": source,
                "file": str(path.relative_to(DATA)),
                "expected": expected,
                "predicted": d.condition.id,
                "crop_expected": crop_of(expected),
                "crop_predicted": d.crop,
                "status": d.status,
                "confidence": d.confidence,
                "top1": d.condition.id == expected,
                "top1_lenient": lenient_match(d.condition.id, expected),
                "top3": expected in top3,
                "crop_ok": d.crop == crop_of(expected),
                "seconds": round(seconds, 2),
                "symptoms": d.visible_symptoms,
            }
            rows.append(row)
            mark = "OK " if row["top1"] else ("~  " if row["top1_lenient"] else "X  ")
            print(
                f"  {mark}{source:5} {expected:30} -> {d.condition.id:30} "
                f"{d.status:12} {d.confidence:.2f}  {seconds:4.1f}s"
            )
            await asyncio.sleep(delay)
    await diagnoser.aclose()

    print("\n" + "=" * 92)
    for source in ("lab", "field", "all"):
        subset = [r for r in rows if source == "all" or r["source"] == source]
        if not subset:
            continue
        n = len(subset)
        confident = [r for r in subset if r["status"] == "confident"]
        answered = [r for r in subset if r["status"] in ("confident", "uncertain")]

        def rate(key: str, items: list[dict[str, Any]] = subset) -> str:
            return f"{sum(r[key] for r in items) / len(items):.0%}" if items else "-"

        print(
            f"{source:6} n={n:3}  crop {rate('crop_ok')}  top-1 {rate('top1')} "
            f"(lenient {rate('top1_lenient')})  top-3 {rate('top3')}  |  "
            f"confident {len(confident)}/{n}, precision when confident "
            f"{rate('top1_lenient', confident)}  |  retake/not-plant {n - len(answered)}"
        )
    errors: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if not r["top1_lenient"]:
            errors[r["expected"]].append(r["predicted"])
    if errors:
        print("\nconfusions (expected -> predicted):")
        for expected, predicted in errors.items():
            print(f"  {expected:30} -> {', '.join(predicted)}")

    reports = ROOT / "evals" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = reports / f"vision_{stamp}.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport: {path}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--only", choices=["lab", "field"])
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--crop-hint", action="store_true", help="pass the true crop as a hint")
    args = parser.parse_args()
    setup_logging(get_settings())
    raise SystemExit(asyncio.run(main(args.per_class, args.only, args.delay, args.crop_hint)))
