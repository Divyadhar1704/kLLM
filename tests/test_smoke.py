"""Repository smoke tests: package imports and layout are intact."""

from pathlib import Path

import app

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_PACKAGES = [
    "app/api",
    "app/api/v1",
    "app/core",
    "app/models",
    "app/schemas",
    "app/services",
    "app/repositories",
    "app/rag",
    "app/vision",
    "app/video",
    "app/worker",
]


def test_version() -> None:
    assert app.__version__ == "0.1.0"


def test_package_layout() -> None:
    missing = [p for p in EXPECTED_PACKAGES if not (ROOT / p / "__init__.py").is_file()]
    assert missing == []


def test_env_example_has_no_real_secrets() -> None:
    content = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "change-me" in content
