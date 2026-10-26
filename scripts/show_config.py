"""Print the effective configuration with secrets masked.

Usage:
    python -m scripts.show_config
"""

from __future__ import annotations

import json
import sys

from pydantic import ValidationError

from app.core.config import get_settings


def main() -> int:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print(f"Configuration is invalid:\n{exc}", file=sys.stderr)
        return 1
    print(json.dumps(settings.public_summary(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
