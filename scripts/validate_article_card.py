"""Validate one or more Article Literature Card JSON files against the schema.

Usage:
    python scripts/validate_article_card.py path/to/article_card.json
    python scripts/validate_article_card.py card1.json card2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from jsonschema import Draft7Validator
except ImportError:
    print("ERROR: jsonschema is not installed. Run: pip install jsonschema", file=sys.stderr)
    sys.exit(2)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "article_card.schema.json"


def load_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def validate_file(path: Path, schema: dict) -> list[str]:
    """Return a list of error message strings; empty means the file is valid."""
    try:
        with open(path) as f:
            card = json.load(f)
    except json.JSONDecodeError as e:
        return [f"JSON parse error: {e}"]

    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(card), key=lambda e: list(e.path))
    return [
        f"{'> '.join(str(p) for p in e.absolute_path) or '(root)'}: {e.message}"
        for e in errors
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Article Literature Card JSON files against the schema."
    )
    parser.add_argument("files", nargs="+", type=Path, help="JSON file(s) to validate")
    args = parser.parse_args()

    schema = load_schema()
    any_failed = False

    for path in args.files:
        if not path.exists():
            print(f"MISSING  {path}")
            any_failed = True
            continue

        errors = validate_file(path, schema)
        if errors:
            print(f"FAIL     {path}")
            for msg in errors:
                print(f"         {msg}")
            any_failed = True
        else:
            print(f"OK       {path}")

    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
