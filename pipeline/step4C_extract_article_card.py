#!/usr/bin/env python3
"""Step 4C: Extract Article Literature Card modules from one article.

Calls the configured LLM (DeepSeek via OpenAI-compatible API) nine times —
once per module — using the prompt templates from Step 4B.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step4B_article_card_prompts import (
    get_article_card_module_question,
    get_article_reading_system_prompt,
    list_article_card_modules,
)

SCHEMA_PATH = REPO_ROOT / "schemas" / "article_card.schema.json"
SCHEMA_VERSION = "0.1.0"
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash")
DEFAULT_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
ALL_MODULES = list_article_card_modules()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_parents(article_id: str, data_root: Path) -> list[dict]:
    """Load parents.jsonl sorted by parent_index."""
    path = data_root / "index" / article_id / "parents.jsonl"
    parents: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                parents.append(json.loads(line))
    return sorted(parents, key=lambda p: p.get("parent_index", 0))


def assemble_reading_text(parents: list[dict]) -> str:
    """Build the ordered article_reading_text from parent chunks.

    Each block includes parent_id, section_path, chunk_type, and text so the
    LLM can use parent_id values verbatim for evidence anchoring.
    """
    blocks: list[str] = []
    for p in parents:
        section = " > ".join(p.get("section_path", []))
        block = (
            f"parent_id: {p['parent_id']}\n"
            f"section_path: {section}\n"
            f"chunk_type: {p.get('chunk_type', '')}\n"
            f"{p.get('text', '').strip()}"
        )
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)


def load_metadata(article_id: str, data_root: Path) -> dict | None:
    """Load metadata.json if it exists; return None otherwise."""
    path = data_root / "articles" / article_id / "metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _clean_title(title: str | None) -> str | None:
    """Strip GROBID web artifacts from parsed titles."""
    if not title:
        return title
    # GROBID sometimes scrapes journal page UI text (e.g. "Check for updates")
    # into the title when parsing PDFs downloaded from publisher websites.
    for prefix in ("Check for updates", "check for updates"):
        if title.startswith(prefix):
            title = title[len(prefix):].lstrip()
    return title or None


def build_metadata_block(raw: dict | None) -> dict:
    """Return a schema-conforming metadata block.

    Handles both naming conventions from Step 1:
      - doi / DOI
      - year / publication_year
    """
    if raw is None:
        return {
            "title": None,
            "authors": [],
            "journal": None,
            "year": None,
            "doi": None,
            "article_type": None,
        }
    return {
        "title": _clean_title(raw.get("title")),
        "authors": raw.get("authors") or [],
        "journal": raw.get("journal"),
        "year": raw.get("year") or raw.get("publication_year"),
        "doi": raw.get("doi") or raw.get("DOI"),
        "article_type": raw.get("article_type"),
    }


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class DeepSeekClient:
    """Thin OpenAI-compatible wrapper pointed at DeepSeek."""

    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        from openai import OpenAI
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.usage: dict[str, int] = {"prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "calls": 0}

    def reset_usage(self) -> None:
        self.usage = {"prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "calls": 0}

    def call(self, system_content: str, user_content: str) -> str:
        """Call the LLM with a system message and a user message.

        Keeping the article text in system_content (constant per article) and
        the module question in user_content enables server-side prompt caching:
        the provider caches the system prefix after the first call and charges
        a fraction of normal input-token cost for the remaining 8 module calls.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
        )
        u = response.usage
        if u:
            self.usage["prompt_tokens"] += u.prompt_tokens
            self.usage["completion_tokens"] += u.completion_tokens
            self.usage["calls"] += 1
            details = getattr(u, "prompt_tokens_details", None)
            if details:
                self.usage["cached_tokens"] += getattr(details, "cached_tokens", 0)
        return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# JSON parsing with fence stripping
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Remove ``` or ```json fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1
        end = len(lines)
        for i in range(1, len(lines)):
            if lines[i].strip() == "```":
                end = i
                break
        text = "\n".join(lines[start:end])
    return text.strip()


def _parse_module_response(raw: str) -> dict:
    """Parse LLM response to a dict, stripping markdown fences first."""
    cleaned = _strip_fences(raw)
    result = json.loads(cleaned)  # raises json.JSONDecodeError if malformed
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object, got {type(result).__name__}")
    return result


_REPAIR_PREFIX = (
    "Your previous response was not valid JSON. "
    "Return ONLY a valid JSON object matching the schema below. "
    "No markdown fences, no prose, no code fences. Just the JSON object:\n\n"
)


def extract_module(module_name: str, system_content: str, user_content: str, client: Any) -> dict:
    """Call client, parse response. Retries once on malformed JSON.

    Raises json.JSONDecodeError or ValueError if both attempts fail.
    """
    raw = client.call(system_content, user_content)
    try:
        return _parse_module_response(raw)
    except (json.JSONDecodeError, ValueError):
        repair_user = _REPAIR_PREFIX + user_content
        raw2 = client.call(system_content, repair_user)
        return _parse_module_response(raw2)  # propagates on second failure


# ---------------------------------------------------------------------------
# Empty module stubs (for modules not in the extraction run)
# ---------------------------------------------------------------------------

def _empty_module(module_name: str) -> dict:
    """Return a minimal schema-valid stub for an unextracted module."""
    base: dict = {"summary": None, "evidence": []}
    if module_name == "research_question":
        base.update({
            "disease_context": [], "biological_context": [],
            "main_question": None, "stated_gap": None, "study_type": [],
        })
    elif module_name == "study_design":
        base.update({
            "design_features": [], "comparison_groups": [],
            "discovery_validation_structure": None, "perturbation_or_intervention": None,
        })
    elif module_name == "mechanism_model":
        base["claims"] = []
    else:
        base["items"] = []
    return base


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_card(card: dict) -> list[str]:
    """Return validation error strings; empty list means the card is valid."""
    try:
        from jsonschema import Draft7Validator
    except ImportError:
        return ["jsonschema not installed — schema validation skipped"]
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(card), key=lambda e: list(e.path))
    return [
        f"{'> '.join(str(p) for p in e.absolute_path) or '(root)'}: {e.message}"
        for e in errors
    ]


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_article_card(
    article_id: str,
    *,
    modules: list[str] | None = None,
    overwrite: bool = False,
    client: Any = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    data_root: Path | None = None,
) -> dict:
    """Extract an Article Literature Card for *article_id*.

    Parameters
    ----------
    article_id:   Article folder name, e.g. 'pdf_b984e5bc1768479a'.
    modules:      Subset of modules to extract; None means all nine.
    overwrite:    Overwrite existing article_card.json if present.
    client:       Injectable LLM client (for testing). If None, a real
                  DeepSeekClient is created from env vars / parameters.
    model:        LLM model name.
    base_url:     LLM API base URL.
    api_key:      API key; falls back to DEEPSEEK_API_KEY / OPENAI_API_KEY.
    data_root:    Override data directory root (for testing).

    Returns
    -------
    The assembled article_card dict (also written to disk).
    """
    root = data_root or (REPO_ROOT / "data")
    modules_to_run: list[str] = modules if modules is not None else ALL_MODULES

    invalid = [m for m in modules_to_run if m not in ALL_MODULES]
    if invalid:
        raise ValueError(f"Unknown module names: {invalid}. Valid: {ALL_MODULES}")

    card_dir = root / "literature_cards" / article_id
    modules_dir = card_dir / "modules"
    card_path = card_dir / "article_card.json"
    manifest_path = card_dir / "extraction_manifest.json"

    if card_path.exists() and not overwrite:
        raise FileExistsError(
            f"article_card.json already exists for '{article_id}'. "
            "Use --overwrite to replace it."
        )

    card_dir.mkdir(parents=True, exist_ok=True)
    modules_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    parents = load_parents(article_id, root)
    reading_text = assemble_reading_text(parents)
    metadata_raw = load_metadata(article_id, root)
    metadata_block = build_metadata_block(metadata_raw)
    token_estimate = max(1, len(reading_text.split()))

    # Build LLM client if not injected
    if client is None:
        resolved_key = (
            api_key
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        client = DeepSeekClient(model=model, api_key=resolved_key, base_url=base_url)

    # Build system content once — article text is the same for all 9 module calls.
    # Keeping it in the system message enables server-side prompt caching.
    system_content = get_article_reading_system_prompt(reading_text)

    # Extract modules
    module_results: dict[str, dict] = {}
    warnings: list[str] = []
    completed: list[str] = []

    for module_name in ALL_MODULES:
        if module_name not in modules_to_run:
            module_results[module_name] = _empty_module(module_name)
            continue

        user_content = get_article_card_module_question(module_name)
        result = extract_module(module_name, system_content, user_content, client)  # raises on double failure

        module_results[module_name] = result
        completed.append(module_name)

        mod_path = modules_dir / f"{module_name}.json"
        with open(mod_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # Assemble article card
    now = datetime.now(timezone.utc).isoformat()
    card: dict = {
        "schema_version": SCHEMA_VERSION,
        "article_id": article_id,
        "metadata": metadata_block,
        "modules": module_results,
        "extraction_metadata": {
            "extraction_method": f"llm:{model}",
            "created_at": now,
            "source_files": [
                str(root / "index" / article_id / "parents.jsonl"),
            ],
            "module_status": {
                m: ("completed" if m in completed else "skipped")
                for m in ALL_MODULES
            },
            "warnings": warnings,
        },
    }

    # Validate
    validation_errors = validate_card(card)
    validation_status = "passed" if not validation_errors else "failed"
    if validation_errors:
        warnings.extend([f"schema: {e}" for e in validation_errors])
        card["extraction_metadata"]["warnings"] = warnings

    # Write article card
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2, ensure_ascii=False)

    # Write manifest
    manifest: dict = {
        "article_id": article_id,
        "input_paths": {
            "parents_jsonl": str(root / "index" / article_id / "parents.jsonl"),
            "schema": str(SCHEMA_PATH),
            "metadata_json": str(root / "articles" / article_id / "metadata.json"),
        },
        "output_paths": {
            "article_card": str(card_path),
            "modules_dir": str(modules_dir),
            "manifest": str(manifest_path),
        },
        "modules_requested": modules_to_run,
        "modules_completed": completed,
        "llm_model": model,
        "created_at": now,
        "article_reading_parent_count": len(parents),
        "article_reading_token_count_estimate": token_estimate,
        "schema_validation_status": validation_status,
        "warnings": warnings,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return card


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract an Article Literature Card for one article."
    )
    parser.add_argument("--article-id", required=True, help="Article folder name")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing outputs")
    parser.add_argument("--modules", nargs="+", default=None, metavar="MODULE",
                        help="Extract only these modules (default: all nine)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="LLM API base URL")
    args = parser.parse_args()

    card = extract_article_card(
        args.article_id,
        modules=args.modules,
        overwrite=args.overwrite,
        model=args.model,
        base_url=args.base_url,
    )

    card_dir = REPO_ROOT / "data" / "literature_cards" / args.article_id
    manifest_path = card_dir / "extraction_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"article_id:       {args.article_id}")
    print(f"modules done:     {len(manifest['modules_completed'])}/{len(manifest['modules_requested'])} requested")
    print(f"parents read:     {manifest['article_reading_parent_count']}")
    print(f"token estimate:   {manifest['article_reading_token_count_estimate']}")
    print(f"schema:           {manifest['schema_validation_status']}")
    print(f"article_card:     {manifest['output_paths']['article_card']}")
    if manifest["warnings"]:
        for w in manifest["warnings"]:
            print(f"WARNING:          {w}")


if __name__ == "__main__":
    main()
