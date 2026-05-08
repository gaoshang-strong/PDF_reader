#!/usr/bin/env python3
"""Step 4 batch runner: extract and render Article Cards for all articles.

For each article found in data/index/, runs Step 4C (LLM extraction) and
Step 4D (Markdown rendering). Reports per-article and aggregate token usage,
including cache hits when the provider supports prompt caching.

Usage
-----
  # Run all articles
  python pipeline/step4_batch_run.py

  # Run specific articles
  python pipeline/step4_batch_run.py --article-ids pdf_abc pdf_def

  # Re-extract already-processed articles
  python pipeline/step4_batch_run.py --overwrite

  # Extract only (skip Markdown rendering)
  python pipeline/step4_batch_run.py --skip-render
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step4C_extract_article_card import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DeepSeekClient,
    extract_article_card,
)
from pipeline.step4D_render_article_card_md import render_article_card_file


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_article_ids(data_root: Path) -> list[str]:
    """Return sorted list of article IDs that have a parents.jsonl file."""
    index_dir = data_root / "index"
    if not index_dir.exists():
        return []
    return sorted(
        d.name
        for d in index_dir.iterdir()
        if d.is_dir() and (d / "parents.jsonl").exists()
    )


# ---------------------------------------------------------------------------
# Per-article result
# ---------------------------------------------------------------------------

@dataclass
class ArticleResult:
    article_id: str
    status: str           # "ok" | "skipped" | "failed"
    modules_done: int = 0
    modules_total: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    elapsed_s: float = 0.0
    error: str = ""
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-article runner
# ---------------------------------------------------------------------------

def run_one(
    article_id: str,
    client: DeepSeekClient,
    data_root: Path,
    overwrite: bool,
    skip_render: bool,
) -> ArticleResult:
    result = ArticleResult(article_id=article_id, status="failed")
    client.reset_usage()
    t0 = time.monotonic()

    try:
        card = extract_article_card(
            article_id,
            overwrite=overwrite,
            client=client,
            data_root=data_root,
        )
        em = card.get("extraction_metadata", {})
        ms = em.get("module_status", {})
        result.modules_done = sum(1 for s in ms.values() if s == "completed")
        result.modules_total = len(ms)
        result.warnings = em.get("warnings", [])

        if not skip_render:
            render_article_card_file(article_id, overwrite=overwrite, data_root=data_root)

        result.status = "ok"

    except FileExistsError:
        result.status = "skipped"

    except Exception as exc:
        result.status = "failed"
        result.error = f"{type(exc).__name__}: {exc}"

    result.elapsed_s = time.monotonic() - t0
    result.prompt_tokens = client.usage["prompt_tokens"]
    result.cached_tokens = client.usage["cached_tokens"]
    result.completion_tokens = client.usage["completion_tokens"]
    result.llm_calls = client.usage["calls"]
    return result


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_tokens(n: int) -> str:
    return f"{n:,}"


def _cache_pct(cached: int, prompt: int) -> str:
    if prompt == 0:
        return "—"
    return f"{100 * cached / prompt:.1f}%"


def _print_header(n: int) -> None:
    print(f"\nRunning Step 4 (extract + render) for {n} article{'s' if n != 1 else ''}...\n", flush=True)
    col = "{:<5} {:<32} {:<8} {:<6} {:>12} {:>12} {:>12} {:>8} {:>7}"
    print(col.format("#", "article_id", "status", "mods", "prompt", "cached", "completion", "calls", "time"), flush=True)
    print("-" * 105, flush=True)


def _print_row(i: int, total: int, r: ArticleResult) -> None:
    col = "{:<5} {:<32} {:<8} {:<6} {:>12} {:>12} {:>12} {:>8} {:>7}"
    mods = f"{r.modules_done}/{r.modules_total}" if r.status == "ok" else "—"
    prompt = _fmt_tokens(r.prompt_tokens) if r.status == "ok" else "—"
    cached = _fmt_tokens(r.cached_tokens) if r.status == "ok" else "—"
    completion = _fmt_tokens(r.completion_tokens) if r.status == "ok" else "—"
    calls = str(r.llm_calls) if r.status == "ok" else "—"
    elapsed = f"{r.elapsed_s:.1f}s"
    print(col.format(f"[{i}/{total}]", r.article_id, r.status, mods, prompt, cached, completion, calls, elapsed), flush=True)
    if r.status == "failed":
        print(f"      ERROR: {r.error}", flush=True)
    for w in r.warnings:
        print(f"      WARN:  {w}", flush=True)


def _print_summary(results: list[ArticleResult]) -> None:
    ok = [r for r in results if r.status == "ok"]
    skipped = [r for r in results if r.status == "skipped"]
    failed = [r for r in results if r.status == "failed"]

    total_prompt = sum(r.prompt_tokens for r in ok)
    total_cached = sum(r.cached_tokens for r in ok)
    total_completion = sum(r.completion_tokens for r in ok)
    total_calls = sum(r.llm_calls for r in ok)
    total_time = sum(r.elapsed_s for r in results)

    print("\n" + "=" * 105)
    print("Summary")
    print(f"  Articles:    {len(ok)} ok  {len(failed)} failed  {len(skipped)} skipped")
    print(f"  LLM calls:   {total_calls}")
    print(f"  Prompt:      {_fmt_tokens(total_prompt)} tokens  "
          f"(cached: {_fmt_tokens(total_cached)} = {_cache_pct(total_cached, total_prompt)})")
    print(f"  Completion:  {_fmt_tokens(total_completion)} tokens")
    print(f"  Wall time:   {total_time:.1f}s")

    if failed:
        print("\nFailed articles:")
        for r in failed:
            print(f"  {r.article_id}: {r.error}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-run Step 4 (extract + render) for all articles."
    )
    parser.add_argument(
        "--article-ids", nargs="+", metavar="ID", default=None,
        help="Article IDs to process. Default: all articles in data/index/.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing article_card.json and article_card.md.",
    )
    parser.add_argument(
        "--skip-render", action="store_true",
        help="Skip Step 4D Markdown rendering.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="LLM API base URL.")
    args = parser.parse_args()

    data_root = REPO_ROOT / "data"

    if args.article_ids:
        article_ids = args.article_ids
    else:
        article_ids = discover_article_ids(data_root)
        if not article_ids:
            print("No articles found in data/index/. Nothing to do.")
            sys.exit(0)

    resolved_key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY", "")
    )
    client = DeepSeekClient(model=args.model, api_key=resolved_key, base_url=args.base_url)

    _print_header(len(article_ids))
    results: list[ArticleResult] = []

    for i, article_id in enumerate(article_ids, 1):
        r = run_one(
            article_id,
            client=client,
            data_root=data_root,
            overwrite=args.overwrite,
            skip_render=args.skip_render,
        )
        results.append(r)
        _print_row(i, len(article_ids), r)

    _print_summary(results)


if __name__ == "__main__":
    main()
