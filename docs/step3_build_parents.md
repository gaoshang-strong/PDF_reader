# Step 3B-1: Build Parent Chunks

## Goal

Group Step 3A structured blocks into **parent chunks** — the primary
retrieval unit for the RAG pipeline. Parents are semantic units that respect
section boundaries and stay within a configurable token limit.

---

## Key Idea

Parents are built by **greedy accumulation**: body paragraph blocks are
accumulated into one parent until either the section path changes or the
token budget is exceeded. Caption blocks become their own single-block
parent. References are excluded.

No text is rewritten or summarized. Parent text is formed only by joining
source block texts with `\n\n`.

The `parent_id` is the stable identifier used downstream — Step 4C's LLM
reads the article as a sequence of parent chunks and cites them by
`parent_id` in evidence arrays.

---

## Inputs

| Source | Path |
|---|---|
| Structured blocks | `data/index/{article_id}/structured_blocks.jsonl` |

---

## Outputs

| File | Path | Description |
|---|---|---|
| Parent chunks | `data/index/{article_id}/parents.jsonl` | One JSON object per line |
| Manifest | `data/index/{article_id}/parent_manifest.json` | Counts and parameters used |

### Parent Schema

```json
{
  "parent_id": "pdf_abc::parent_000003",
  "article_id": "pdf_abc",
  "parent_index": 3,
  "chunk_level": "parent",
  "chunk_type": "body",
  "section_path": ["Text", "Methods", "Statistics"],
  "source_block_ids": ["pdf_abc_000010", "pdf_abc_000011"],
  "text": "First paragraph.\n\nSecond paragraph.",
  "char_start": 4210,
  "char_end": 4890,
  "token_count": 87,
  "oversized": false,
  "include_in_default_qa": true
}
```

### Chunk Types

| `chunk_type` | Source block types | `include_in_default_qa` |
|---|---|---|
| `body` | `paragraph` | `true` |
| `figure_caption` | `figure_caption` | `true` |
| `table_caption` | `table_caption` | `true` |
| `back_matter` | `back_matter` content | `false` |

---

## Default Parameters

| Parameter | Default | Description |
|---|---|---|
| `parent_max_tokens` | 1800 | Hard limit; triggers a new parent when exceeded |
| `parent_target_tokens` | 1200 | Soft target (informational only) |
| `parent_min_tokens` | 200 | Audit floor; very small parents are flagged |

---

## Grouping Rules

1. **Skipped**: `heading`, `title`, `reference` blocks, and `back_matter` headings produce no parent.
2. **Captions**: each `figure_caption` or `table_caption` block → one parent.
3. **Body paragraphs**: accumulated greedily until section_path changes or `parent_max_tokens` is exceeded. A block that alone exceeds the limit becomes a single-block parent with `oversized: true`.
4. **Back matter**: same logic as body paragraphs, `chunk_type = "back_matter"`, `include_in_default_qa = false`.
5. **References**: excluded entirely.
6. **Section boundary**: a change in `section_path` always starts a new parent.

---

## How to Run

```bash
# Single article
python pipeline/step3_build_parents.py --article-id pdf_16edbbde296287d6

# All articles
python pipeline/step3_build_parents.py --all

# Custom token limit
python pipeline/step3_build_parents.py --all --parent-max-tokens 2000 --overwrite
```

---

## Python API

```python
from pathlib import Path
from pipeline.step3_build_parents import build_parents, process_article

# Build from a list of block dicts (e.g. loaded from structured_blocks.jsonl)
parents = build_parents(blocks, article_id="pdf_abc", parent_max_tokens=1800)

# Process a full article (reads JSONL, writes parents.jsonl)
result = process_article("pdf_abc", data_root=Path("data"), overwrite=True)
```

---

## Known Limitations

- `parent_min_tokens` is not enforced at build time — small parents are kept as-is.
- References are excluded; no parent is created for reference entries.
- Token count uses word count approximation, not subword tokenization.
