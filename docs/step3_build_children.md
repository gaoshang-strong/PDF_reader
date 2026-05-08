# Step 3B-2: Build Child Chunks

## Goal

Split Step 3B-1 parent chunks into **child chunks** — the primary retrieval
unit for embedding and BM25 search. Children fit within embedding-model
context limits and carry pre-formatted text fields ready for indexing.

---

## Key Idea

Children are created by **greedy sentence grouping** within each parent:

1. Decompose parent text into paragraphs → sentences → words.
2. Group sentences greedily up to `child_max_tokens`.
3. Add an **overlap prefix** to each non-first child so that context from
   the previous child is preserved across chunk boundaries.
4. Never cross a parent boundary — overlap does not carry over between
   different parents.

No text is rewritten. Child text is assembled only from substrings of the
parent text. Two additional text fields are pre-formatted for indexing:
`text_for_embedding` (prefixed with section path) and `text_for_bm25`
(section path tokens prepended for keyword search).

---

## Inputs

| Source | Path |
|---|---|
| Parent chunks | `data/index/{article_id}/parents.jsonl` |

---

## Outputs

| File | Path | Description |
|---|---|---|
| Child chunks | `data/index/{article_id}/children.jsonl` | One JSON object per line |
| Manifest | `data/index/{article_id}/child_manifest.json` | Counts and parameters used |

### Child Schema

```json
{
  "chunk_id": "pdf_abc::child_000012",
  "parent_id": "pdf_abc::parent_000003",
  "article_id": "pdf_abc",
  "child_index": 12,
  "child_index_within_parent": 1,
  "chunk_level": "child",
  "chunk_type": "body",
  "section_path": ["Text", "Methods", "Statistics"],
  "text": "Child text assembled from parent.",
  "text_for_embedding": "Section: Text > Methods > Statistics\n\nChild text assembled from parent.",
  "text_for_bm25": "Text Methods Statistics Child text assembled from parent.",
  "char_start": 4210,
  "char_end": 4890,
  "token_count": 87,
  "oversized": false,
  "include_in_default_qa": true,
  "span_is_approximate": false
}
```

| Field | Description |
|---|---|
| `chunk_id` | `{article_id}::child_{index:06d}` — globally unique |
| `parent_id` | The parent chunk this child comes from |
| `text` | Raw child text (substring of parent text) |
| `text_for_embedding` | `"Section: {path}\n\n{text}"` — used for dense embedding |
| `text_for_bm25` | `"{path tokens} {text}"` — used for sparse keyword search |
| `span_is_approximate` | `true` when the overlap prefix shifts char offsets |

---

## Default Parameters

| Parameter | Default | Description |
|---|---|---|
| `child_max_tokens` | 550 | Hard limit per child |
| `child_target_tokens` | 380 | Soft target (informational) |
| `child_min_tokens` | 120 | Audit floor |
| `child_overlap_tokens` | 60 | Overlap prefix size from the previous child |

---

## How to Run

```bash
# Single article
python pipeline/step3_build_children.py --article-id pdf_16edbbde296287d6

# All articles
python pipeline/step3_build_children.py --all

# Custom token limits
python pipeline/step3_build_children.py --all --child-max-tokens 600 --child-overlap-tokens 80 --overwrite
```

---

## Python API

```python
from pathlib import Path
from pipeline.step3_build_children import split_parent, process_article

# Split a single parent dict into children
children = split_parent(parent, article_id="pdf_abc", start_child_index=0,
                        child_max_tokens=550, child_overlap_tokens=60)

# Process a full article (reads parents.jsonl, writes children.jsonl)
result = process_article("pdf_abc", data_root=Path("data"), overwrite=True)
```

---

## Known Limitations

- Sentence splitting uses a regex heuristic (`(?<=[.!?])\s+(?=[A-Z"(])`); abbreviations and mid-sentence capitals may split incorrectly.
- Overlap does not cross parent boundaries.
- Token count uses word count, not subword tokenization.
- `child_min_tokens` is not enforced at build time — short children are kept.
