# Step 3B-2: Build Child Chunks

## Purpose

Step 3B-2 splits Step 3B-1 parent chunks into **child chunks** — the
primary retrieval unit for embedding and BM25 search. Children are smaller
than parents, fitting within embedding-model context limits, and carry
additional text fields ready for indexing.

No text is rewritten, summarized, or otherwise modified. Child text is
assembled exclusively from substrings of the parent text.

## Input / Output

| | Path |
|---|---|
| Input | `data/index/{article_id}/parents.jsonl` |
| Output children | `data/index/{article_id}/children.jsonl` |
| Output manifest | `data/index/{article_id}/child_manifest.json` |

## Child Schema

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

### Field Descriptions

| Field | Type | Description |
|---|---|---|
| `chunk_id` | string | `{article_id}::child_{index:06d}` — globally unique within the article |
| `parent_id` | string | ID of the parent chunk this child comes from |
| `article_id` | string | Article folder name |
| `child_index` | int | 0-based sequential index within the article (across all parents) |
| `child_index_within_parent` | int | 0-based sequential index within its parent |
| `chunk_level` | string | Always `"child"` |
| `chunk_type` | string | Inherited from parent: `body`, `figure_caption`, `table_caption`, `back_matter` |
| `section_path` | list[str] | Inherited from parent |
| `text` | string | Exact child text; may include an overlap prefix from the previous sibling |
| `text_for_embedding` | string | Section/caption prefix followed by child text (see below) |
| `text_for_bm25` | string | Section path words joined with spaces followed by child text |
| `char_start` | int | Byte offset in `normalized.md`; uses `parent.char_start` as base |
| `char_end` | int | Byte offset of last character in `normalized.md` |
| `token_count` | int | Word-count estimate of `text` (recomputed from actual child text) |
| `oversized` | bool | `true` if `token_count > child_max_tokens` |
| `include_in_default_qa` | bool | Inherited from parent (always `false` for `back_matter`) |
| `span_is_approximate` | bool | `true` when the child has an overlap prefix (span covers ≥ 2 positions in parent) |

## Default Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `child_target_tokens` | 380 | Soft target size (informational; stored in manifest) |
| `child_max_tokens` | 550 | Hard limit for the grouping algorithm |
| `child_min_tokens` | 120 | Floor used in auditing; not enforced at build time |
| `child_overlap_tokens` | 60 | Target size of the overlap prefix prepended to each non-first child |

## Splitting Algorithm

All blocks are split in `parent_index` order.  Within a parent:

1. **Decompose into atomic units** (`_build_units`):
   - Split parent text at `\n\n` into paragraphs.
   - If a paragraph fits within `child_max_tokens`, it is a single unit.
   - If a paragraph exceeds `child_max_tokens`, split by sentence boundaries
     (`(?<=[.!?])\s+(?=[A-Z"(])`).
   - If a sentence still exceeds `child_max_tokens`, split by words (hard
     `child_max_tokens`-word chunks).

2. **Group units** (`_group_units`):
   - Pack units into groups greedily: accumulate until adding the next unit
     would exceed `child_max_tokens`, then start a new group.

3. **Add overlap** (`_overlap_suffix`):
   - For each group after the first, prepend the tail of the previous group
     fitting within `child_overlap_tokens`.
   - Prefers whole paragraph/sentence units for the overlap.  If the last
     unit alone exceeds the budget, falls back to a word-level suffix.

4. **Join** within a group:
   - Two units from different paragraphs are joined with `\n\n`.
   - Two units from the same paragraph are joined with a single space.

## Text Fields

### `text`
Exact assembled child text.  For non-first children, may include an overlap
prefix from the preceding sibling.

### `text_for_embedding`

| `chunk_type` | Prefix |
|---|---|
| `body` / `back_matter` | `Section: PATH1 > PATH2\n\n{text}` |
| `figure_caption` | `Figure caption: PATH1 > PATH2\n\n{text}` |
| `table_caption` | `Table caption: PATH1 > PATH2\n\n{text}` |

### `text_for_bm25`
`"PATH1 PATH2 ... {text}"` — section path joined with spaces, then the child text.

## Character Spans

- `char_start` and `char_end` are byte offsets into the article's
  `normalized.md` file.
- Computed by finding `child.text` as an exact substring of `parent.text`
  (deterministic substring search), then adding `parent.char_start` as the
  base offset.
- When the child includes an overlap prefix: `span_is_approximate = true`
  because the span deliberately covers text from a preceding semantic region.
- When no overlap: `span_is_approximate = false` (the span is an exact match
  in the parent string).
- Fallback (substring search fails): paragraph-level bounds from the source
  units are used and `span_is_approximate = true`.

## Running Step 3B-2

```bash
# Single article
python pipeline/step3_build_children.py --article-id pdf_16edbbde296287d6

# All articles
python pipeline/step3_build_children.py --all

# Custom limits
python pipeline/step3_build_children.py --all --child-max-tokens 600 --overwrite
```

## Auditing

```bash
python scripts/audit_child_chunks.py
python scripts/audit_child_chunks.py --article-id pdf_16edbbde296287d6
python scripts/audit_child_chunks.py --json
```

## Known Limitations

- **`child_min_tokens` is not enforced**: very short children (e.g., a
  single short caption sentence) are kept as-is.
- **Sentence splitting is heuristic**: the regex
  `(?<=[.!?])\s+(?=[A-Z"(])` does not handle abbreviations such as
  "Dr.", "et al.", or "U.S.A." correctly — it may split mid-sentence.
- **Token count uses word count**: `max(1, len(text.split()))` differs from
  subword tokenisation; actual LLM token counts will vary for scientific
  notation and LaTeX fragments.
- **`child_target_tokens` is informational**: stored in the manifest but not
  used during splitting.
- **Overlap does not cross parent boundaries**: each child belongs to exactly
  one parent.
- **Overlap children may exceed `child_max_tokens`**: when a core segment is
  near the token limit and the overlap prefix is prepended, the combined
  `token_count` can exceed `child_max_tokens` by up to `child_overlap_tokens`
  words.  The audit flags these as WARN; this is expected and not a data
  integrity issue.
