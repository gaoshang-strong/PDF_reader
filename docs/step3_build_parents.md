# Step 3B-1: Build Parent Chunks

## Purpose

Step 3B-1 groups Step 3A structured blocks into **parent chunks** — the
retrieval unit for the RAG pipeline. Parents are semantic chunks of text that
respect document section boundaries and stay within configurable token limits.

No text is rewritten, summarized, or otherwise modified. Parent text is formed
exclusively by joining source block texts with two newlines (`\n\n`).

## Input / Output

| | Path |
|---|---|
| Input | `data/index/{article_id}/structured_blocks.jsonl` |
| Output parents | `data/index/{article_id}/parents.jsonl` |
| Output manifest | `data/index/{article_id}/parent_manifest.json` |

## Parent Schema

```json
{
  "parent_id": "pdf_abc::parent_000003",
  "article_id": "pdf_abc",
  "parent_index": 3,
  "chunk_level": "parent",
  "chunk_type": "body",
  "section_path": ["Text", "Methods", "Statistics"],
  "source_block_ids": ["pdf_abc_000010", "pdf_abc_000011"],
  "text": "First paragraph text.\n\nSecond paragraph text.",
  "char_start": 4210,
  "char_end": 4890,
  "token_count": 87,
  "oversized": false,
  "include_in_default_qa": true
}
```

### Field Descriptions

| Field | Type | Description |
|---|---|---|
| `parent_id` | string | `{article_id}::parent_{index:06d}` — globally unique |
| `article_id` | string | Article folder name |
| `parent_index` | int | 0-based sequential index within the article |
| `chunk_level` | string | Always `"parent"` |
| `chunk_type` | string | One of `body`, `figure_caption`, `table_caption`, `back_matter` |
| `section_path` | list[str] | Taken from the first source block |
| `source_block_ids` | list[str] | Block IDs from Step 3A that form this parent |
| `text` | string | Source block texts joined with `\n\n` (no other modification) |
| `char_start` | int | `char_start` of the first source block in `normalized.md` |
| `char_end` | int | `char_end` of the last source block in `normalized.md` |
| `token_count` | int | Word-count estimate of joined text |
| `oversized` | bool | `true` if a single block exceeded `parent_max_tokens` |
| `include_in_default_qa` | bool | `false` for `back_matter`; `true` for all other types |

### Chunk Types

| Type | Source block types | `include_in_default_qa` |
|---|---|---|
| `body` | `paragraph` | `true` |
| `back_matter` | `back_matter` content (heading_level=None) | `false` |
| `figure_caption` | `figure_caption` | `true` |
| `table_caption` | `table_caption` | `true` |

## Default Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `parent_target_tokens` | 1200 | Soft target size (informational) |
| `parent_max_tokens` | 1800 | Hard limit; triggers a new parent when exceeded |
| `parent_min_tokens` | 200 | Floor used in auditing to flag very small parents |

## Grouping Rules

Blocks are processed in `block_index` order (document order).

1. **Skipped block types**: `heading`, `title`, `reference`, and `back_matter`
   blocks whose `heading_level` is not `null` (section heading markers).

2. **Caption blocks** (`figure_caption`, `table_caption`): each becomes a
   single-block parent with `chunk_type` matching the block type.

3. **Body paragraphs** (`block_type = "paragraph"`):
   - Accumulated into the current parent if `section_path` matches and adding
     the block would not exceed `parent_max_tokens`.
   - A new parent is started when:
     - `section_path` changes, OR
     - `current_tokens + block_tokens > parent_max_tokens`
   - If a single block exceeds `parent_max_tokens` by itself, it becomes a
     single-block parent marked `"oversized": true`.

4. **Back matter content** (`block_type = "back_matter"`, `heading_level = null`):
   grouped identically to body paragraphs but assigned `chunk_type = "back_matter"`
   and `include_in_default_qa = false`.

5. **References** (`block_type = "reference"`): skipped entirely — no parent
   is created for reference entries in Step 3B-1.

6. **Section boundary enforcement**: parents never span two different
   `section_path` values. A change in `section_path` always flushes the
   current accumulator before starting a new parent.

## Text Preservation

Parent text is formed by:
```python
text = "\n\n".join(block["text"] for block in source_blocks)
```

No spelling correction, normalization, summarization, or rewriting occurs.
The `char_start` and `char_end` fields point to the corresponding range in
the original `normalized.md` file (via the source block metadata).

## Running Step 3B-1

```bash
# Single article
python pipeline/step3_build_parents.py --article-id pdf_16edbbde296287d6

# All articles
python pipeline/step3_build_parents.py --all

# Custom token limit
python pipeline/step3_build_parents.py --all --parent-max-tokens 2000 --overwrite
```

## Auditing

```bash
python scripts/audit_parent_chunks.py
python scripts/audit_parent_chunks.py --article-id pdf_16edbbde296287d6
python scripts/audit_parent_chunks.py --json
```

## Known Limitations

- **`parent_min_tokens` is not enforced**: small parents (e.g., one short
  paragraph) are kept as-is rather than merged across section boundaries.
  The audit flags them but does not fix them.
- **References are excluded**: Step 3B-1 skips reference blocks. A future step
  may create reference parents for citation retrieval.
- **`# Tables` paragraphs**: markdown table rows classified as `paragraph` by
  Step 3A become body parents. They may contain pipe-delimited text, which
  is awkward but preserved verbatim.
- **Token count approximation**: `token_count` uses word count, not subword
  tokenization. Actual LLM token counts will vary for scientific notation
  and LaTeX fragments.
- **`parent_target_tokens`** is stored in the manifest for reference but is not
  currently used during grouping (only `parent_max_tokens` is enforced).
