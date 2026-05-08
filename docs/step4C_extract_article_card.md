# Step 4C: Article Card Module Extraction

## Goal

Call the configured LLM nine times — once per Article Card module — to
extract structured information from a single article's parent chunks. Write
one JSON file per module, assemble the complete `article_card.json`, validate
it against the schema, and write an `extraction_manifest.json`.

---

## Key Idea

**The LLM does expert reading; the code does validation, merging, and I/O.**

There are no deterministic heuristics (regex, keyword lookup, section-name
matching) for extracting content into the Article Card fields. Every
extraction decision is delegated entirely to the LLM. The code's job is to:

1. Assemble a clean, ordered article reading text from `parents.jsonl`.
2. Issue nine focused per-module prompts (from Step 4B).
3. Parse and validate each LLM response.
4. Merge the nine modules into `article_card.json` and validate against the schema.
5. Write outputs and record provenance in the manifest.

**Prompt caching:** The article reading text lives in the system message
(constant across all 9 calls). Only the module question changes in the user
message. From call 2 onwards the provider returns a cache hit on the ~28k
token system prefix, reducing effective input token cost by ~80–90%.

**Retry on malformed JSON:** If the LLM returns invalid JSON, one retry is
made with a repair prefix. A second failure propagates as an exception.

---

## Inputs

| Source | Path |
|---|---|
| Parent chunks | `data/index/{article_id}/parents.jsonl` |
| JSON schema | `schemas/article_card.schema.json` |
| Metadata (optional) | `data/working/{article_id}/metadata.json` |

### How `article_reading_text` Is Assembled

All parent chunks are sorted by `parent_index` and concatenated. Each block:

```
parent_id: pdf_abc::parent_000001
section_path: Text > Methods > Patient cohort
chunk_type: body
We enrolled 82 treatment-naive CRC patients...

---

parent_id: pdf_abc::parent_000002
section_path: Text > Results
chunk_type: body
CD8+ T cells were significantly reduced in PNI+ tumors...
```

The LLM reads the article in section order and cites `parent_id` values
verbatim in evidence arrays.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | Primary API key |
| `OPENAI_API_KEY` | — | Fallback API key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | LLM endpoint |
| `LLM_MODEL` | `deepseek-v4-flash` | Model name |

---

## Outputs

| File | Path | Description |
|---|---|---|
| Per-module JSON | `data/literature_cards/{article_id}/modules/{module}.json` | Raw LLM output per module |
| Article card | `data/literature_cards/{article_id}/article_card.json` | Complete assembled card |
| Manifest | `data/literature_cards/{article_id}/extraction_manifest.json` | Provenance and token stats |

### Extraction Manifest Fields

| Field | Description |
|---|---|
| `article_id` | Article identifier |
| `modules_requested` | List of modules requested |
| `modules_completed` | Modules successfully extracted |
| `llm_model` | Model name used |
| `article_reading_parent_count` | Number of parent chunks in reading text |
| `article_reading_token_count_estimate` | Word-count estimate of reading text |
| `schema_validation_status` | `"passed"` or `"failed"` |
| `warnings` | Non-fatal issues (schema errors, etc.) |

---

## How to Run

```bash
export DEEPSEEK_API_KEY=sk-...

# Extract all nine modules
python pipeline/step4C_extract_article_card.py --article-id pdf_b984e5bc1768479a

# Overwrite existing outputs
python pipeline/step4C_extract_article_card.py --article-id pdf_b984e5bc1768479a --overwrite

# Extract only specific modules
python pipeline/step4C_extract_article_card.py \
  --article-id pdf_b984e5bc1768479a \
  --modules research_question key_findings mechanism_model

# Custom model / endpoint
python pipeline/step4C_extract_article_card.py \
  --article-id pdf_b984e5bc1768479a \
  --model deepseek-chat \
  --base-url https://api.deepseek.com \
  --overwrite
```

---

## Python API

```python
from pathlib import Path
from pipeline.step4C_extract_article_card import extract_article_card

# Production (reads DEEPSEEK_API_KEY from env)
card = extract_article_card("pdf_b984e5bc1768479a", overwrite=True)

# Testing (inject fake client, redirect data root)
card = extract_article_card(
    "pdf_test",
    client=my_fake_client,
    data_root=Path("/tmp/test_data"),
    overwrite=True,
)

# Partial extraction
card = extract_article_card(
    "pdf_b984e5bc1768479a",
    modules=["key_findings", "mechanism_model"],
    overwrite=True,
)
```

### `DeepSeekClient`

```python
from pipeline.step4C_extract_article_card import DeepSeekClient

client = DeepSeekClient(model="deepseek-v4-flash", api_key="sk-...", base_url="https://api.deepseek.com")

# Make a call (system = article text, user = module question)
response_text = client.call(system_content, user_content)

# Read accumulated token usage
print(client.usage)
# {"prompt_tokens": 132545, "cached_tokens": 96768, "completion_tokens": 19027, "calls": 9}

# Reset before next article
client.reset_usage()
```
