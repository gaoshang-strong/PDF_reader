# Step 4B: Article Card Module Prompt Library

## Goal

Define the prompt templates that instruct an LLM to extract one Article Card
module at a time from a scientific article. Step 4C imports this library and
uses it to drive all nine LLM extraction calls.

This step is **LLM-free** — it only provides and tests prompt strings.

---

## Key Idea

Each module has its own extraction goal, expert reading questions, shared
extraction rules, evidence structure, schema notes, and a JSON skeleton
showing the exact output shape required.

**Two prompt forms per module** are provided to support prompt caching:

| Function | Form | Use |
|---|---|---|
| `get_article_card_module_prompt(module, text)` | Full single-message prompt (article text embedded) | Backward compatibility |
| `get_article_reading_system_prompt(text)` | System message: instructions + full article text | Cached extraction (Step 4C) |
| `get_article_card_module_question(module)` | User message: module question only, no article text | Cached extraction (Step 4C) |

In Step 4C, the system message (article text) is sent once and cached
server-side; subsequent module calls reuse the cache, reducing input token
cost by ~80–90%.

**Common rules enforced in every prompt:**

| Rule | Instruction |
|---|---|
| JSON only | Return a JSON object — no fences, no prose |
| Module scope | Return only this module object |
| Schema fidelity | Use only the fields in the skeleton |
| No invention | Use `null` or `[]` when not found in the text |
| Evidence anchoring | Use only `parent_id` values from the article reading text verbatim |
| No critique | Do not judge evidence quality or study design |
| No hypotheses | Do not generate interpretations beyond what the paper states |
| Non-null summary | Write a 1–3 sentence summary if the module has any extracted content |
| Evidence limits | At most 5 module-level evidence objects; at most 2 per item or claim |

---

## Inputs

None — this is a pure Python module with no file I/O.

---

## Outputs

Prompt strings returned by the public API functions. Nothing is written to disk.

---

## How to Run

Used as a library by Step 4C. Not directly executable.

---

## Python API

```python
from pipeline.step4B_article_card_prompts import (
    list_article_card_modules,
    get_article_card_module_prompt,
    get_article_card_module_question,
    get_article_reading_system_prompt,
)

# List all nine module names in order
modules = list_article_card_modules()

# --- Option A: single-message prompt (article text embedded) ---
prompt = get_article_card_module_prompt("key_findings", article_reading_text)

# --- Option B: cached 2-message design ---
system = get_article_reading_system_prompt(article_reading_text)  # built once
user   = get_article_card_module_question("key_findings")          # per module

# Unknown module raises ValueError
get_article_card_module_prompt("bad_name", text)  # → ValueError
```

---

## The Nine Modules

### A. `research_question`
**Goal:** Biomedical problem, central question, stated knowledge gap, study type.
**Fields:** `summary`, `disease_context`, `biological_context`, `main_question`, `stated_gap`, `study_type`, `evidence`

### B. `study_design`
**Goal:** Overall structure, comparison groups, discovery/validation split, interventions.
**Fields:** `summary`, `design_features` (list of strings), `comparison_groups` (list with `group_a`, `group_b`, `comparison_purpose`), `discovery_validation_structure` (string), `perturbation_or_intervention` (string), `evidence`
**Note:** `design_features`, `discovery_validation_structure`, and `perturbation_or_intervention` must be plain strings/string lists — not objects.

### C. `samples_and_cohorts`
**Goal:** All cohorts, models, and biological materials — species, material, size, usage.
**Fields:** `summary`, `items` (each: `sample_or_cohort_name`, `species`, `disease_or_condition`, `sample_material`, `sample_count`, `used_for`, …), `evidence`

### D. `data_source_and_provenance`
**Goal:** All datasets — generated vs. external, accession IDs, usage.
**Fields:** `summary`, `items` (each: `dataset_name`, `data_modality`, `source_type`, `accession_id`, …), `evidence`
**`source_type` enum:** `generated_in_this_study | external_public_data | external_published_cohort | external_database | unclear`

### E. `omics_and_experimental_methods`
**Goal:** All assays and wet-lab methods — modality, sample material, purpose.
**Fields:** `summary`, `items` (each: `method`, `modality_category`, `sample_material`, `purpose`, …), `evidence`
**`modality_category` guidance:** `transcriptomics`, `single_cell_transcriptomics`, `spatial_transcriptomics`, `proteomics`, `metabolomics`, `epigenomics`, `immune_profiling`, `flow_cytometry`, `imaging`, `functional_assay`, `perturbation`, `in_vivo_model`, `computational_reanalysis`, `clinical_data`, `other`

### F. `computational_analysis_pipeline`
**Goal:** Computational steps — software, parameters, thresholds.
**Fields:** `summary`, `items` (each: `analysis_step`, `software_or_method`, `parameters_or_thresholds`, …), `evidence`

### G. `key_findings`
**Goal:** Main findings and conclusions — supporting data, entities, result type.
**Fields:** `summary`, `items` (each: `finding`, `result_type`, …), `evidence`

### H. `mechanism_model`
**Goal:** Mechanistic model (if any) — causal chain, mediators, cell types, therapeutic implication. `claims: []` if no mechanism proposed.
**Fields:** `summary`, `claims` (each: `mechanism_chain`, `upstream_factor`, `mediators`, `downstream_effect`, `involved_cell_types`, `involved_pathways_or_molecules`, `therapeutic_or_clinical_implication`, …), `evidence`

### I. `author_reported_limitations_and_future_directions`
**Goal:** Only what the authors explicitly report as limitations, caveats, or future directions.
**Fields:** `summary`, `items` (each: `type`, `text`, …), `evidence`
**`type` enum:** `limitation | caveat | future_direction`
