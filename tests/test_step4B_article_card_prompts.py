"""Tests for the Article Card module prompt library (Step 4B).

No LLM is called. All tests operate on the returned prompt strings.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.step4B_article_card_prompts import (
    get_article_card_module_prompt,
    list_article_card_modules,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas" / "article_card.schema.json"

_SAMPLE_TEXT = (
    "parent_id: pdf_abc::parent_000001\n"
    "section_path: Text > Introduction\n"
    "Perineural invasion (PNI) is associated with poor prognosis in colorectal cancer.\n\n"
    "parent_id: pdf_abc::parent_000002\n"
    "section_path: Text > Methods\n"
    "We collected 82 tumor biopsies from treatment-naive CRC patients.\n"
)

ALL_MODULES = [
    "research_question",
    "study_design",
    "samples_and_cohorts",
    "data_source_and_provenance",
    "omics_and_experimental_methods",
    "computational_analysis_pipeline",
    "key_findings",
    "mechanism_model",
    "author_reported_limitations_and_future_directions",
]


# ---------------------------------------------------------------------------
# 1. All nine modules are listed
# ---------------------------------------------------------------------------

def test_list_modules_returns_all_nine():
    modules = list_article_card_modules()
    assert len(modules) == 9


def test_list_modules_contains_expected_names():
    modules = list_article_card_modules()
    for m in ALL_MODULES:
        assert m in modules, f"Module '{m}' missing from list_article_card_modules()"


def test_list_modules_returns_list():
    assert isinstance(list_article_card_modules(), list)


def test_list_modules_is_independent_copy():
    m1 = list_article_card_modules()
    m2 = list_article_card_modules()
    m1.append("fake_module")
    assert "fake_module" not in list_article_card_modules()
    assert m1 != m2


# ---------------------------------------------------------------------------
# 2. Prompt can be generated for every module
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", ALL_MODULES)
def test_get_prompt_returns_string_for_all_modules(module):
    prompt = get_article_card_module_prompt(module, _SAMPLE_TEXT)
    assert isinstance(prompt, str)
    assert len(prompt) > 100


# ---------------------------------------------------------------------------
# 3. Unknown module raises ValueError
# ---------------------------------------------------------------------------

def test_unknown_module_raises_value_error():
    with pytest.raises(ValueError, match="Unknown module"):
        get_article_card_module_prompt("nonexistent_module", _SAMPLE_TEXT)


def test_empty_string_module_raises_value_error():
    with pytest.raises(ValueError):
        get_article_card_module_prompt("", _SAMPLE_TEXT)


def test_close_but_wrong_module_name_raises_value_error():
    with pytest.raises(ValueError):
        get_article_card_module_prompt("key_finding", _SAMPLE_TEXT)  # missing 's'


# ---------------------------------------------------------------------------
# 4. Each prompt includes the provided article_reading_text
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", ALL_MODULES)
def test_prompt_includes_article_reading_text(module):
    unique_text = f"UNIQUE_MARKER_{module}_xyz_12345"
    prompt = get_article_card_module_prompt(module, unique_text)
    assert unique_text in prompt, f"article_reading_text not found in [{module}] prompt"


def test_prompt_placeholder_fully_replaced():
    prompt = get_article_card_module_prompt("key_findings", _SAMPLE_TEXT)
    assert "{article_reading_text}" not in prompt


# ---------------------------------------------------------------------------
# 5. Each prompt includes "Return JSON only"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", ALL_MODULES)
def test_prompt_includes_return_json_only(module):
    prompt = get_article_card_module_prompt(module, _SAMPLE_TEXT)
    assert "Return JSON only" in prompt, f"[{module}] prompt missing 'Return JSON only'"


# ---------------------------------------------------------------------------
# 6. Each prompt includes evidence instruction using parent_id
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", ALL_MODULES)
def test_prompt_includes_parent_id_instruction(module):
    prompt = get_article_card_module_prompt(module, _SAMPLE_TEXT)
    assert "parent_id" in prompt, f"[{module}] prompt missing parent_id instruction"


@pytest.mark.parametrize("module", ALL_MODULES)
def test_prompt_instructs_verbatim_parent_ids(module):
    prompt = get_article_card_module_prompt(module, _SAMPLE_TEXT)
    assert "verbatim" in prompt.lower(), (
        f"[{module}] prompt should instruct to use verbatim parent_ids"
    )


# ---------------------------------------------------------------------------
# 7. Each prompt includes no-hypothesis and no-critique instructions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", ALL_MODULES)
def test_prompt_includes_no_hypothesis_instruction(module):
    prompt = get_article_card_module_prompt(module, _SAMPLE_TEXT).lower()
    assert "hypothes" in prompt, (
        f"[{module}] prompt missing no-hypothesis instruction"
    )


@pytest.mark.parametrize("module", ALL_MODULES)
def test_prompt_includes_no_critique_instruction(module):
    prompt = get_article_card_module_prompt(module, _SAMPLE_TEXT).lower()
    assert "critique" in prompt, (
        f"[{module}] prompt missing no-critique instruction"
    )


@pytest.mark.parametrize("module", ALL_MODULES)
def test_prompt_instructs_do_not_invent(module):
    prompt = get_article_card_module_prompt(module, _SAMPLE_TEXT).lower()
    assert "invent" in prompt or "fabricat" in prompt, (
        f"[{module}] prompt should instruct not to invent or fabricate information"
    )


# ---------------------------------------------------------------------------
# 8. Each prompt includes the expected module-specific JSON skeleton fields
# ---------------------------------------------------------------------------

_MODULE_EXPECTED_FIELDS: dict[str, list[str]] = {
    "research_question": [
        "disease_context", "biological_context", "main_question", "stated_gap", "study_type",
    ],
    "study_design": [
        "design_features", "comparison_groups", "discovery_validation_structure",
        "perturbation_or_intervention", "group_a", "group_b",
    ],
    "samples_and_cohorts": [
        "sample_or_cohort_name", "species", "disease_or_condition", "sample_material",
        "sample_count", "paired_design", "treatment_context", "sorting_or_enrichment",
        "used_for",
    ],
    "data_source_and_provenance": [
        "dataset_name", "source_type", "accession_id", "external_database_or_cohort",
        "generated_in_this_study",
    ],
    "omics_and_experimental_methods": [
        "method", "data_type", "modality_category", "generated_in_this_study", "purpose",
    ],
    "computational_analysis_pipeline": [
        "analysis_step", "software_or_method", "parameters_or_thresholds",
        "input_data", "output",
    ],
    "key_findings": [
        "finding", "associated_data", "associated_entities", "result_type",
    ],
    "mechanism_model": [
        "mechanism_chain", "upstream_factor", "mediators", "downstream_effect",
        "involved_cell_types", "involved_pathways_or_molecules",
        "therapeutic_or_clinical_implication",
    ],
    "author_reported_limitations_and_future_directions": [
        "limitation", "caveat", "future_direction", "text",
    ],
}


@pytest.mark.parametrize("module", ALL_MODULES)
def test_prompt_includes_skeleton_fields(module):
    prompt = get_article_card_module_prompt(module, _SAMPLE_TEXT)
    expected = _MODULE_EXPECTED_FIELDS[module]
    missing = [f for f in expected if f not in prompt]
    assert not missing, (
        f"[{module}] prompt is missing skeleton fields: {missing}"
    )


def test_data_source_prompt_includes_source_type_enum_values():
    prompt = get_article_card_module_prompt("data_source_and_provenance", _SAMPLE_TEXT)
    for val in ["generated_in_this_study", "external_public_data", "external_published_cohort",
                "external_database", "unclear"]:
        assert val in prompt, f"data_source prompt missing source_type enum value: {val}"


def test_limitation_prompt_includes_type_enum_values():
    prompt = get_article_card_module_prompt(
        "author_reported_limitations_and_future_directions", _SAMPLE_TEXT
    )
    for val in ["limitation", "caveat", "future_direction"]:
        assert val in prompt, f"limitations prompt missing type enum value: {val}"


def test_mechanism_prompt_includes_claims_key():
    prompt = get_article_card_module_prompt("mechanism_model", _SAMPLE_TEXT)
    assert '"claims"' in prompt


def test_study_design_prompt_includes_comparison_group_fields():
    prompt = get_article_card_module_prompt("study_design", _SAMPLE_TEXT)
    assert "comparison_purpose" in prompt


# ---------------------------------------------------------------------------
# 9. Prompt module names match schema module names
# ---------------------------------------------------------------------------

def test_module_names_match_schema():
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    schema_modules = set(schema["properties"]["modules"]["properties"].keys())
    prompt_modules = set(list_article_card_modules())
    assert prompt_modules == schema_modules, (
        f"Mismatch between prompt modules and schema modules.\n"
        f"In prompts only: {prompt_modules - schema_modules}\n"
        f"In schema only: {schema_modules - prompt_modules}"
    )


def test_schema_required_modules_all_covered():
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    required = set(schema["properties"]["modules"]["required"])
    prompt_modules = set(list_article_card_modules())
    missing = required - prompt_modules
    assert not missing, f"Schema required modules not in prompt library: {missing}"


# ---------------------------------------------------------------------------
# Additional: prompt structure sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", ALL_MODULES)
def test_prompt_includes_module_label(module):
    prompt = get_article_card_module_prompt(module, _SAMPLE_TEXT)
    assert f"[{module}]" in prompt, f"[{module}] label not found in prompt"


@pytest.mark.parametrize("module", ALL_MODULES)
def test_prompt_includes_summary_field(module):
    prompt = get_article_card_module_prompt(module, _SAMPLE_TEXT)
    assert '"summary"' in prompt, f"[{module}] prompt missing summary field in skeleton"


@pytest.mark.parametrize("module", ALL_MODULES)
def test_prompt_includes_evidence_field(module):
    prompt = get_article_card_module_prompt(module, _SAMPLE_TEXT)
    assert '"evidence"' in prompt, f"[{module}] prompt missing evidence field in skeleton"


@pytest.mark.parametrize("module", ALL_MODULES)
def test_prompt_ends_with_article_reading_text(module):
    prompt = get_article_card_module_prompt(module, _SAMPLE_TEXT)
    assert prompt.strip().endswith(_SAMPLE_TEXT.strip()), (
        f"[{module}] prompt should end with the article reading text"
    )
