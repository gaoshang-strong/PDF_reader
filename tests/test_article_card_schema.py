"""Tests for the Article Literature Card JSON schema (Step 4A).

All test data is constructed inline — no fixture files are required.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

try:
    from jsonschema import Draft7Validator
except ImportError:
    pytest.skip("jsonschema not installed", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas" / "article_card.schema.json"
VALIDATE_SCRIPT = REPO_ROOT / "scripts" / "validate_article_card.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def errors_for(card: dict, schema: dict) -> list[str]:
    validator = Draft7Validator(schema)
    return [e.message for e in validator.iter_errors(card)]


def is_valid(card: dict, schema: dict) -> bool:
    return Draft7Validator(schema).is_valid(card)


# ---------------------------------------------------------------------------
# Minimal valid card (all nulls / empty lists — the absolute floor)
# ---------------------------------------------------------------------------

MINIMAL_VALID_CARD: dict = {
    "schema_version": "0.1.0",
    "article_id": "pdf_test000000000000",
    "metadata": {
        "title": None,
        "authors": [],
        "journal": None,
        "year": None,
        "doi": None,
        "article_type": None,
    },
    "modules": {
        "research_question": {
            "summary": None,
            "disease_context": [],
            "biological_context": [],
            "main_question": None,
            "stated_gap": None,
            "study_type": [],
            "evidence": [],
        },
        "study_design": {
            "summary": None,
            "design_features": [],
            "comparison_groups": [],
            "discovery_validation_structure": None,
            "perturbation_or_intervention": None,
            "evidence": [],
        },
        "samples_and_cohorts": {
            "summary": None,
            "items": [],
            "evidence": [],
        },
        "data_source_and_provenance": {
            "summary": None,
            "items": [],
            "evidence": [],
        },
        "omics_and_experimental_methods": {
            "summary": None,
            "items": [],
            "evidence": [],
        },
        "computational_analysis_pipeline": {
            "summary": None,
            "items": [],
            "evidence": [],
        },
        "key_findings": {
            "summary": None,
            "items": [],
            "evidence": [],
        },
        "mechanism_model": {
            "summary": None,
            "claims": [],
            "evidence": [],
        },
        "author_reported_limitations_and_future_directions": {
            "summary": None,
            "items": [],
            "evidence": [],
        },
    },
    "extraction_metadata": {
        "extraction_method": None,
        "created_at": None,
        "source_files": [],
        "module_status": {},
        "warnings": [],
    },
}


# ---------------------------------------------------------------------------
# 1. Minimal valid card passes validation
# ---------------------------------------------------------------------------

def test_minimal_valid_card_passes(schema):
    errs = errors_for(MINIMAL_VALID_CARD, schema)
    assert errs == [], f"Unexpected errors: {errs}"


def test_populated_card_passes(schema):
    card = copy.deepcopy(MINIMAL_VALID_CARD)
    card["metadata"]["title"] = "Perineural invasion drives immune exclusion"
    card["metadata"]["authors"] = ["Smith J", "Lee K"]
    card["metadata"]["year"] = 2024
    card["modules"]["key_findings"]["summary"] = "PNI correlates with T cell exclusion."
    card["modules"]["key_findings"]["items"] = [
        {
            "finding": "CD8+ T cells are reduced in PNI+ tumors.",
            "associated_data": ["scRNA-seq", "IHC"],
            "associated_entities": ["CD8+ T cells", "PNI"],
            "result_type": "association",
            "evidence": [
                {
                    "parent_id": "pdf_test000000000000::parent_000001",
                    "section_path": ["Text", "Results"],
                    "supporting_text": "CD8+ T cells were significantly reduced.",
                    "char_start": 1200,
                    "char_end": 1260,
                }
            ],
        }
    ]
    errs = errors_for(card, schema)
    assert errs == [], f"Populated card should be valid, got: {errs}"


# ---------------------------------------------------------------------------
# 2. Missing required top-level field fails
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", [
    "schema_version", "article_id", "metadata", "modules", "extraction_metadata",
])
def test_missing_top_level_field_fails(schema, field):
    card = copy.deepcopy(MINIMAL_VALID_CARD)
    del card[field]
    assert not is_valid(card, schema), f"Should fail when '{field}' is missing"


def test_extra_top_level_field_fails(schema):
    card = copy.deepcopy(MINIMAL_VALID_CARD)
    card["unexpected_field"] = "oops"
    assert not is_valid(card, schema)


# ---------------------------------------------------------------------------
# 3. Missing required module fails
# ---------------------------------------------------------------------------

REQUIRED_MODULES = [
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


@pytest.mark.parametrize("module", REQUIRED_MODULES)
def test_missing_required_module_fails(schema, module):
    card = copy.deepcopy(MINIMAL_VALID_CARD)
    del card["modules"][module]
    assert not is_valid(card, schema), f"Should fail when module '{module}' is missing"


def test_all_required_modules_present_passes(schema):
    card = copy.deepcopy(MINIMAL_VALID_CARD)
    for m in REQUIRED_MODULES:
        assert m in card["modules"], f"MINIMAL_VALID_CARD is missing module '{m}'"
    assert is_valid(card, schema)


# ---------------------------------------------------------------------------
# 4. Invalid evidence object fails
# ---------------------------------------------------------------------------

def test_evidence_missing_parent_id_fails(schema):
    card = copy.deepcopy(MINIMAL_VALID_CARD)
    card["modules"]["key_findings"]["evidence"] = [
        {"section_path": ["Results"], "supporting_text": "some text"}
    ]
    assert not is_valid(card, schema)


def test_evidence_missing_section_path_fails(schema):
    card = copy.deepcopy(MINIMAL_VALID_CARD)
    card["modules"]["key_findings"]["evidence"] = [
        {"parent_id": "pdf_test000000000000::parent_000001"}
    ]
    assert not is_valid(card, schema)


def test_evidence_extra_field_fails(schema):
    card = copy.deepcopy(MINIMAL_VALID_CARD)
    card["modules"]["key_findings"]["evidence"] = [
        {
            "parent_id": "pdf_test000000000000::parent_000001",
            "section_path": ["Results"],
            "unexpected_key": "bad",
        }
    ]
    assert not is_valid(card, schema)


def test_valid_evidence_object_passes(schema):
    card = copy.deepcopy(MINIMAL_VALID_CARD)
    card["modules"]["key_findings"]["evidence"] = [
        {
            "parent_id": "pdf_test000000000000::parent_000001",
            "section_path": ["Text", "Results"],
            "supporting_text": "CD8+ T cells were reduced in PNI+ tumors.",
            "char_start": 1200,
            "char_end": 1260,
        }
    ]
    assert is_valid(card, schema)


def test_evidence_null_optional_fields_passes(schema):
    card = copy.deepcopy(MINIMAL_VALID_CARD)
    card["modules"]["key_findings"]["evidence"] = [
        {
            "parent_id": "pdf_test000000000000::parent_000001",
            "section_path": [],
            "supporting_text": None,
            "char_start": None,
            "char_end": None,
        }
    ]
    assert is_valid(card, schema)


# ---------------------------------------------------------------------------
# 5. Invalid data source_type fails
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_type", ["unknown", "internal", "public", "", "GENERATED"])
def test_invalid_source_type_fails(schema, bad_type):
    card = copy.deepcopy(MINIMAL_VALID_CARD)
    card["modules"]["data_source_and_provenance"]["items"] = [
        {"source_type": bad_type}
    ]
    assert not is_valid(card, schema), f"source_type '{bad_type}' should be invalid"


@pytest.mark.parametrize("source_type", [
    "generated_in_this_study",
    "external_public_data",
    "external_published_cohort",
    "external_database",
    "unclear",
])
def test_valid_source_types_pass(schema, source_type):
    card = copy.deepcopy(MINIMAL_VALID_CARD)
    card["modules"]["data_source_and_provenance"]["items"] = [
        {"source_type": source_type}
    ]
    errs = errors_for(card, schema)
    assert errs == [], f"source_type '{source_type}' should be valid, got: {errs}"


def test_data_source_missing_source_type_fails(schema):
    card = copy.deepcopy(MINIMAL_VALID_CARD)
    card["modules"]["data_source_and_provenance"]["items"] = [
        {"dataset_name": "TCGA", "accession_id": "phs000178"}
        # missing required source_type
    ]
    assert not is_valid(card, schema)


# ---------------------------------------------------------------------------
# 6. Invalid limitation type fails
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_type", ["weakness", "note", "suggestion", "", "Limitation"])
def test_invalid_limitation_type_fails(schema, bad_type):
    card = copy.deepcopy(MINIMAL_VALID_CARD)
    card["modules"]["author_reported_limitations_and_future_directions"]["items"] = [
        {"type": bad_type, "text": "Some text.", "evidence": []}
    ]
    assert not is_valid(card, schema), f"limitation type '{bad_type}' should be invalid"


@pytest.mark.parametrize("lim_type", ["limitation", "caveat", "future_direction"])
def test_valid_limitation_types_pass(schema, lim_type):
    card = copy.deepcopy(MINIMAL_VALID_CARD)
    card["modules"]["author_reported_limitations_and_future_directions"]["items"] = [
        {"type": lim_type, "text": "Some text.", "evidence": []}
    ]
    errs = errors_for(card, schema)
    assert errs == [], f"limitation type '{lim_type}' should be valid, got: {errs}"


def test_limitation_missing_type_fails(schema):
    card = copy.deepcopy(MINIMAL_VALID_CARD)
    card["modules"]["author_reported_limitations_and_future_directions"]["items"] = [
        {"text": "Small sample size."}
    ]
    assert not is_valid(card, schema)


# ---------------------------------------------------------------------------
# 7. Validator CLI exits 0 on a valid JSON file
# ---------------------------------------------------------------------------

def test_cli_valid_file_exits_zero():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(MINIMAL_VALID_CARD, f)
        tmp_path = Path(f.name)
    try:
        result = subprocess.run(
            [sys.executable, str(VALIDATE_SCRIPT), str(tmp_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"CLI should exit 0 for valid file.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "OK" in result.stdout
    finally:
        tmp_path.unlink(missing_ok=True)


def test_cli_multiple_valid_files_exits_zero():
    cards = [copy.deepcopy(MINIMAL_VALID_CARD) for _ in range(3)]
    paths = []
    try:
        for card in cards:
            f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            json.dump(card, f)
            f.close()
            paths.append(Path(f.name))
        result = subprocess.run(
            [sys.executable, str(VALIDATE_SCRIPT)] + [str(p) for p in paths],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert result.stdout.count("OK") == 3
    finally:
        for p in paths:
            p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 8. Validator CLI exits non-zero on an invalid JSON file
# ---------------------------------------------------------------------------

def test_cli_invalid_file_exits_nonzero():
    invalid_card = {"article_id": "missing_other_required_fields"}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(invalid_card, f)
        tmp_path = Path(f.name)
    try:
        result = subprocess.run(
            [sys.executable, str(VALIDATE_SCRIPT), str(tmp_path)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0, "CLI should exit non-zero for invalid file"
        assert "FAIL" in result.stdout
    finally:
        tmp_path.unlink(missing_ok=True)


def test_cli_missing_file_exits_nonzero():
    result = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT), "/nonexistent/path/card.json"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "MISSING" in result.stdout


def test_cli_mixed_valid_invalid_exits_nonzero():
    valid_path = invalid_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(MINIMAL_VALID_CARD, f)
            valid_path = Path(f.name)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"bad": "card"}, f)
            invalid_path = Path(f.name)
        result = subprocess.run(
            [sys.executable, str(VALIDATE_SCRIPT), str(valid_path), str(invalid_path)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "OK" in result.stdout
        assert "FAIL" in result.stdout
    finally:
        if valid_path:
            valid_path.unlink(missing_ok=True)
        if invalid_path:
            invalid_path.unlink(missing_ok=True)
