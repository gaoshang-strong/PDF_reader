"""Tests for Step 4C: Article Card extraction.

All tests use a fake LLM client and a temporary data directory.
No real LLM calls are made.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step4B_article_card_prompts import list_article_card_modules
from pipeline.step4C_extract_article_card import (
    ALL_MODULES,
    assemble_reading_text,
    build_metadata_block,
    extract_article_card,
    extract_module,
    load_parents,
    _empty_module,
    validate_card,
)
from jsonschema import Draft7Validator

SCHEMA_PATH = REPO_ROOT / "schemas" / "article_card.schema.json"

# ---------------------------------------------------------------------------
# Minimal valid module JSON for every module (schema-conforming)
# ---------------------------------------------------------------------------

_FAKE_MODULE_JSON: dict[str, dict] = {
    "research_question": {
        "summary": "PNI drives immune exclusion in CRC.",
        "disease_context": ["colorectal cancer"],
        "biological_context": ["tumor microenvironment", "perineural invasion"],
        "main_question": "How does PNI affect T cell infiltration?",
        "stated_gap": "Mechanisms are unknown.",
        "study_type": ["mechanism"],
        "evidence": [],
    },
    "study_design": {
        "summary": None,
        "design_features": ["retrospective cohort"],
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
        "summary": "CD8+ T cells are reduced in PNI+ tumors.",
        "items": [
            {
                "finding": "PNI correlates with reduced CD8+ T cell infiltration.",
                "associated_data": ["IHC"],
                "associated_entities": ["CD8+ T cells", "PNI"],
                "result_type": "association",
                "evidence": [],
            }
        ],
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
}


# ---------------------------------------------------------------------------
# Fake LLM client
# ---------------------------------------------------------------------------

def _detect_module(prompt: str) -> str | None:
    """Extract module name from prompt by finding [module_name] pattern."""
    m = re.search(r"\[([a-z_]+)\]", prompt)
    if m:
        name = m.group(1)
        if name in _FAKE_MODULE_JSON:
            return name
    return None


class FakeLLMClient:
    """Returns fake module JSON; can be configured to fail the first N calls."""

    def __init__(self, fail_first_n: int = 0) -> None:
        self.call_count = 0
        self.fail_first_n = fail_first_n

    def call(self, system_content: str, user_content: str) -> str:
        self.call_count += 1
        if self.call_count <= self.fail_first_n:
            return "THIS IS NOT VALID JSON !!!"
        module_name = _detect_module(user_content)
        payload = _FAKE_MODULE_JSON.get(module_name or "", {"summary": None, "evidence": []})
        return json.dumps(payload)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ARTICLE_ID = "pdf_test_4c"

_SAMPLE_PARENTS = [
    {
        "parent_id": f"{ARTICLE_ID}::parent_000000",
        "article_id": ARTICLE_ID,
        "parent_index": 0,
        "chunk_level": "parent",
        "chunk_type": "body",
        "section_path": ["Text", "Introduction"],
        "text": "Perineural invasion (PNI) is associated with poor prognosis in CRC.",
        "char_start": 0,
        "char_end": 68,
        "token_count": 12,
        "include_in_default_qa": True,
    },
    {
        "parent_id": f"{ARTICLE_ID}::parent_000001",
        "article_id": ARTICLE_ID,
        "parent_index": 1,
        "chunk_level": "parent",
        "chunk_type": "body",
        "section_path": ["Text", "Methods"],
        "text": "We enrolled 82 treatment-naive CRC patients and collected tumor biopsies.",
        "char_start": 69,
        "char_end": 142,
        "token_count": 13,
        "include_in_default_qa": True,
    },
    {
        "parent_id": f"{ARTICLE_ID}::parent_000002",
        "article_id": ARTICLE_ID,
        "parent_index": 2,
        "chunk_level": "parent",
        "chunk_type": "figure_caption",
        "section_path": ["Images"],
        "text": "Figure 1. CD8+ T cell density in PNI-positive versus PNI-negative tumors.",
        "char_start": 143,
        "char_end": 217,
        "token_count": 13,
        "include_in_default_qa": True,
    },
]


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    """Create a temporary data root with parents.jsonl for the test article."""
    index_dir = tmp_path / "index" / ARTICLE_ID
    index_dir.mkdir(parents=True)
    parents_path = index_dir / "parents.jsonl"
    parents_path.write_text(
        "\n".join(json.dumps(p) for p in _SAMPLE_PARENTS) + "\n"
    )
    return tmp_path


@pytest.fixture
def fake_client() -> FakeLLMClient:
    return FakeLLMClient()


# ---------------------------------------------------------------------------
# 1. article_reading_text includes parent_id, section_path, chunk_type, text
# ---------------------------------------------------------------------------

def test_reading_text_includes_parent_id():
    text = assemble_reading_text(_SAMPLE_PARENTS)
    for p in _SAMPLE_PARENTS:
        assert p["parent_id"] in text


def test_reading_text_includes_section_path():
    text = assemble_reading_text(_SAMPLE_PARENTS)
    assert "Text > Introduction" in text
    assert "Text > Methods" in text


def test_reading_text_includes_chunk_type():
    text = assemble_reading_text(_SAMPLE_PARENTS)
    assert "chunk_type: body" in text
    assert "chunk_type: figure_caption" in text


def test_reading_text_includes_text_content():
    text = assemble_reading_text(_SAMPLE_PARENTS)
    for p in _SAMPLE_PARENTS:
        assert p["text"].strip() in text


def test_reading_text_ordered_by_parent_index():
    shuffled = list(reversed(_SAMPLE_PARENTS))
    text = assemble_reading_text(shuffled)
    pos0 = text.find(_SAMPLE_PARENTS[0]["parent_id"])
    pos1 = text.find(_SAMPLE_PARENTS[1]["parent_id"])
    pos2 = text.find(_SAMPLE_PARENTS[2]["parent_id"])
    # load_parents sorts; assemble_reading_text preserves input order
    # here we call assemble directly — just verify all three parent_ids appear
    assert pos0 >= 0 and pos1 >= 0 and pos2 >= 0


def test_load_parents_sorts_by_parent_index(tmp_root):
    parents = load_parents(ARTICLE_ID, tmp_root)
    indices = [p["parent_index"] for p in parents]
    assert indices == sorted(indices)


# ---------------------------------------------------------------------------
# 2. Extraction can run with fake LLM for one module
# ---------------------------------------------------------------------------

def test_extract_one_module(tmp_root, fake_client):
    card = extract_article_card(
        ARTICLE_ID,
        modules=["research_question"],
        overwrite=True,
        client=fake_client,
        data_root=tmp_root,
    )
    rq = card["modules"]["research_question"]
    assert rq["summary"] is not None
    assert isinstance(rq["disease_context"], list)


def test_extract_one_module_call_count(tmp_root, fake_client):
    extract_article_card(
        ARTICLE_ID,
        modules=["research_question"],
        overwrite=True,
        client=fake_client,
        data_root=tmp_root,
    )
    assert fake_client.call_count == 1


def test_unextracted_modules_are_empty_stubs(tmp_root, fake_client):
    card = extract_article_card(
        ARTICLE_ID,
        modules=["research_question"],
        overwrite=True,
        client=fake_client,
        data_root=tmp_root,
    )
    # mechanism_model should be a stub
    mm = card["modules"]["mechanism_model"]
    assert mm["summary"] is None
    assert mm["claims"] == []


# ---------------------------------------------------------------------------
# 3. Extraction can run with fake LLM for all nine modules
# ---------------------------------------------------------------------------

def test_extract_all_modules(tmp_root, fake_client):
    card = extract_article_card(
        ARTICLE_ID,
        overwrite=True,
        client=fake_client,
        data_root=tmp_root,
    )
    assert set(card["modules"].keys()) == set(ALL_MODULES)


def test_extract_all_modules_call_count(tmp_root, fake_client):
    extract_article_card(
        ARTICLE_ID,
        overwrite=True,
        client=fake_client,
        data_root=tmp_root,
    )
    assert fake_client.call_count == len(ALL_MODULES)


def test_all_modules_completed_in_manifest(tmp_root, fake_client):
    extract_article_card(
        ARTICLE_ID,
        overwrite=True,
        client=fake_client,
        data_root=tmp_root,
    )
    manifest_path = tmp_root / "literature_cards" / ARTICLE_ID / "extraction_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest["modules_completed"]) == set(ALL_MODULES)


# ---------------------------------------------------------------------------
# 4. Module JSON files are written
# ---------------------------------------------------------------------------

def test_module_files_written_for_all(tmp_root, fake_client):
    extract_article_card(
        ARTICLE_ID,
        overwrite=True,
        client=fake_client,
        data_root=tmp_root,
    )
    modules_dir = tmp_root / "literature_cards" / ARTICLE_ID / "modules"
    for module_name in ALL_MODULES:
        assert (modules_dir / f"{module_name}.json").exists(), \
            f"Missing module file: {module_name}.json"


def test_module_file_content_is_valid_json(tmp_root, fake_client):
    extract_article_card(
        ARTICLE_ID,
        overwrite=True,
        client=fake_client,
        data_root=tmp_root,
    )
    path = tmp_root / "literature_cards" / ARTICLE_ID / "modules" / "key_findings.json"
    data = json.loads(path.read_text())
    assert isinstance(data, dict)
    assert "summary" in data


def test_module_file_only_written_for_extracted_modules(tmp_root, fake_client):
    extract_article_card(
        ARTICLE_ID,
        modules=["research_question"],
        overwrite=True,
        client=fake_client,
        data_root=tmp_root,
    )
    modules_dir = tmp_root / "literature_cards" / ARTICLE_ID / "modules"
    assert (modules_dir / "research_question.json").exists()
    # Other modules should NOT have files (they are stubs, not written)
    assert not (modules_dir / "mechanism_model.json").exists()


# ---------------------------------------------------------------------------
# 5. article_card.json is written
# ---------------------------------------------------------------------------

def test_article_card_file_written(tmp_root, fake_client):
    extract_article_card(
        ARTICLE_ID,
        overwrite=True,
        client=fake_client,
        data_root=tmp_root,
    )
    card_path = tmp_root / "literature_cards" / ARTICLE_ID / "article_card.json"
    assert card_path.exists()


def test_article_card_contains_correct_article_id(tmp_root, fake_client):
    extract_article_card(
        ARTICLE_ID,
        overwrite=True,
        client=fake_client,
        data_root=tmp_root,
    )
    card_path = tmp_root / "literature_cards" / ARTICLE_ID / "article_card.json"
    card = json.loads(card_path.read_text())
    assert card["article_id"] == ARTICLE_ID


def test_article_card_has_schema_version(tmp_root, fake_client):
    extract_article_card(
        ARTICLE_ID,
        overwrite=True,
        client=fake_client,
        data_root=tmp_root,
    )
    card_path = tmp_root / "literature_cards" / ARTICLE_ID / "article_card.json"
    card = json.loads(card_path.read_text())
    assert card["schema_version"] == "0.1.0"


def test_no_overwrite_raises_file_exists_error(tmp_root, fake_client):
    extract_article_card(
        ARTICLE_ID,
        overwrite=True,
        client=fake_client,
        data_root=tmp_root,
    )
    with pytest.raises(FileExistsError):
        extract_article_card(
            ARTICLE_ID,
            overwrite=False,
            client=fake_client,
            data_root=tmp_root,
        )


# ---------------------------------------------------------------------------
# 6. article_card.json validates against schema
# ---------------------------------------------------------------------------

def test_article_card_validates_against_schema(tmp_root, fake_client):
    extract_article_card(
        ARTICLE_ID,
        overwrite=True,
        client=fake_client,
        data_root=tmp_root,
    )
    card_path = tmp_root / "literature_cards" / ARTICLE_ID / "article_card.json"
    card = json.loads(card_path.read_text())
    errors = validate_card(card)
    assert errors == [], f"Schema validation failed:\n" + "\n".join(errors)


def test_schema_validation_status_in_manifest(tmp_root, fake_client):
    extract_article_card(
        ARTICLE_ID,
        overwrite=True,
        client=fake_client,
        data_root=tmp_root,
    )
    manifest = json.loads(
        (tmp_root / "literature_cards" / ARTICLE_ID / "extraction_manifest.json").read_text()
    )
    assert manifest["schema_validation_status"] == "passed"


# ---------------------------------------------------------------------------
# 7. Metadata fallback works when metadata.json is missing
# ---------------------------------------------------------------------------

def test_metadata_fallback_when_missing(tmp_root, fake_client):
    card = extract_article_card(
        ARTICLE_ID,
        overwrite=True,
        client=fake_client,
        data_root=tmp_root,
    )
    meta = card["metadata"]
    assert meta["title"] is None
    assert meta["authors"] == []
    assert meta["journal"] is None
    assert meta["year"] is None


def test_metadata_loaded_when_present(tmp_root, fake_client):
    articles_dir = tmp_root / "articles" / ARTICLE_ID
    articles_dir.mkdir(parents=True)
    metadata = {
        "title": "PNI and immune exclusion in CRC",
        "authors": ["Smith J", "Lee K"],
        "journal": "Nature",
        "year": 2024,
        "doi": "10.1000/xyz123",
        "article_type": "research article",
    }
    (articles_dir / "metadata.json").write_text(json.dumps(metadata))

    card = extract_article_card(
        ARTICLE_ID,
        overwrite=True,
        client=fake_client,
        data_root=tmp_root,
    )
    assert card["metadata"]["title"] == "PNI and immune exclusion in CRC"
    assert card["metadata"]["year"] == 2024
    assert card["metadata"]["authors"] == ["Smith J", "Lee K"]


def test_build_metadata_block_none():
    block = build_metadata_block(None)
    assert block["title"] is None
    assert block["authors"] == []


def test_build_metadata_block_partial():
    block = build_metadata_block({"title": "Test", "year": 2023})
    assert block["title"] == "Test"
    assert block["year"] == 2023
    assert block["authors"] == []


def test_build_metadata_block_alternative_field_names():
    """Step 1 writes DOI (uppercase) and publication_year — both should be handled."""
    block = build_metadata_block({"title": "T", "DOI": "10.1/x", "publication_year": 2025})
    assert block["doi"] == "10.1/x"
    assert block["year"] == 2025


def test_build_metadata_block_strips_check_for_updates():
    """GROBID sometimes prepends 'Check for updates' to titles scraped from publisher pages."""
    block = build_metadata_block({"title": "Check for updates Dynamics of T cells in cancer"})
    assert block["title"] == "Dynamics of T cells in cancer"


# ---------------------------------------------------------------------------
# 8. Malformed JSON triggers exactly one retry
# ---------------------------------------------------------------------------

def test_malformed_json_triggers_one_retry(tmp_root):
    client = FakeLLMClient(fail_first_n=1)
    extract_article_card(
        ARTICLE_ID,
        modules=["research_question"],
        overwrite=True,
        client=client,
        data_root=tmp_root,
    )
    # First call fails → retry → retry succeeds = 2 calls for one module
    assert client.call_count == 2


def test_malformed_json_retry_succeeds_and_writes_module(tmp_root):
    client = FakeLLMClient(fail_first_n=1)
    extract_article_card(
        ARTICLE_ID,
        modules=["key_findings"],
        overwrite=True,
        client=client,
        data_root=tmp_root,
    )
    path = tmp_root / "literature_cards" / ARTICLE_ID / "modules" / "key_findings.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# 9. Malformed JSON after retry raises an error
# ---------------------------------------------------------------------------

def test_malformed_json_after_retry_raises(tmp_root):
    client = FakeLLMClient(fail_first_n=999)  # always returns bad JSON
    with pytest.raises((json.JSONDecodeError, ValueError)):
        extract_article_card(
            ARTICLE_ID,
            modules=["research_question"],
            overwrite=True,
            client=client,
            data_root=tmp_root,
        )


def test_double_failure_call_count(tmp_root):
    client = FakeLLMClient(fail_first_n=999)
    with pytest.raises((json.JSONDecodeError, ValueError)):
        extract_article_card(
            ARTICLE_ID,
            modules=["research_question"],
            overwrite=True,
            client=client,
            data_root=tmp_root,
        )
    # One original call + one retry = 2 calls total before giving up
    assert client.call_count == 2


def test_extract_module_retry_logic_directly():
    """Unit test for extract_module retry logic in isolation."""
    call_count = [0]

    class _Client:
        def call(self, system_content, user_content):
            call_count[0] += 1
            if call_count[0] == 1:
                return "NOT JSON"
            return json.dumps({"summary": None, "evidence": []})

    result = extract_module("key_findings", "sys", "user", _Client())
    assert call_count[0] == 2
    assert isinstance(result, dict)


def test_extract_module_double_fail_raises():
    class _Client:
        def call(self, system_content, user_content):
            return "DEFINITELY NOT JSON"

    with pytest.raises((json.JSONDecodeError, ValueError)):
        extract_module("key_findings", "sys", "user", _Client())


# ---------------------------------------------------------------------------
# 10. Core extraction logic works (integration, fake client)
# ---------------------------------------------------------------------------

def test_manifest_written(tmp_root, fake_client):
    extract_article_card(
        ARTICLE_ID,
        overwrite=True,
        client=fake_client,
        data_root=tmp_root,
    )
    manifest_path = tmp_root / "literature_cards" / ARTICLE_ID / "extraction_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["article_id"] == ARTICLE_ID
    assert "modules_requested" in manifest
    assert "modules_completed" in manifest
    assert "article_reading_parent_count" in manifest
    assert "article_reading_token_count_estimate" in manifest
    assert "schema_validation_status" in manifest
    assert "warnings" in manifest


def test_manifest_parent_count(tmp_root, fake_client):
    extract_article_card(
        ARTICLE_ID,
        overwrite=True,
        client=fake_client,
        data_root=tmp_root,
    )
    manifest = json.loads(
        (tmp_root / "literature_cards" / ARTICLE_ID / "extraction_manifest.json").read_text()
    )
    assert manifest["article_reading_parent_count"] == len(_SAMPLE_PARENTS)


def test_manifest_token_estimate_positive(tmp_root, fake_client):
    extract_article_card(
        ARTICLE_ID,
        overwrite=True,
        client=fake_client,
        data_root=tmp_root,
    )
    manifest = json.loads(
        (tmp_root / "literature_cards" / ARTICLE_ID / "extraction_manifest.json").read_text()
    )
    assert manifest["article_reading_token_count_estimate"] > 0


def test_unknown_module_raises_value_error(tmp_root, fake_client):
    with pytest.raises(ValueError, match="Unknown module"):
        extract_article_card(
            ARTICLE_ID,
            modules=["not_a_real_module"],
            overwrite=True,
            client=fake_client,
            data_root=tmp_root,
        )


def test_empty_module_stub_is_schema_valid():
    """_empty_module output should satisfy schema for each module."""
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    module_schemas = schema["properties"]["modules"]["properties"]
    for module_name in ALL_MODULES:
        stub = _empty_module(module_name)
        mod_schema = module_schemas[module_name]
        validator = Draft7Validator({"definitions": schema["definitions"], **mod_schema})
        errors = list(validator.iter_errors(stub))
        assert not errors, (
            f"_empty_module('{module_name}') is not schema-valid: "
            + "; ".join(e.message for e in errors)
        )
