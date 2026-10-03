"""JSON Schema Draft 2020-12 structural validation for contract artifacts.

The bundled schema files are the normative cross-language shape (copied as fixtures by
the Rust runtime and the data factory). Structural checking runs before the semantic
Pydantic models so that a structurally malformed artifact is rejected with schema-level
diagnostics, and so the schema files stay executable rather than decorative.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from jsonschema import Draft202012Validator, FormatChecker

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"

SCHEMAS: Dict[str, Tuple[str, str]] = {
    "task_spec": ("data_v1/task_spec.schema.json", "TaskSpec"),
    "dataset_manifest": ("data_v1/dataset_manifest.schema.json", "DatasetManifest"),
    "sample_record": ("data_v1/sample_record.schema.json", "SampleRecord"),
    "annotation_record": ("data_v1/annotation_record.schema.json", "AnnotationRecord"),
    "experiment_spec": ("experiment_spec.schema.json", "ExperimentSpec"),
    "run_result": ("run_result.schema.json", "RunResult"),
    "model_manifest": ("model_manifest.schema.json", "ModelManifest"),
    "evaluation_report": ("evaluation_report.schema.json", "EvaluationReport"),
}


class SchemaValidationError(ValueError):
    """Raised when an artifact does not satisfy its JSON Schema."""


def schema_path(contract: str) -> Path:
    if contract not in SCHEMAS:
        raise KeyError(f"Unknown contract '{contract}'; known: {sorted(SCHEMAS)}")
    return SCHEMA_DIR / SCHEMAS[contract][0]


def load_schema(contract: str) -> Dict[str, Any]:
    path = schema_path(contract)
    try:
        with path.open("r", encoding="utf-8") as f:
            schema = json.load(f)
    except FileNotFoundError:
        raise SchemaValidationError(f"Missing schema file for contract '{contract}': {path}")
    except json.JSONDecodeError as e:
        raise SchemaValidationError(f"Failed to parse schema file {path}: {e}")

    if schema.get("$schema") != SCHEMA_DRAFT:
        raise SchemaValidationError(
            f"Schema {path.name} declares $schema={schema.get('$schema')!r}, expected {SCHEMA_DRAFT!r}"
        )
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as e:  # noqa: BLE001 - re-raised as a contract-level error
        raise SchemaValidationError(f"Schema {path.name} is not a valid Draft 2020-12 schema: {e}")
    return schema


def schema_errors(payload: Any, contract: str) -> List[Tuple[str, str]]:
    """Return (json-pointer, message) pairs for every structural violation."""
    schema = load_schema(contract)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors: List[Tuple[str, str]] = []
    for err in sorted(validator.iter_errors(payload), key=lambda e: list(e.absolute_path)):
        pointer = "/".join(str(p) for p in err.absolute_path) or "<root>"
        errors.append((pointer, err.message))
    return errors


def validate_instance(payload: Any, contract: str) -> None:
    """Structurally validate an in-memory artifact against its contract schema."""
    errors = schema_errors(payload, contract)
    if errors:
        name = SCHEMAS[contract][1]
        details = "; ".join(f"{pointer}: {message}" for pointer, message in errors[:10])
        raise SchemaValidationError(f"JSON Schema validation failed for {name} ({len(errors)} error(s)): {details}")


def validate_json_file(path: Path, contract: str) -> Any:
    """Load a JSON artifact from disk and structurally validate it."""
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        raise SchemaValidationError(f"Missing {SCHEMAS[contract][1]} artifact: {path}")
    except json.JSONDecodeError as e:
        raise SchemaValidationError(f"Failed to parse {path}: {e}")
    validate_instance(payload, contract)
    return payload


def list_schema_files() -> List[Path]:
    return [SCHEMA_DIR / filename for filename, _ in SCHEMAS.values()]


__all__ = [
    "SCHEMA_DIR",
    "SCHEMA_DRAFT",
    "SCHEMAS",
    "SchemaValidationError",
    "load_schema",
    "schema_errors",
    "schema_path",
    "list_schema_files",
    "validate_instance",
    "validate_json_file",
]
