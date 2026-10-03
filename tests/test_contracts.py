"""Tests for contract schemas: executability, round-trip fidelity, and tamper rejection.

The schemas in `contracts/schemas/` were decorative for a long time — referenced by no
code, shipped by no packaging config. These tests keep them executable: every Model-owned
artifact must survive a write/re-read round trip through *both* its JSON Schema and its
Pydantic model, in the exact form the production writers emit.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List

import pytest
from factories import DEFAULT_CONFIG, make_evaluation_report, make_gate_policy
from pydantic import ValidationError as PydanticValidationError

from vision_model_factory.contracts.gate_policy import build_gate
from vision_model_factory.contracts.hashing import compute_sha256_file
from vision_model_factory.contracts.models import (
    ClassMapItem,
    DatasetRef,
    DecoderSpec,
    EvaluationReport,
    ExperimentSpec,
    FileRef,
    InputSpec,
    ModelManifest,
    OutputSpec,
    ParamsSpec,
    PostprocessSpec,
    PreprocessSpec,
    RunResult,
    TargetSpec,
    TrainerSpec,
    model_json,
)
from vision_model_factory.contracts.schema_validation import (
    SCHEMAS,
    SchemaValidationError,
    load_schema,
    schema_errors,
    schema_path,
    validate_instance,
)
from vision_model_factory.contracts.validators import ValidationError, validate_dataset_package
from vision_model_factory.evaluation.evaluator import parity_not_measured

MODEL_OWNED_SCHEMAS = ("experiment_spec", "run_result", "model_manifest", "evaluation_report")


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        schema_version="1.0.0",
        experiment_id="exp-schema-001",
        dataset=DatasetRef(dataset_id="ds-001", manifest_sha256="a" * 64),
        trainer=TrainerSpec(
            adapter_id="yolo_detection_v1", model_id="mock_yolo_v1", checkpoint_sha256="0" * 64
        ),
        params=ParamsSpec(imgsz=640, batch=8, epochs=1, seed=42, lr0=0.01, mosaic=1.0),
        reason="Schema round-trip probe",
    )


def _run_result() -> RunResult:
    return RunResult(
        schema_version="1.0.0",
        run_id="run-schema-001",
        experiment_id="exp-schema-001",
        status="succeeded",
        dataset=_spec().dataset,
        effective_config_sha256="c" * 64,
        environment={"python_version": "3.13.0", "torch_version": "2.14.0"},
        duration_seconds=1.5,
        artifacts={"checkpoint": FileRef(path="attempt/train/best.pt", sha256="b" * 64)},
        val_metrics={"mAP50": 0.88},
    )


def _manifest() -> ModelManifest:
    return ModelManifest(
        schema_version="1.0.0",
        package_id="model-schema-001",
        created_at="2026-10-03T00:00:00Z",
        task_type="object_detection",
        dataset=_spec().dataset,
        run_id="run-schema-001",
        task=FileRef(path="task.json", sha256="a" * 64),
        model=FileRef(path="model.onnx", sha256="b" * 64),
        evaluation=FileRef(path="evaluation.json", sha256="c" * 64),
        target=TargetSpec(),
        input=InputSpec(name="images", dtype="float32", shape=[1, 3, 640, 640]),
        preprocess=PreprocessSpec(),
        output=OutputSpec(name="output0", dtype="float32", shape=[1, 8, 8400]),
        decoder=DecoderSpec(),
        class_map=[ClassMapItem(index=0, class_id="bottle"), ClassMapItem(index=1, class_id="can")],
        postprocess=PostprocessSpec(),
    )


def test_every_referenced_schema_file_exists_and_is_draft_2020_12():
    for contract in SCHEMAS:
        schema = load_schema(contract)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert "properties" in schema or "allOf" in schema or "$defs" in schema


def test_model_owned_artifacts_round_trip_through_their_own_schema():
    """Write with the production serializer, re-read through schema + model.

    The two writers differ: `model_dump_json()` emits `"extensions": null`, which the
    schema declares as an object and therefore rejects. Artifacts must be written with
    `model_json`, so that is the form under test.
    """
    artifacts = {
        "experiment_spec": _spec(),
        "run_result": _run_result(),
        "model_manifest": _manifest(),
        "evaluation_report": make_evaluation_report(),
    }
    for contract, obj in artifacts.items():
        payload = json.loads(model_json(obj))
        errors = schema_errors(payload, contract)
        assert errors == [], f"{contract} fails its own JSON Schema: {errors}"

    # And the failure mode that this test exists to prevent, stated explicitly so it
    # cannot silently return: the naive writer is *not* schema-valid.
    naive = json.loads(_spec().model_dump_json())
    assert schema_errors(naive, "experiment_spec") != [], (
        "model_dump_json() unexpectedly satisfies the schema; if optional fields are now "
        "omitted by default, delete model_json() instead of keeping two mechanisms"
    )


def test_evaluation_report_parses_back_into_its_model():
    report = make_evaluation_report()
    from vision_model_factory.contracts.models import EvaluationReport

    restored = EvaluationReport.model_validate(json.loads(model_json(report)))
    assert restored == report


def test_schema_rejects_structurally_invalid_payloads():
    with pytest.raises(SchemaValidationError, match="JSON Schema validation failed"):
        validate_instance({"schema_version": "1.0", "experiment_id": ""}, "experiment_spec")

    with pytest.raises(SchemaValidationError):
        validate_instance("not even an object", "model_manifest")

    with pytest.raises(SchemaValidationError):
        validate_instance({}, "run_result")

    with pytest.raises(KeyError):
        validate_instance({}, "no_such_contract")


def test_schema_rejects_a_vacuous_parity_pass_at_the_wire_level():
    """The self-test requirement is enforced in the schema, not only in Python.

    Without this, a non-Python producer could publish `status: passed` with zero
    detections and the Rust runtime would accept it.
    """
    report = json.loads(make_evaluation_report().model_dump_json(exclude_none=True))

    # 1. Passed with a genuine self-test: valid.
    assert schema_errors(report, "evaluation_report") == []

    # 2. Passed with the self-test removed: invalid.
    missing = json.loads(json.dumps(report))
    del missing["export_parity"]["self_test"]
    assert schema_errors(missing, "evaluation_report") != []

    # 3. Passed with empty detections: invalid.
    empty = json.loads(json.dumps(report))
    empty["export_parity"]["detections"]["ref_count"] = 0
    empty["export_parity"]["detections"]["ort_count"] = 0
    assert schema_errors(empty, "evaluation_report") != []

    # 4. Passed while the raw tensor comparison failed: invalid.
    tensor_failed = json.loads(json.dumps(report))
    tensor_failed["export_parity"]["raw_tensor"]["passed"] = False
    assert schema_errors(tensor_failed, "evaluation_report") != []

    # 5. "Parity was never measured" stays representable: failed, with no self-test.
    #    Built through the model so the surrounding gate stays coherent, then checked
    #    against the schema end to end.
    from vision_model_factory.contracts.models import EvaluationReport

    not_run = make_evaluation_report(parity=parity_not_measured(DEFAULT_CONFIG))
    assert isinstance(not_run, EvaluationReport)
    assert not_run.export_parity.status == "failed"
    assert not_run.export_parity.self_test is None
    payload = json.loads(model_json(not_run))
    assert schema_errors(payload, "evaluation_report") == []
    assert "self_test" not in payload["export_parity"], (
        "an unmeasured parity must not serialise a null self_test: the schema declares an "
        "object, and null is a third state no consumer can interpret"
    )
    # And such a report fails any policy that requires parity, at the gate level.
    strict = make_gate_policy(require_export_parity_passed=True)
    assert build_gate(strict, not_run.test, not_run.export_parity, []).status == "failed"


def test_gate_placeholder_digest_is_rejected_by_the_model():
    from vision_model_factory.contracts.models import GateSection

    with pytest.raises(PydanticValidationError, match="not a placeholder"):
        GateSection(
            policy_id="p", policy_sha256="0" * 64, status="passed",
            checks=[{"metric": "mAP50", "threshold": 0.5, "actual": 0.9, "passed": True}],
        )


def test_gate_status_must_agree_with_its_checks():
    from vision_model_factory.contracts.models import GateSection

    # A check whose own numbers contradict its `passed` flag is caught first.
    with pytest.raises(PydanticValidationError, match="implies passed=False"):
        GateSection(
            policy_id="p", policy_sha256="a" * 64, status="passed",
            checks=[{"metric": "mAP50", "threshold": 0.9, "actual": 0.1, "passed": True}],
        )

    # Every check passing while the overall status claims failure is caught too.
    with pytest.raises(PydanticValidationError, match="recorded checks imply"):
        GateSection(
            policy_id="p", policy_sha256="a" * 64, status="failed",
            checks=[{"metric": "mAP50", "threshold": 0.5, "actual": 0.9, "passed": True}],
        )

    # A check that claims to fail while its own numbers say it passed is refused.
    with pytest.raises(PydanticValidationError, match="implies passed=True"):
        GateSection(
            policy_id="p", policy_sha256="a" * 64, status="failed",
            checks=[{"metric": "mAP50", "threshold": 0.5, "actual": 0.9, "passed": False}],
        )


def test_policy_sha256_changes_when_a_threshold_changes():
    from factories import make_gate_policy

    from vision_model_factory.contracts.gate_policy import policy_sha256

    a = make_gate_policy(policy_id="p", min_map50=0.5)
    b = make_gate_policy(policy_id="p", min_map50=0.51)
    assert policy_sha256(a) != policy_sha256(b)
    # Same document, same digest regardless of construction order.
    assert policy_sha256(make_gate_policy(policy_id="p", min_map50=0.5)) == policy_sha256(a)


def test_policy_cannot_invert_its_mAP_ceilings():
    from factories import make_gate_policy

    with pytest.raises(PydanticValidationError, match="must not exceed"):
        make_gate_policy(min_map50=0.4, min_map50_95=0.8)


def test_data_v1_mirror_matches_contract_manifest():
    """Data-owned schemas are vendored by digest; a locally edited mirror must fail loudly."""
    mirror = schema_path("task_spec").parent
    manifest = json.loads((mirror / "contract-manifest.json").read_text(encoding="utf-8"))

    assert manifest["owner"], "mirror must name the owning repository"
    assert set(manifest["files"]) == {p.name for p in mirror.glob("*.schema.json")}, (
        "mirror directory and manifest disagree about which files are vendored"
    )
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((mirror / name).read_bytes()).hexdigest()
        assert actual == expected, f"vendored Data-owned schema {name} does not match its digest"


def test_no_data_owned_schema_is_duplicated_outside_the_mirror():
    """One wire contract, one source of truth. Shadows drift and the drift is invisible."""
    schemas_dir = schema_path("task_spec").parent.parent
    data_owned = {name for name, (path, _) in SCHEMAS.items() if path.startswith("data_v1/")}

    for name in data_owned:
        shadow = schemas_dir / f"{name}.schema.json"
        assert not shadow.exists(), (
            f"{shadow} duplicates the Data-owned {name} schema; delete it and use data_v1/"
        )


def test_dataset_package_rejects_a_dropped_bytes_and_a_bad_digest(synthetic_dataset_dir: Path):
    """The digest checks that make a package immutable must actually fire."""
    samples = synthetic_dataset_dir / "samples.jsonl"
    original = samples.read_bytes()
    samples.write_bytes(original.replace(b"complete_verified", b"partial", 1))

    with pytest.raises(ValidationError, match="Samples file SHA-256 mismatch"):
        validate_dataset_package(synthetic_dataset_dir)

    samples.write_bytes(original)
    manifest_path = synthetic_dataset_dir / "dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["annotations"]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValidationError, match="Annotations file SHA-256 mismatch"):
        validate_dataset_package(synthetic_dataset_dir)


def test_dataset_package_rejects_identical_image_bytes_across_splits(synthetic_dataset_dir: Path):
    """Train/test leakage by identical pixels is caught even with distinct sample ids."""
    import numpy as np
    from PIL import Image


    # Copy the train image bytes over the val image, keeping different paths/ids.
    src = synthetic_dataset_dir / "images" / "img1.jpg"
    dst = synthetic_dataset_dir / "images" / "img2.jpg"
    Image.fromarray(np.full((480, 640, 3), 99, dtype=np.uint8)).save(dst)
    with src.open("rb") as f:
        payload = f.read()
    dst.write_bytes(payload)

    samples = synthetic_dataset_dir / "samples.jsonl"
    records = [json.loads(line) for line in samples.read_text().splitlines() if line.strip()]
    for rec in records:
        if rec["sample_id"] == "s-002":
            rec["file"]["sha256"] = compute_sha256_file(dst)
    samples.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    manifest_path = synthetic_dataset_dir / "dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["samples"]["sha256"] = compute_sha256_file(samples)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValidationError, match="Identical image bytes cross splits"):
        validate_dataset_package(synthetic_dataset_dir)


def test_timestamps_are_utc_only():
    """Published artifacts carry RFC 3339 UTC; an offset stamp would sort wrongly."""
    from vision_model_factory.contracts.models import DatasetManifest

    base = {
        "schema_version": "1.0.0",
        "dataset_id": "ds-t",
        "task": {"path": "task.json", "sha256": "a" * 64},
        "samples": {"path": "samples.jsonl", "sha256": "b" * 64},
        "annotations": {"path": "annotations.jsonl", "sha256": "c" * 64},
        "quality": {"path": "quality.json", "sha256": "d" * 64},
    }
    with pytest.raises(PydanticValidationError):
        DatasetManifest(**base, created_at="2026-10-03T00:00:00+08:00")
    with pytest.raises(PydanticValidationError):
        DatasetManifest(**base, created_at="2026-10-03 00:00:00")
    DatasetManifest(**base, created_at="2026-10-03T00:00:00Z")

    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
    assert pattern.match("2026-10-03T00:00:00Z")
    assert not pattern.match("2026-10-03T00:00:00+08:00")


def test_package_paths_cannot_escape_their_directory(tmp_path: Path):
    from vision_model_factory.contracts.validators import validate_relative_path

    with pytest.raises(ValidationError, match="traverse parent"):
        validate_relative_path("../outside.json")
    with pytest.raises(ValidationError, match="must be relative"):
        validate_relative_path("/etc/passwd")

    # A path with no literal `..` that still resolves outside the package via a symlink.
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (tmp_path / "secret.json").write_text("{}", encoding="utf-8")
    (pkg / "link.json").symlink_to(tmp_path / "secret.json")

    with pytest.raises(ValidationError, match="escapes base"):
        validate_relative_path("link.json", base_dir=pkg)


def test_package_data_ships_the_schemas(tmp_path: Path):
    """A wheel without the schema files would validate nothing at import time."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert "contracts/schemas/*.json" in text
    assert "contracts/schemas/data_v1/*.json" in text
    assert "jsonschema" in text


def _model_field_sets(model_cls) -> Dict[str, set]:
    """Field-name sets of `model_cls` and every contract model reachable from it."""
    from pydantic import BaseModel

    found: Dict[str, set] = {}
    stack = [model_cls]
    seen = set()
    while stack:
        cls = stack.pop()
        if cls in seen or not (isinstance(cls, type) and issubclass(cls, BaseModel)):
            continue
        seen.add(cls)
        found[cls.__name__] = set(cls.model_fields)
        for hint in cls.model_fields.values():
            annotation = hint.annotation
            candidates = [annotation]
            for arg in getattr(annotation, "__args__", ()) or ():
                candidates.append(arg)
            for cand in candidates:
                if isinstance(cand, type) and issubclass(cand, BaseModel):
                    stack.append(cand)
    return found


def _schema_property_sets(schema: dict) -> List[set]:
    """Every distinct property-name set declared in a hand-written schema."""
    sets = []

    def walk(node):
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict):
                sets.append(set(props))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    return sets


@pytest.mark.parametrize(
    "model_cls,contract",
    [
        (EvaluationReport, "evaluation_report"),
        (ExperimentSpec, "experiment_spec"),
        (ModelManifest, "model_manifest"),
        (RunResult, "run_result"),
    ],
)
def test_schema_and_pydantic_model_agree_on_every_field_name(model_cls, contract):
    """The schema and the model describing the same artifact must not drift apart.

    Both are maintained by hand, and the gap between them is load-bearing: a field that
    exists only in the model is persisted without ever being validated, which is how a
    required parity self-test evidence and a relative-difference figure could be added to
    one side and silently missed on the other. Field *names* are checked here; nullability
    and numeric bounds are checked by the round-trip tests.
    """
    schema = load_schema(contract)
    declared = _schema_property_sets(schema)
    everything = set().union(*declared) if declared else set()
    missing = []
    for cls_name, fields in _model_field_sets(model_cls).items():
        # A schema subschema may omit an optional field, but then the field must be absent
        # from every property set - so require at least one superset that covers it.
        if not any(fields <= have for have in declared):
            absent = sorted(fields - everything)
            missing.append(f"{cls_name}: absent from schema entirely -> {absent or sorted(fields)}")
    assert not missing, f"{contract} schema does not describe these model fields: {missing}"
