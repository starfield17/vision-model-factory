"""Tests for atomic model package publication and the release gate."""

import json
from pathlib import Path
from typing import Optional

import pytest
from factories import (
    DEFAULT_CONFIG,
    DEFAULT_TASK_SPEC,
    make_evaluation_report,
    make_gate_policy,
    make_parity_section,
    write_dataset_package,
)

from vision_model_factory.contracts.gate_policy import policy_sha256
from vision_model_factory.contracts.hashing import compute_sha256_file
from vision_model_factory.contracts.models import (
    ClassMapItem,
    DatasetRef,
    EvaluationReport,
    PostprocessSpec,
    model_json,
)
from vision_model_factory.contracts.validators import (
    ValidationError,
    validate_dataset_package,
    validate_model_package,
)
from vision_model_factory.export.exporter import export_torch_model_to_onnx
from vision_model_factory.release.publisher import ReleasePublicationError, publish_model_package
from vision_model_factory.release.registry import ModelRegistry
from vision_model_factory.trainers.yolo import TinyYoloMockNet

CLASS_MAP = [ClassMapItem(index=0, class_id="bottle"), ClassMapItem(index=1, class_id="can")]
DATASET_REF = DatasetRef(dataset_id="ds-001", manifest_sha256="a" * 64)


def _write_task(path: Path) -> Path:
    payload = json.dumps(
            {
                "schema_version": "1.0.0",
                "task_id": "test-task",
                "task_type": "object_detection",
                "categories": [
                    {"class_id": "bottle", "display_name": "Bottle", "prompt": "bottle"},
                    {"class_id": "can", "display_name": "Can", "prompt": "can"},
                ],
            }
    )
    path.write_text(payload)
    return path


def _publish(
    work_dir: Path,
    *,
    package_id: str = "model-demo-001",
    report: EvaluationReport = None,
    policy=None,
    omit_gate_policy: bool = False,
    releases_dir: Path = None,
    dataset_dir: Optional[Path] = None,
    dataset_ref: Optional[DatasetRef] = None,
    postprocess: PostprocessSpec = None,
    class_map=None,
    onnx_path: Path = None,
    num_anchors: int = 10,
):
    """Stage publish inputs under `work_dir` and run the real publisher.

    `work_dir` doubles as the staging root so a test can pre-place an ONNX file (a
    differently-sized export, for instance) and have the publisher read those bytes.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    # Publication re-validates the data package it cites, so the default is a real package
    # with a real digest. `DATASET_REF` stays available for the tests that assert a refusal
    # when the cited dataset does not match the one supplied.
    if dataset_dir is None:
        dataset_dir = write_dataset_package(work_dir / "dataset_pkg")
    if dataset_ref is None:
        manifest, _task, _samples, _anns = validate_dataset_package(dataset_dir)
        dataset_ref = DatasetRef(
            dataset_id=manifest.dataset_id,
            manifest_sha256=compute_sha256_file(dataset_dir / "dataset.json"),
        )
    # The report must describe the data it was actually scored on. A test that passes its
    # own report is exercising something else (gate verdicts, parity, digests), so its
    # placeholder dataset identity is re-pointed at this package; the identity agreement
    # rule itself is asserted directly by test_publication_refuses_a_report_of_other_data.
    if report is not None and report.dataset.manifest_sha256 == DATASET_REF.manifest_sha256:
        report = report.model_copy(
            update={"dataset": dataset_ref},
        )
    elif report is None:
        report = make_evaluation_report(
            dataset_id=dataset_ref.dataset_id, manifest_sha256=dataset_ref.manifest_sha256
        )

    if onnx_path is None:
        model = TinyYoloMockNet(num_classes=len(class_map or CLASS_MAP), num_anchors=num_anchors)
        onnx_path, _ = export_torch_model_to_onnx(model, work_dir / "model.onnx")

    task_path = _write_task(work_dir / "task.json")
    report = report if report is not None else make_evaluation_report()
    eval_path = work_dir / "evaluation.json"
    eval_path.write_text(model_json(report, indent=2), encoding="utf-8")

    return publish_model_package(
        package_id=package_id,
        model_onnx_path=onnx_path,
        task_json_path=task_path,
        evaluation_json_path=eval_path,
        dataset_ref=dataset_ref,
        dataset_dir=dataset_dir,
        run_id=report.run_id,
        class_map=class_map or CLASS_MAP,
        releases_dir=releases_dir or (work_dir / "releases"),
        postprocess_spec=postprocess,
        # `omit_gate_policy` exists to exercise the refusal; the default keeps every other
        # test focused on its own assertion instead of repeating a policy argument.
        gate_policy=None if omit_gate_policy else (policy or make_gate_policy()),
        package_created_at="2026-10-03T00:00:00Z",
    )


def test_atomic_model_package_publication(tmp_path: Path):
    pub_dir, manifest = _publish(tmp_path)

    assert pub_dir.is_dir()
    for name in ("model.json", "model.onnx", "task.json", "evaluation.json"):
        assert (pub_dir / name).is_file()

    validated, task, evaluation = validate_model_package(pub_dir)
    assert validated.package_id == "model-demo-001"
    assert validated.created_at == "2026-10-03T00:00:00Z"
    assert task.task_id == "test-task"
    assert evaluation.gate.status == "passed"

    # `extensions` is an optional object; publishing an explicit null for it is a third,
    # undefined state a consumer would have to guess about.
    raw = json.loads((pub_dir / "model.json").read_text(encoding="utf-8"))
    assert "extensions" not in raw

    # Immutability: a published package id can never be overwritten.
    with pytest.raises(ReleasePublicationError, match="already exists"):
        _publish(tmp_path)


def test_manifest_interface_is_read_from_the_graph(tmp_path: Path):
    """Declared tensor facts come from the ONNX bytes, not from a code default."""
    from vision_model_factory.export.graph_inspection import describe_onnx_graph

    pub_dir, manifest = _publish(tmp_path)
    graph = describe_onnx_graph(pub_dir / "model.onnx")

    assert manifest.input.name == graph["input"]["name"]
    assert manifest.input.dtype == graph["input"]["dtype"]
    assert manifest.input.shape == graph["input"]["shape"]
    assert manifest.output.shape == graph["output"]["shape"]
    assert manifest.target.precision == "fp32"


def test_a_320_export_cannot_be_declared_as_640(tmp_path: Path):
    """The manifest follows the bytes; a smaller export publishes its real geometry."""
    small = tmp_path / "small320"
    model = TinyYoloMockNet(num_classes=2, num_anchors=10)
    onnx_path, _ = export_torch_model_to_onnx(model, small / "model.onnx", input_shape=(1, 3, 320, 320))

    config = DEFAULT_CONFIG.model_copy(update={"target_shape": [320, 320]})
    report = make_evaluation_report(inference_config=config, parity=make_parity_section(inference_config=config))
    _, manifest = _publish(small, package_id="model-small-001", report=report, onnx_path=onnx_path)

    assert manifest.input.shape == [1, 3, 320, 320]


def test_publication_requires_a_gate_policy(tmp_path: Path):
    """No policy, no publish. An implicit bar is what made the old gate unaccountable."""
    with pytest.raises(ReleasePublicationError, match="requires an explicit gate policy"):
        _publish(tmp_path, omit_gate_policy=True)


def test_hand_edited_gate_verdict_cannot_be_represented(tmp_path: Path):
    """Flipping a `"status"` string is refused before the publisher is even reached.

    Scope, stated honestly: a document whose measurements, gate checks and verdict are
    *all* rewritten to be self-consistent is indistinguishable from a genuine one by any
    check that reads only the document. That case is covered elsewhere - the publisher
    re-derives the verdict from the operator's own copy of the policy, and the evaluation
    report is produced by the evaluator rather than by hand. What this test pins is that
    the cheap forgeries (status flip, check flip, actual/measurements disagreement) are
    structurally impossible.
    """
    weak = make_evaluation_report(mAP50=0.2, mAP50_95=0.1, policy=make_gate_policy())
    assert weak.gate.status == "failed"

    forged = json.loads(weak.model_dump_json())
    forged["gate"]["status"] = "passed"
    with pytest.raises(Exception, match="recorded checks imply"):
        EvaluationReport.model_validate(forged)

    # Editing the recorded checks to agree with the forged verdict is not enough either:
    # a gate check must restate the number the report actually measured.
    doctored = json.loads(weak.model_dump_json())
    doctored["gate"]["status"] = "passed"
    for check in doctored["gate"]["checks"]:
        check["passed"] = True
        check["actual"] = max(check["actual"], check["threshold"])
    with pytest.raises(Exception, match="records actual"):
        EvaluationReport.model_validate(doctored)


def test_evaluation_gated_under_a_different_policy_is_refused(tmp_path: Path):
    """A report carries the digest of the policy that produced its verdict."""
    policy_a = make_gate_policy(policy_id="lenient-v1")
    policy_b = make_gate_policy(policy_id="strict-v1")
    assert policy_sha256(policy_a) != policy_sha256(policy_b)

    report = make_evaluation_report(policy=policy_a)
    with pytest.raises(ReleasePublicationError, match="hashes to"):
        _publish(tmp_path, report=report, policy=policy_b)


def test_quality_thresholds_are_re_derived_by_the_publisher(tmp_path: Path):
    """A report the operator's *current* policy rejects cannot be released, verdict or not."""
    lenient = make_gate_policy(min_map50=0.5, min_map50_95=0.4)
    strict = make_gate_policy(min_map50=0.95, min_map50_95=0.9)

    report = make_evaluation_report(mAP50=0.6, mAP50_95=0.5, policy=lenient)
    assert report.gate.status == "passed"
    assert policy_sha256(strict) != report.gate.policy_sha256
    with pytest.raises(ReleasePublicationError, match="hashes to"):
        _publish(tmp_path, report=report, policy=strict)


def test_failed_gate_blocks_publication(tmp_path: Path):
    report = make_evaluation_report(mAP50=0.2, mAP50_95=0.1, policy=make_gate_policy())
    with pytest.raises(ReleasePublicationError, match="fails gate policy"):
        _publish(tmp_path, report=report, policy=make_gate_policy())
    assert not (tmp_path / "releases" / "model-demo-001").exists()


def test_parity_failure_blocks_publication_when_policy_requires_it(tmp_path: Path):
    """Quality alone cannot carry a release once the operator demands parity."""
    strict = make_gate_policy(require_export_parity_passed=True)
    unverified = make_parity_section(
        status="failed", ref_count=0, ort_count=0, matched=False, tensor_passed=False, self_test=None
    )
    report = make_evaluation_report(policy=strict, parity=unverified)

    with pytest.raises(ReleasePublicationError, match="fails gate policy"):
        _publish(tmp_path, report=report, policy=strict)

    # Same measurements, parity actually verified -> publishable.
    good = make_evaluation_report(policy=strict, parity=make_parity_section())
    pub_dir, _ = _publish(tmp_path, package_id="model-parity-ok", report=good, policy=strict)
    assert pub_dir.is_dir()


def test_parity_measured_at_other_thresholds_is_refused(tmp_path: Path):
    """The published thresholds must be the ones the evidence was gathered under."""
    other = DEFAULT_CONFIG.model_copy(update={"score_threshold": 0.6})
    report = make_evaluation_report(
        inference_config=other, parity=make_parity_section(inference_config=other)
    )
    with pytest.raises(ReleasePublicationError, match="score_threshold"):
        _publish(tmp_path, report=report, postprocess=PostprocessSpec(score_threshold=0.25))

    # Publishing the thresholds the evidence actually used is accepted.
    pub_dir, manifest = _publish(
        tmp_path, package_id="model-threshold-match", report=report,
        postprocess=PostprocessSpec(score_threshold=0.6),
    )
    assert manifest.postprocess.score_threshold == 0.6
    assert pub_dir.is_dir()


def test_publish_refuses_a_graph_whose_output_disagrees_with_the_class_map(
    tmp_path: Path, synthetic_dataset_dir: Path
):
    """A 6-channel graph cannot be published under a declaration of 4 classes."""
    three_classes = CLASS_MAP + [ClassMapItem(index=2, class_id="lid")]
    task = tmp_path / "task3.json"
    task.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "task_id": "test-task-3",
                "task_type": "object_detection",
                "categories": [
                    {"class_id": "bottle", "display_name": "B", "prompt": "b"},
                    {"class_id": "can", "display_name": "C", "prompt": "c"},
                    {"class_id": "lid", "display_name": "L", "prompt": "l"},
                ],
            }
        )
    )
    # Cite the fixture package consistently: this test's subject is the class-map/graph
    # disagreement, not the dataset identity check.
    consistent_ref = DatasetRef(
        dataset_id="ds-test-001",
        manifest_sha256=compute_sha256_file(synthetic_dataset_dir / "dataset.json"),
    )
    report = make_evaluation_report(dataset_id="ds-test-001",
                                    manifest_sha256=consistent_ref.manifest_sha256)
    eval_path = tmp_path / "evaluation.json"
    eval_path.write_text(model_json(report, indent=2), encoding="utf-8")

    model = TinyYoloMockNet(num_classes=2, num_anchors=10)
    onnx_path, _ = export_torch_model_to_onnx(model, tmp_path / "model.onnx")

    with pytest.raises(ReleasePublicationError, match="channels"):
        publish_model_package(
            package_id="model-channel-mismatch",
            model_onnx_path=onnx_path,
            task_json_path=task,
            evaluation_json_path=eval_path,
            dataset_ref=consistent_ref,
            dataset_dir=synthetic_dataset_dir,
            run_id=report.run_id,
            class_map=three_classes,
            releases_dir=tmp_path / "rel",
            gate_policy=make_gate_policy(),
        )
    assert not (tmp_path / "rel" / "model-channel-mismatch").exists()


def test_publish_refuses_an_evaluation_report_from_another_run(tmp_path: Path, synthetic_dataset_dir: Path):
    report = make_evaluation_report(run_id="run-other")
    model = TinyYoloMockNet(num_classes=2, num_anchors=10)
    onnx_path, _ = export_torch_model_to_onnx(model, tmp_path / "model.onnx")
    eval_path = tmp_path / "evaluation.json"
    eval_path.write_text(model_json(report, indent=2), encoding="utf-8")

    with pytest.raises(ReleasePublicationError, match="describes run"):
        publish_model_package(
            package_id="model-run-mismatch",
            model_onnx_path=onnx_path,
            task_json_path=_write_task(tmp_path / "task.json"),
            evaluation_json_path=eval_path,
            dataset_ref=DATASET_REF,
            dataset_dir=synthetic_dataset_dir,
            run_id="run-001",
            class_map=CLASS_MAP,
            releases_dir=tmp_path / "rel",
            gate_policy=make_gate_policy(),
        )


def test_failed_staging_leaves_no_partial_release(tmp_path: Path):
    """A rejected package leaves the releases directory exactly as it was."""
    releases = tmp_path / "releases"
    report = make_evaluation_report(mAP50=0.1, mAP50_95=0.05, policy=make_gate_policy())
    with pytest.raises(ReleasePublicationError):
        _publish(tmp_path, report=report, policy=make_gate_policy(), releases_dir=releases)

    assert not (releases / "model-demo-001").exists()
    # No staging debris either: the finally-block removes the whole staging root.
    assert list(releases.iterdir()) == []


def test_tampered_package_bytes_are_refused_on_validation(tmp_path: Path):
    """A published package whose model bytes changed afterwards is rejected."""
    pub_dir, _ = _publish(tmp_path)

    victim = pub_dir / "model.onnx"
    original = victim.read_bytes()
    victim.write_bytes(original[:-8] + b"\x00" * 8)

    with pytest.raises(ValidationError, match="Model file SHA-256 mismatch"):
        validate_model_package(pub_dir)

    victim.write_bytes(original)
    validate_model_package(pub_dir)


def test_model_registry_lifecycle(tmp_path: Path):
    reg = ModelRegistry(tmp_path / "registry.json")

    reg.register_candidate("model-001", tmp_path / "releases" / "model-001", "0" * 64)
    assert reg.entries["model-001"]["status"] == "candidate"

    with pytest.raises(ValueError, match="Must be approved first"):
        reg.activate_model("model-001")

    reg.approve_model("model-001", evidence={"mAP50": 0.92, "verifier": "audit-suite"})
    assert reg.entries["model-001"]["status"] == "approved"

    reg.activate_model("model-001")
    assert reg.entries["model-001"]["status"] == "active"
    active = reg.get_active_model()
    assert active is not None and active["package_id"] == "model-001"

    # Activating a second package demotes the first: exactly one active model at a time.
    reg.register_candidate("model-002", tmp_path / "releases" / "model-002", "1" * 64)
    reg.approve_model("model-002", evidence={"mAP50": 0.95})
    reg.activate_model("model-002")
    assert reg.entries["model-001"]["status"] == "approved"
    assert reg.get_active_model()["package_id"] == "model-002"
    assert sum(1 for e in reg.entries.values() if e["status"] == "active") == 1


def test_publication_digests_describe_the_promoted_bytes(tmp_path: Path):
    """Every FileRef in the manifest hashes to exactly what landed in the release."""
    pub_dir, manifest = _publish(tmp_path)
    for ref in (manifest.task, manifest.model, manifest.evaluation):
        target = pub_dir / ref.path
        assert target.is_file()
        assert compute_sha256_file(target) == ref.sha256


def _republish_manifest(ds_dir: Path) -> None:
    """Rewrite dataset.json so its digests describe the tampered files.

    Without this the test would only prove a stale digest is caught; with it the package is
    internally consistent and *still* unsound, which is the case digest checks cannot see.
    """
    import json as _json

    manifest = _json.loads((ds_dir / "dataset.json").read_text(encoding="utf-8"))
    for key in ("samples", "annotations"):
        manifest[key]["sha256"] = compute_sha256_file(ds_dir / f"{key}.jsonl")
    (ds_dir / "dataset.json").write_text(_json.dumps(manifest, indent=2), encoding="utf-8")


def test_publication_refuses_a_dataset_that_is_not_what_the_package_cites(
    tmp_path: Path, synthetic_dataset_dir: Path
):
    """A5: digest agreement between report and manifest is not data soundness.

    Release 001 was published over a dataset with identical image bytes crossing splits.
    Every digest in that chain matched, because digests only prove the artifacts agree with
    each other - none of them re-reads the data. Publication now re-validates the package
    and refuses a directory whose dataset identity differs from the one being cited.
    """
    report = make_evaluation_report(
        dataset_id="ds-test-001",
        manifest_sha256=compute_sha256_file(synthetic_dataset_dir / "dataset.json"),
    )
    eval_path = tmp_path / "evaluation.json"
    eval_path.write_text(model_json(report, indent=2), encoding="utf-8")
    model = TinyYoloMockNet(num_classes=2, num_anchors=10)
    onnx_path, _ = export_torch_model_to_onnx(model, tmp_path / "model.onnx")

    # The report and the package must name the same dataset; here the package cites a
    # dataset id the report was not produced from.
    with pytest.raises(ReleasePublicationError, match="package claims dataset 'ds-other'"):
        publish_model_package(
            package_id="model-wrong-data",
            model_onnx_path=onnx_path,
            task_json_path=_write_task(tmp_path / "task.json"),
            evaluation_json_path=eval_path,
            dataset_ref=DatasetRef(dataset_id="ds-other", manifest_sha256="b" * 64),
            dataset_dir=synthetic_dataset_dir,
            run_id=report.run_id,
            class_map=CLASS_MAP,
            releases_dir=tmp_path / "rel",
            gate_policy=make_gate_policy(),
        )

    # Report and package agree with each other, but the directory handed over is different
    # bytes. This is the gap release 001 slipped through: mutual digest agreement proves
    # nothing about the data itself, so the validated directory must be the cited one.
    # A genuinely different package: the default builder is deterministic, so a second
    # call with the same task spec would produce identical bytes and an identical digest,
    # and this sub-case would prove nothing.
    other_dir = write_dataset_package(
        tmp_path / "other_ds",
        task_spec={
            **DEFAULT_TASK_SPEC,
            "task_id": "test-task-002",
            "categories": [
                {"class_id": "bottle", "display_name": "Bottle", "prompt": "bottle"},
                {"class_id": "can", "display_name": "Can", "prompt": "can"},
                {"class_id": "crate", "display_name": "Crate", "prompt": "crate"},
            ],
        },
    )
    other_sha = compute_sha256_file(other_dir / "dataset.json")
    agreeing_report = make_evaluation_report(dataset_id="ds-test-001", manifest_sha256=other_sha)
    agreeing_eval = tmp_path / "evaluation-agreeing.json"
    agreeing_eval.write_text(model_json(agreeing_report, indent=2), encoding="utf-8")
    with pytest.raises(ReleasePublicationError, match="hashes to"):
        publish_model_package(
            package_id="model-agreeing-but-other-bytes",
            model_onnx_path=onnx_path,
            task_json_path=_write_task(tmp_path / "task.json"),
            evaluation_json_path=agreeing_eval,
            dataset_ref=DatasetRef(dataset_id="ds-test-001", manifest_sha256=other_sha),
            dataset_dir=synthetic_dataset_dir,
            run_id=agreeing_report.run_id,
            class_map=CLASS_MAP,
            releases_dir=tmp_path / "rel",
            gate_policy=make_gate_policy(),
        )
    assert not (tmp_path / "rel" / "model-agreeing-but-other-bytes").exists()

    # A correct id with a stale digest is refused too.
    with pytest.raises(ReleasePublicationError, match="hashes to"):
        publish_model_package(
            package_id="model-stale-digest",
            model_onnx_path=onnx_path,
            task_json_path=_write_task(tmp_path / "task.json"),
            evaluation_json_path=eval_path,
            dataset_ref=DatasetRef(dataset_id="ds-test-001", manifest_sha256="b" * 64),
            dataset_dir=synthetic_dataset_dir,
            run_id=report.run_id,
            class_map=CLASS_MAP,
            releases_dir=tmp_path / "rel",
            gate_policy=make_gate_policy(),
        )
    assert not (tmp_path / "rel" / "model-wrong-data").exists()


def test_publication_refuses_a_leaking_dataset_even_when_every_digest_matches(tmp_path: Path):
    """The exact failure that blocked release 001, reproduced on a package.

    Identical image bytes across splits make the evaluation meaningless, yet every artifact
    can still agree with every other. Re-validation must stop the release on its own
    evidence, not because somebody happened to notice.
    """
    from factories import write_dataset_package

    ds_dir = write_dataset_package(tmp_path / "leaky_ds")
    # Make one audit image's bytes identical to a test image's, exactly as the shipped
    # package does, and update the sample record to describe those bytes. The record and
    # the file still agree, so nothing but the cross-split rule can see the duplication.
    victim = ds_dir / "images" / "img3.jpg"
    dup = ds_dir / "images" / "img5.jpg"
    dup.write_bytes(victim.read_bytes())
    rows = [json.loads(line) for line in
            (ds_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row["sample_id"] == "s-005":
            row["file"]["sha256"] = compute_sha256_file(dup)
    with (ds_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    _republish_manifest(ds_dir)

    with pytest.raises(ValidationError, match="Identical image bytes cross splits"):
        validate_dataset_package(ds_dir)

    report = make_evaluation_report(
        dataset_id="ds-test-001", manifest_sha256=compute_sha256_file(ds_dir / "dataset.json")
    )
    eval_path = tmp_path / "evaluation.json"
    eval_path.write_text(model_json(report, indent=2), encoding="utf-8")
    model = TinyYoloMockNet(num_classes=2, num_anchors=10)
    onnx_path, _ = export_torch_model_to_onnx(model, tmp_path / "model.onnx")

    with pytest.raises(ValidationError, match="Identical image bytes cross splits"):
        publish_model_package(
            package_id="model-leaky",
            model_onnx_path=onnx_path,
            task_json_path=_write_task(tmp_path / "task.json"),
            evaluation_json_path=eval_path,
            dataset_ref=DatasetRef(
                dataset_id="ds-test-001",
                manifest_sha256=compute_sha256_file(ds_dir / "dataset.json"),
            ),
            dataset_dir=ds_dir,
            run_id=report.run_id,
            class_map=CLASS_MAP,
            releases_dir=tmp_path / "rel",
            gate_policy=make_gate_policy(),
        )
    assert not (tmp_path / "rel" / "model-leaky").exists()


def test_an_int8_policy_cannot_be_satisfied_by_a_float32_graph(tmp_path: Path):
    """`precision` is read off the graph, so an int8 release cannot be claimed by paperwork.

    INT8 candidate production is out of scope for this repository, which means a policy that
    requires an int8 profile can only be satisfied by a graph that actually carries integer
    quantization nodes. The publisher derives the target from `infer_precision(model.onnx)`
    rather than from a caller's declaration, so an unmeasurable profile fails its gate
    instead of being asserted.
    """
    from vision_model_factory.contracts.gate_policy import TargetProfile

    policy = make_gate_policy(
        policy_id="needs-int8",
        min_map50=0.0,
        min_map50_95=0.0,
        required_target_profiles=[TargetProfile(precision="int8", max_latency_ms_p95=50.0)],
    )
    report = make_evaluation_report(policy=policy)
    assert report.gate.status == "failed"
    int8_checks = [c for c in report.gate.checks if c.metric.startswith("target_benchmark")]
    assert int8_checks and int8_checks[0].passed is False

    # Publication refuses, and leaves nothing staged behind.
    with pytest.raises(ReleasePublicationError, match="target_benchmark"):
        _publish(tmp_path, report=report, policy=policy, package_id="model-int8-claimed")
    assert not (tmp_path / "releases" / "model-int8-claimed").exists()
