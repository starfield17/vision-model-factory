"""Command-line composition root for Vision Model Factory.

The CLI is the only place where concrete policy and registry objects are assembled. It
holds no thresholds and no model whitelist of its own: budgets come from flags, gate
thresholds from an explicit policy document, and executable adapters/models/checkpoints
from `trainers.registry`.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from vision_model_factory.check_boundaries import check_boundaries
from vision_model_factory.contracts.gate_policy import GatePolicy, load_gate_policy_file
from vision_model_factory.contracts.hashing import compute_sha256_file
from vision_model_factory.contracts.models import (
    ClassMapItem,
    DatasetRef,
    ExportParitySection,
    InferenceConfig,
    PostprocessSpec,
    model_json,
)
from vision_model_factory.contracts.validators import (
    validate_dataset_package,
    validate_model_package,
)
from vision_model_factory.evaluation.evaluator import (
    OnnxPackagePredictor,
    TorchPackagePredictor,
    evaluate_test_split,
    measure_target_benchmark,
)
from vision_model_factory.experiments.policy import ExperimentPolicy
from vision_model_factory.experiments.runner import ExperimentRunner
from vision_model_factory.export.exporter import (
    export_torch_model_to_onnx,
    export_yolo_checkpoint_to_onnx,
)
from vision_model_factory.export.graph_inspection import describe_onnx_graph
from vision_model_factory.export.parity import check_export_parity
from vision_model_factory.release.publisher import (
    PACKAGE_MANIFEST_FILENAME,
    publish_model_package,
)
from vision_model_factory.release.registry import ModelRegistry
from vision_model_factory.trainers.registry import load_production_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-factory",
        description="Deterministic tools for training, evaluation, export, and release.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sb = subparsers.add_parser("check-boundaries", help="Check architectural layer boundaries.")
    sb.add_argument(
        "--src-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Directory containing the vision_model_factory package.",
    )

    sb = subparsers.add_parser("validate-dataset", help="Validate an immutable Dataset Package.")
    sb.add_argument("dataset_dir", type=Path)

    sb = subparsers.add_parser("validate-model", help="Validate an immutable Model Package.")
    sb.add_argument("package_dir", type=Path)

    sb = subparsers.add_parser("validate-spec", help="Validate an ExperimentSpec against policy + registry.")
    sb.add_argument("spec_file", type=Path)
    _add_budget_flags(sb)

    sb = subparsers.add_parser("run-experiment", help="Run one training experiment under a budget.")
    sb.add_argument("spec_file", type=Path)
    sb.add_argument("dataset_dir", type=Path)
    sb.add_argument("--work-dir", type=Path, default=Path("workspaces/experiments"))
    sb.add_argument("--run-id", type=str, required=True)
    sb.add_argument("--mock", action="store_true", help="Use the deterministic mock trainer.")
    sb.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Local base checkpoint file. Required for real training: unpinned downloads are refused.",
    )
    _add_budget_flags(sb)

    sb = subparsers.add_parser(
        "evaluate",
        help="Locked test evaluation + export parity + target benchmark -> evaluation.json.",
    )
    sb.add_argument("dataset_dir", type=Path)
    sb.add_argument("model_onnx", type=Path, help="Exported ONNX model to evaluate.")
    sb.add_argument("gate_policy", type=Path, help="Gate policy JSON document.")
    sb.add_argument("--task-json", type=Path, required=True, help="Upstream task.json for class coverage.")
    sb.add_argument("--run-id", type=str, required=True)
    sb.add_argument("--class-map-json", type=str, required=True, help='e.g. [{"index":0,"class_id":"bottle"}]')
    sb.add_argument("--reference-weights", type=Path, default=None, help="PyTorch weights for parity reference.")
    sb.add_argument(
        "--parity-samples",
        type=int,
        default=5,
        help="Real dataset samples used for export parity (must decode non-empty detections).",
    )
    sb.add_argument("--benchmark-runs", type=int, default=50)
    sb.add_argument("--output", type=Path, default=None, help="Write evaluation.json here.")
    sb.add_argument("--score-threshold", type=float, default=0.25)
    sb.add_argument("--nms-iou-threshold", type=float, default=0.45)
    sb.add_argument("--max-detections", type=int, default=100)
    sb.add_argument("--imgsz", type=int, default=640)

    sb = subparsers.add_parser("export", help="Export PyTorch weights to ONNX.")
    sb.add_argument("model_file", type=Path)
    sb.add_argument("output_onnx", type=Path)
    sb.add_argument("--imgsz", type=int, default=640)
    sb.add_argument("--mock", action="store_true", help="Export the deterministic mock architecture.")
    sb.add_argument("--classes", type=int, default=2)

    sb = subparsers.add_parser("publish", help="Atomically publish a verified Model Package.")
    sb.add_argument("--package-id", type=str, required=True)
    sb.add_argument("--model-path", type=Path, required=True)
    sb.add_argument("--task-path", type=Path, required=True)
    sb.add_argument("--eval-path", type=Path, required=True)
    sb.add_argument("--dataset-id", type=str, required=True)
    sb.add_argument("--dataset-manifest-sha", type=str, required=True)
    sb.add_argument("--run-id", type=str, required=True)
    sb.add_argument("--class-map-json", type=str, required=True)
    sb.add_argument("--gate-policy", type=Path, required=True, help="Publication thresholds.")
    sb.add_argument("--releases-dir", type=Path, default=Path("releases"))
    sb.add_argument("--score-threshold", type=float, default=0.25)
    sb.add_argument("--nms-iou-threshold", type=float, default=0.45)
    sb.add_argument("--max-detections", type=int, default=100)

    reg = subparsers.add_parser(
        "registry",
        help="Track published packages through candidate/approved/active states.",
    )
    reg.add_argument(
        "--registry-file", type=Path, default=Path("workspaces/registry.json"),
        help="Index file. Registry state is separate from the immutable packages it names.",
    )
    reg_actions = reg.add_mutually_exclusive_group(required=True)
    reg_actions.add_argument("--list", action="store_true", help="Show every tracked package.")
    reg_actions.add_argument("--active", action="store_true", help="Show the active package.")
    reg_actions.add_argument("--register", type=str, metavar="PACKAGE_ID",
                             help="Record a published package as a candidate.")
    reg_actions.add_argument("--approve", type=str, metavar="PACKAGE_ID",
                             help="Approve a candidate using its validation evidence.")
    reg_actions.add_argument("--activate", type=str, metavar="PACKAGE_ID",
                             help="Activate an approved package.")
    reg.add_argument("--package-dir", type=Path, default=None,
                     help="Package directory (required with --register/--approve).")
    reg.add_argument("--evidence", type=str, default=None,
                     help='Approval evidence JSON, e.g. \'{"mAP50": 0.9}\'. Required with --approve.')

    return parser


def _add_budget_flags(parser: argparse.ArgumentParser) -> None:
    """Budget flags with no defaults: a run without a stated budget is refused."""
    group = parser.add_argument_group("budget")
    group.add_argument("--max-runs", type=int, default=None, help="Maximum runs permitted (required).")
    group.add_argument("--max-total-seconds", type=float, default=None, help="Total wall clock budget (required).")
    group.add_argument(
        "--per-run-timeout-seconds", type=float, default=None, help="Per-run ceiling, enforced by termination."
    )
    group.add_argument("--fixed-seed", type=int, default=None, help="Pin every run to this seed.")


def policy_from_args(args: argparse.Namespace) -> ExperimentPolicy:
    """Build a policy from explicit flags only.

    A budget must be stated. Supplying a partial budget is an error rather than a
    silent mix of supplied values and defaults, because the default would then decide
    how much compute an Agent loop may consume.
    """
    values = {
        "max_runs": args.max_runs,
        "max_total_seconds": args.max_total_seconds,
        "per_run_timeout_seconds": args.per_run_timeout_seconds,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise SystemExit(
            "Budget flags are required to run experiments; missing: "
            + ", ".join(f"--{m.replace('_', '-')}" for m in missing)
            + ". Running an experiment loop without a declared budget is refused."
        )
    kwargs: Dict[str, Any] = dict(values)
    if args.fixed_seed is not None:
        kwargs["fixed_seed"] = args.fixed_seed
    return ExperimentPolicy(**kwargs)


def parse_class_map(raw: str) -> List[ClassMapItem]:
    return [ClassMapItem.model_validate(item) for item in json.loads(raw)]


def parse_gate_policy(path: Path) -> GatePolicy:
    return load_gate_policy_file(path)


def cmd_check_boundaries(args: argparse.Namespace) -> None:
    violations = check_boundaries(args.src_dir)
    if violations:
        print(f"FAILED: Found {len(violations)} architectural boundary violation(s):", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        raise SystemExit(1)
    print("OK: All architectural boundary checks passed successfully.")


def cmd_validate_dataset(args: argparse.Namespace) -> None:
    manifest, _task, samples, annotations = validate_dataset_package(args.dataset_dir)
    print(
        f"OK: Dataset package '{manifest.dataset_id}' valid. "
        f"Samples: {len(samples)}, Annotations: {len(annotations)}"
    )


def cmd_validate_model(args: argparse.Namespace) -> None:
    manifest, task, evaluation = validate_model_package(args.package_dir)
    print(
        f"OK: Model package '{manifest.package_id}' valid. Task: {task.task_id}, "
        f"target: {manifest.target.backend}/{manifest.target.provider}/{manifest.target.precision}, "
        f"gate: {evaluation.gate.status} (policy {evaluation.gate.policy_id})"
    )


def cmd_validate_spec(args: argparse.Namespace) -> None:
    from vision_model_factory.experiments.agent import validate_agent_proposal

    policy = policy_from_args(args)
    spec = validate_agent_proposal(
        json.loads(args.spec_file.read_text(encoding="utf-8")),
        policy,
        load_production_registry(),
    )
    print(f"OK: ExperimentSpec '{spec.experiment_id}' accepted under the declared budget and registry.")


def cmd_run_experiment(args: argparse.Namespace) -> None:
    policy = policy_from_args(args)
    spec = _load_spec(args.spec_file)
    runner = ExperimentRunner(
        policy=policy,
        work_dir=args.work_dir,
        registry=load_production_registry(),
    )
    result = runner.run_experiment(
        spec,
        args.dataset_dir,
        run_id=args.run_id,
        mock_mode=args.mock,
        checkpoint_path=args.checkpoint,
    )
    print(f"Run '{result.run_id}' finished with status '{result.status}' in {result.duration_seconds}s")
    print(model_json(result, indent=2))
    if result.status != "succeeded":
        raise SystemExit(1)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Locked test evaluation, export parity, and target benchmark -> evaluation.json.

    Geometry and thresholds are supplied explicitly here instead of being defaulted
    inside the evaluator, and the gate verdict is derived from the policy document.
    """
    class_map = parse_class_map(args.class_map_json)
    config = InferenceConfig(
        score_threshold=args.score_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        max_detections=args.max_detections,
        target_shape=[args.imgsz, args.imgsz],
    )
    gate_policy = parse_gate_policy(args.gate_policy)

    graph = describe_onnx_graph(args.model_onnx)
    _assert_graph_matches_setup(graph, config, class_map)

    _manifest, _task, samples, _annotations = validate_dataset_package(args.dataset_dir)
    test_samples = [s for s in samples if s.split == "test"]
    if len(test_samples) < args.parity_samples:
        raise SystemExit(
            f"Need at least {args.parity_samples} test samples for export parity, "
            f"found {len(test_samples)}."
        )
    parity_samples = test_samples[: args.parity_samples]

    predictor = OnnxPackagePredictor(
        class_map=class_map,
        inference_config=config,
        model_path=args.model_onnx,
        dataset_dir=args.dataset_dir,
    )

    parity_section = None
    if args.reference_weights is not None:
        parity_section = _measure_parity(args, class_map, config, parity_samples)
    else:
        print(
            "WARNING: no --reference-weights supplied; export parity is recorded as failed "
            "and gates requiring parity will not pass.",
            file=sys.stderr,
        )

    benchmark = measure_target_benchmark(
        onnx_path=args.model_onnx,
        inference_config=config,
        input_reference=f"random:seed42:{args.model_onnx.name}",
        benchmark_runs=args.benchmark_runs,
    )

    report = evaluate_test_split(
        dataset_dir=args.dataset_dir,
        predict_fn=predictor.predict,
        class_map=class_map,
        inference_config=config,
        run_id=args.run_id,
        gate_policy=gate_policy,
        export_parity=parity_section,
        target_benchmarks=[benchmark],
    )

    payload = model_json(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + chr(10), encoding="utf-8")
        print(f"Wrote evaluation report to {args.output}")
    else:
        print(payload)

    print(
        f"test mAP50={report.test.mAP50} mAP50_95={report.test.mAP50_95} "
        f"| parity={report.export_parity.status} | gate={report.gate.status} "
        f"(policy {report.gate.policy_id})"
    )
    if report.gate.status != "passed":
        failing = [_describe_check(c) for c in report.gate.checks if not c.passed]
        print(f"Failing gate checks: {failing}", file=sys.stderr)
        raise SystemExit(1)


def _describe_check(check) -> str:
    return check.metric if check.class_id is None else f"{check.metric}/{check.class_id}"


def _measure_parity(args, class_map, config, parity_samples) -> ExportParitySection:
    """Compare the exported graph against the PyTorch reference on real test images."""
    reference = TorchPackagePredictor(
        class_map=class_map,
        inference_config=config,
        weights_path=args.reference_weights,
        dataset_dir=args.dataset_dir,
    )

    report = check_export_parity(
        torch_model=reference.model,
        onnx_path=args.model_onnx,
        class_map=class_map,
        samples=parity_samples,
        dataset_dir=args.dataset_dir,
        target_shape=(config.target_shape[0], config.target_shape[1]),
        score_threshold=config.score_threshold,
        nms_iou_threshold=config.nms_iou_threshold,
        max_detections=config.max_detections,
    )
    return ExportParitySection.model_validate(report)


def _assert_graph_matches_setup(graph, config, class_map) -> None:
    """Refuse to measure a model whose graph contradicts the evaluation setup."""
    expected_spatial = [config.target_shape[1], config.target_shape[0]]
    graph_spatial = [d for d in graph["input"]["shape"][2:4] if isinstance(d, int)]
    if graph_spatial != expected_spatial:
        raise SystemExit(
            f"--imgsz spatial {expected_spatial} does not match the exported graph input "
            f"{graph['input']['shape']}; measure the artifact you intend to publish."
        )
    channels = graph["output"]["shape"][1]
    if isinstance(channels, int) and channels != 4 + len(class_map):
        raise SystemExit(
            f"Graph output has {channels} channels but --class-map-json declares "
            f"{len(class_map)} classes (decoder yolo_xywh_scores_v1 requires 4 + K)."
        )


def cmd_export(args: argparse.Namespace) -> None:
    if args.mock or not args.model_file.exists():
        from vision_model_factory.trainers.yolo import TinyYoloMockNet

        model = TinyYoloMockNet(num_classes=args.classes)
        out_path, sha = export_torch_model_to_onnx(
            model, args.output_onnx, input_shape=(1, 3, args.imgsz, args.imgsz)
        )
    else:
        out_path, sha = export_yolo_checkpoint_to_onnx(
            args.model_file, args.output_onnx, imgsz=args.imgsz
        )
    print(f"Exported ONNX model to {out_path} (SHA-256: {sha})")


def _message(exc: KeyError) -> str:
    """KeyError's str() wraps its argument in repr quotes; unwrap for operators."""
    return str(exc.args[0]) if exc.args else str(exc)


def cmd_registry(args: argparse.Namespace) -> None:
    """Drive the candidate/approved/active index.

    The registry records an operational decision about a package; it never modifies the
    package, which stays immutable once published. Approval requires evidence rather than
    a bare confirmation, so the index cannot assert a package was verified without saying
    by what.
    """
    registry = ModelRegistry(args.registry_file)

    if args.register:
        if args.package_dir is None:
            raise SystemExit("--register requires --package-dir so the index names real bytes.")
        manifest_path = args.package_dir / PACKAGE_MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise SystemExit(f"No {PACKAGE_MANIFEST_FILENAME} in {args.package_dir}: nothing to register.")
        registry.register_candidate(args.register, args.package_dir, compute_sha256_file(manifest_path))
        print(f"OK: '{args.register}' registered as candidate in {args.registry_file}")
        return

    if args.approve:
        if not args.evidence:
            raise SystemExit(
                "--approve requires --evidence: an approval must record what verified the "
                "package, otherwise the index claims a verification nobody performed."
            )
        try:
            evidence = json.loads(args.evidence)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--evidence is not valid JSON: {exc}")
        if not isinstance(evidence, dict) or not evidence:
            raise SystemExit("--evidence must be a non-empty JSON object.")
        try:
            registry.approve_model(args.approve, evidence)
        except KeyError as exc:
            raise SystemExit(f"Cannot approve: {_message(exc)}")
        print(f"OK: '{args.approve}' approved with evidence keys {sorted(evidence)}")
        return

    if args.activate:
        # Wrong-state activation is an expected operator mistake, so it is reported as a
        # refusal rather than a traceback. Integrity failures still propagate.
        try:
            registry.activate_model(args.activate)
        except KeyError as exc:
            raise SystemExit(f"Cannot activate '{args.activate}': {_message(exc)}")
        except ValueError as exc:
            raise SystemExit(f"Cannot activate '{args.activate}': {exc}")
        print(f"OK: '{args.activate}' is now the active model package")
        return

    if args.active:
        active = registry.get_active_model()
        if active is None:
            print("No active model package.")
            return
        print(json.dumps(active, indent=2, sort_keys=True))
        return

    if not registry.entries:
        print(f"Registry {args.registry_file} is empty.")
        return
    for package_id, entry in sorted(registry.entries.items()):
        print(
            f"{entry['status']:9s} {package_id}  manifest={entry['manifest_sha256'][:12]}..  "
            f"path={entry['package_path']}"
        )


def cmd_publish(args: argparse.Namespace) -> None:
    dataset_ref = DatasetRef(dataset_id=args.dataset_id, manifest_sha256=args.dataset_manifest_sha)
    postprocess = PostprocessSpec(
        score_threshold=args.score_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        max_detections=args.max_detections,
    )
    dest, manifest = publish_model_package(
        package_id=args.package_id,
        model_onnx_path=args.model_path,
        task_json_path=args.task_path,
        evaluation_json_path=args.eval_path,
        dataset_ref=dataset_ref,
        run_id=args.run_id,
        class_map=parse_class_map(args.class_map_json),
        releases_dir=args.releases_dir,
        postprocess_spec=postprocess,
        gate_policy=parse_gate_policy(args.gate_policy),
    )
    print(f"OK: Model package published atomically to {dest}")
    print(f"    declared input {manifest.input.shape}, output {manifest.output.shape} (read from graph)")


def _load_spec(path: Path):
    from vision_model_factory.contracts.models import ExperimentSpec
    from vision_model_factory.contracts.schema_validation import validate_instance

    raw = json.loads(path.read_text(encoding="utf-8"))
    validate_instance(raw, "experiment_spec")
    return ExperimentSpec.model_validate(raw)


def _load_task(path: Path):
    from vision_model_factory.contracts.models import TaskSpec
    from vision_model_factory.contracts.schema_validation import validate_instance

    raw = json.loads(path.read_text(encoding="utf-8"))
    validate_instance(raw, "task_spec")
    return TaskSpec.model_validate(raw)


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    handlers = {
        "check-boundaries": cmd_check_boundaries,
        "validate-dataset": cmd_validate_dataset,
        "validate-model": cmd_validate_model,
        "validate-spec": cmd_validate_spec,
        "run-experiment": cmd_run_experiment,
        "evaluate": cmd_evaluate,
        "export": cmd_export,
        "publish": cmd_publish,
        "registry": cmd_registry,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
