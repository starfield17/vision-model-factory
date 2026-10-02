"""Command-line interface for Vision Model Factory."""

import argparse
import json
import sys
from pathlib import Path

from vision_model_factory.check_boundaries import check_boundaries
from vision_model_factory.contracts.models import (
    ClassMapItem,
    DatasetRef,
    ExperimentSpec,
)
from vision_model_factory.contracts.validators import (
    validate_dataset_package,
    validate_model_package,
)
from vision_model_factory.experiments.policy import DEFAULT_EXPERIMENT_POLICY
from vision_model_factory.experiments.runner import ExperimentRunner
from vision_model_factory.export.exporter import (
    export_torch_model_to_onnx,
    export_yolo_checkpoint_to_onnx,
)
from vision_model_factory.release.publisher import publish_model_package
from vision_model_factory.trainers.yolo import TinyYoloMockNet


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="model-factory",
        description="Deterministic tools for model training, evaluation, export, and release.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: check-boundaries
    sb_boundaries = subparsers.add_parser("check-boundaries", help="Check architectural layer boundaries.")
    sb_boundaries.add_argument("--src-dir", type=Path, default=Path(__file__).resolve().parent.parent)

    # Subcommand: validate-dataset
    sb_val_ds = subparsers.add_parser("validate-dataset", help="Validate an immutable Dataset Package.")
    sb_val_ds.add_argument("dataset_dir", type=Path, help="Path to Dataset Package directory.")

    # Subcommand: validate-model
    sb_val_m = subparsers.add_parser("validate-model", help="Validate an immutable Model Package.")
    sb_val_m.add_argument("package_dir", type=Path, help="Path to Model Package directory.")

    # Subcommand: run-experiment
    sb_exp = subparsers.add_parser("run-experiment", help="Run training experiment from specification.")
    sb_exp.add_argument("spec_file", type=Path, help="Path to ExperimentSpec JSON file.")
    sb_exp.add_argument("dataset_dir", type=Path, help="Path to Dataset Package directory.")
    sb_exp.add_argument("--work-dir", type=Path, default=Path("workspaces/experiments"))
    sb_exp.add_argument("--run-id", type=str, default="run-001")
    sb_exp.add_argument("--mock", action="store_true", help="Run with mock trainer engine.")

    # Subcommand: export
    sb_exp_onnx = subparsers.add_parser("export", help="Export PyTorch model weights to ONNX format.")
    sb_exp_onnx.add_argument("model_file", type=Path, help="Path to PyTorch model or weights.")
    sb_exp_onnx.add_argument("output_onnx", type=Path, help="Path for exported .onnx file.")
    sb_exp_onnx.add_argument("--classes", type=int, default=2)
    sb_exp_onnx.add_argument("--mock", action="store_true", help="Export mock architecture for testing.")

    # Subcommand: publish
    sb_pub = subparsers.add_parser("publish", help="Atomically publish a verified Model Package.")
    sb_pub.add_argument("--package-id", type=str, required=True)
    sb_pub.add_argument("--model-path", type=Path, required=True)
    sb_pub.add_argument("--task-path", type=Path, required=True)
    sb_pub.add_argument("--eval-path", type=Path, required=True)
    sb_pub.add_argument("--dataset-id", type=str, required=True)
    sb_pub.add_argument("--dataset-manifest-sha", type=str, required=True)
    sb_pub.add_argument("--run-id", type=str, required=True)
    sb_pub.add_argument(
        "--class-map-json", type=str, required=True, help='JSON list e.g. [{"index":0,"class_id":"bottle"}]'
    )
    sb_pub.add_argument("--releases-dir", type=Path, default=Path("releases"))

    args = parser.parse_args()

    if args.command == "check-boundaries":
        violations = check_boundaries(args.src_dir)
        if violations:
            print(f"FAILED: Found {len(violations)} architectural boundary violation(s):", file=sys.stderr)
            for v in violations:
                print(f"  - {v}", file=sys.stderr)
            sys.exit(1)
        print("OK: All architectural boundary checks passed.")

    elif args.command == "validate-dataset":
        manifest, task, samples, annotations = validate_dataset_package(args.dataset_dir)
        print(
            f"OK: Dataset package '{manifest.dataset_id}' valid. "
            f"Samples: {len(samples)}, Annotations: {len(annotations)}"
        )

    elif args.command == "validate-model":
        manifest, task, evaluation = validate_model_package(args.package_dir)
        print(
            f"OK: Model package '{manifest.package_id}' valid. "
            f"Task: {task.task_id}, Target: {manifest.target.backend}"
        )

    elif args.command == "run-experiment":
        with args.spec_file.open("r", encoding="utf-8") as f:
            spec_data = json.load(f)
        spec = ExperimentSpec.model_validate(spec_data)
        runner = ExperimentRunner(DEFAULT_EXPERIMENT_POLICY, work_dir=args.work_dir)
        result = runner.run_experiment(spec, args.dataset_dir, run_id=args.run_id, mock_mode=args.mock)
        print(f"Experiment finished with status '{result.status}'. Duration: {result.duration_seconds}s")
        print(result.model_dump_json(indent=2))

    elif args.command == "export":
        if args.mock or not args.model_file.exists() or args.model_file.name == "mock_yolo_v1":
            model = TinyYoloMockNet(num_classes=args.classes)
            out_path, sha = export_torch_model_to_onnx(model, args.output_onnx)
        else:
            out_path, sha = export_yolo_checkpoint_to_onnx(args.model_file, args.output_onnx)
        print(f"Exported ONNX model to {out_path} (SHA-256: {sha})")

    elif args.command == "publish":
        class_map_raw = json.loads(args.class_map_json)
        class_map = [ClassMapItem.model_validate(item) for item in class_map_raw]
        dataset_ref = DatasetRef(dataset_id=args.dataset_id, manifest_sha256=args.dataset_manifest_sha)
        dest, manifest = publish_model_package(
            package_id=args.package_id,
            model_onnx_path=args.model_path,
            task_json_path=args.task_path,
            evaluation_json_path=args.eval_path,
            dataset_ref=dataset_ref,
            run_id=args.run_id,
            class_map=class_map,
            releases_dir=args.releases_dir,
        )
        print(f"OK: Model package published atomically to {dest}")


if __name__ == "__main__":
    main()
