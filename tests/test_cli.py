"""Tests for the CLI, exercised as the composition root callers actually use."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from factories import make_evaluation_report, make_gate_policy

from vision_model_factory.cli import build_parser, main
from vision_model_factory.contracts.models import model_json

TASK_JSON = {
    "schema_version": "1.0.0",
    "task_id": "test-task",
    "task_type": "object_detection",
    "categories": [
        {"class_id": "bottle", "display_name": "Bottle", "prompt": "bottle"},
        {"class_id": "can", "display_name": "Can", "prompt": "can"},
    ],
}
CLASS_MAP_JSON = json.dumps([{"index": 0, "class_id": "bottle"}, {"index": 1, "class_id": "can"}])


def test_cli_help_lists_every_subcommand(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for command in (
        "check-boundaries", "validate-dataset", "validate-model", "validate-spec",
        "run-experiment", "evaluate", "export", "publish",
    ):
        assert command in out


def test_check_boundaries_subcommand_reports_clean(capsys):
    main(["check-boundaries"])
    assert "boundary checks passed" in capsys.readouterr().out


def test_module_entry_point_runs_without_a_installed_console_script():
    """The documented invocation (`python -m ...cli`) must work from the repo root."""
    repo = Path(__file__).resolve().parent.parent
    env_result = subprocess.run(
        [sys.executable, "-m", "vision_model_factory.cli", "--help"],
        cwd=repo, capture_output=True, text=True, env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
    )
    assert env_result.returncode == 0, env_result.stderr
    assert "check-boundaries" in env_result.stdout


def test_cli_validate_dataset(synthetic_dataset_dir: Path, capsys):
    main(["validate-dataset", str(synthetic_dataset_dir)])
    out = capsys.readouterr().out
    assert "ds-test-001" in out
    assert "Samples: 4" in out


def test_cli_validate_dataset_fails_loudly_on_tampered_bytes(synthetic_dataset_dir: Path):
    """The CLI surfaces validation failures rather than printing a warning and continuing."""
    from vision_model_factory.contracts.validators import ValidationError

    (synthetic_dataset_dir / "samples.jsonl").write_text("corrupted\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="Samples file SHA-256 mismatch"):
        main(["validate-dataset", str(synthetic_dataset_dir)])


def _write_policy(path: Path, policy) -> Path:
    path.write_text(json.dumps(policy.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def _publish_inputs(tmp_path: Path, policy=None):
    """Produce a publishable package through the CLI, returning the paths involved.

    The report and the policy file must come from the *same* policy object: the publisher
    re-derives the verdict from the supplied policy and compares digests, which is exactly
    what it is supposed to do.
    """
    from vision_model_factory.export.exporter import export_torch_model_to_onnx
    from vision_model_factory.trainers.yolo import TinyYoloMockNet

    policy = policy or make_gate_policy(policy_id="cli-policy-v1")
    onnx_path, _ = export_torch_model_to_onnx(TinyYoloMockNet(num_classes=2, num_anchors=10),
                                             tmp_path / "model.onnx")
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(TASK_JSON), encoding="utf-8")

    report = make_evaluation_report(dataset_id="ds-cli-001", manifest_sha256="c" * 64, policy=policy)
    eval_path = tmp_path / "evaluation.json"
    eval_path.write_text(model_json(report, indent=2), encoding="utf-8")

    return onnx_path, task_path, eval_path, _write_policy(tmp_path / "policy.json", policy), report


def test_cli_publish_then_validate_round_trip(tmp_path: Path, capsys):
    onnx_path, task_path, eval_path, policy_path, report = _publish_inputs(tmp_path)
    releases = tmp_path / "releases"

    main([
        "publish",
        "--package-id", "model-cli-001",
        "--model-path", str(onnx_path),
        "--task-path", str(task_path),
        "--eval-path", str(eval_path),
        "--dataset-id", "ds-cli-001",
        "--dataset-manifest-sha", "c" * 64,
        "--run-id", report.run_id,
        "--class-map-json", CLASS_MAP_JSON,
        "--gate-policy", str(policy_path),
        "--releases-dir", str(releases),
    ])
    out = capsys.readouterr().out
    assert "published atomically" in out
    assert (releases / "model-cli-001" / "model.json").is_file()

    main(["validate-model", str(releases / "model-cli-001")])
    out = capsys.readouterr().out
    assert "model-cli-001" in out
    assert "gate: passed" in out
    assert "cli-policy-v1" in out


def test_cli_publish_requires_a_gate_policy(tmp_path: Path, capsys):
    onnx_path, task_path, eval_path, _policy, report = _publish_inputs(tmp_path)

    argv = [
        "publish", "--package-id", "model-cli-np", "--model-path", str(onnx_path),
        "--task-path", str(task_path), "--eval-path", str(eval_path),
        "--dataset-id", "ds-cli-001", "--dataset-manifest-sha", "c" * 64,
        "--run-id", report.run_id, "--class-map-json", CLASS_MAP_JSON,
        "--releases-dir", str(tmp_path / "releases"),
    ]
    # argparse reports a missing required flag on stderr and exits non-zero. Unlike the
    # budget check above, this message comes from argparse rather than SystemExit text.
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code != 0
    assert "--gate-policy" in capsys.readouterr().err


def test_cli_publish_refuses_a_failing_gate(tmp_path: Path, capsys):
    """The CLI cannot be used to release a model the policy rejects."""
    from vision_model_factory.export.exporter import export_torch_model_to_onnx
    from vision_model_factory.trainers.yolo import TinyYoloMockNet

    onnx_path, _ = export_torch_model_to_onnx(TinyYoloMockNet(num_classes=2, num_anchors=10),
                                             tmp_path / "model.onnx")
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(TASK_JSON), encoding="utf-8")

    report = make_evaluation_report(
        dataset_id="ds-cli-001", manifest_sha256="c" * 64, mAP50=0.2, mAP50_95=0.1,
        policy=make_gate_policy(policy_id="cli-strict"),
    )
    eval_path = tmp_path / "evaluation.json"
    eval_path.write_text(model_json(report, indent=2), encoding="utf-8")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(make_gate_policy(policy_id="cli-strict").model_dump(mode="json")), encoding="utf-8"
    )

    with pytest.raises(Exception, match="fails gate policy"):
        main([
            "publish", "--package-id", "model-cli-bad", "--model-path", str(onnx_path),
            "--task-path", str(task_path), "--eval-path", str(eval_path),
            "--dataset-id", "ds-cli-001", "--dataset-manifest-sha", "c" * 64,
            "--run-id", report.run_id, "--class-map-json", CLASS_MAP_JSON,
            "--gate-policy", str(policy_path), "--releases-dir", str(tmp_path / "releases"),
        ])
    assert not (tmp_path / "releases" / "model-cli-bad").exists()


def test_run_experiment_requires_every_budget_flag(synthetic_dataset_dir: Path, tmp_path: Path, capsys):
    spec_path = tmp_path / "spec.json"
    spec = {
        "schema_version": "1.0.0",
        "experiment_id": "exp-cli-001",
        "dataset": {"dataset_id": "ds-test-001", "manifest_sha256": "0" * 64},
        "trainer": {"adapter_id": "yolo_detection_v1", "model_id": "mock_yolo_v1",
                    "checkpoint_sha256": "0" * 64},
        "params": {"imgsz": 640, "batch": 8, "epochs": 1, "seed": 42, "lr0": 0.01, "mosaic": 1.0},
        "reason": "CLI budget check",
    }
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    base = ["run-experiment", str(spec_path), str(synthetic_dataset_dir),
            "--work-dir", str(tmp_path / "work"), "--run-id", "run-cli"]

    # A missing budget exits with the missing flags in its message. `SystemExit` text is
    # printed by the interpreter, so a caught exit is asserted through `.code`.
    names = ("--max-runs", "--max-total-seconds", "--per-run-timeout-seconds")
    with pytest.raises(SystemExit) as exit_info:
        main(base)
    message = str(exit_info.value.args[0])
    for flag in names:
        assert flag in message

    # A partial budget is refused too: a defaulted half-budget would silently decide how
    # much compute an agent loop may consume.
    with pytest.raises(SystemExit) as partial:
        main(base + ["--max-runs", "1"])
    assert "--max-total-seconds" in str(partial.value.args[0])
    assert "--per-run-timeout-seconds" in str(partial.value.code)

    # The full budget is accepted and the digest check fires before any training starts.
    budget = base + ["--max-runs", "1", "--max-total-seconds", "60", "--per-run-timeout-seconds", "30"]
    with pytest.raises(ValueError, match="Dataset package digest mismatch"):
        main(budget)


def test_validate_spec_accepts_a_registry_pinned_proposal(tmp_path: Path, capsys):
    spec = {
        "schema_version": "1.0.0",
        "experiment_id": "exp-cli-002",
        "dataset": {"dataset_id": "ds-001", "manifest_sha256": "a" * 64},
        "trainer": {"adapter_id": "yolo_detection_v1", "model_id": "mock_yolo_v1",
                    "checkpoint_sha256": "0" * 64},
        "params": {"imgsz": 640, "batch": 8, "epochs": 1, "seed": 42, "lr0": 0.01, "mosaic": 1.0},
        "reason": "CLI spec validation",
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    main(["validate-spec", str(spec_path), "--max-runs", "1", "--max-total-seconds", "60",
          "--per-run-timeout-seconds", "30"])
    assert "exp-cli-002" in capsys.readouterr().out

    bad = dict(spec, trainer={"adapter_id": "nope", "model_id": "mock_yolo_v1",
                             "checkpoint_sha256": "0" * 64})
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    from vision_model_factory.experiments.agent import AgentProposalViolation

    with pytest.raises(AgentProposalViolation, match="rejected by registry"):
        main(["validate-spec", str(bad_path), "--max-runs", "1", "--max-total-seconds", "60",
              "--per-run-timeout-seconds", "30"])


def test_export_subcommand_writes_a_loadable_graph(tmp_path: Path, capsys):
    out = tmp_path / "cli_model.onnx"
    main(["export", "does_not_exist.pt", str(out), "--mock", "--classes", "2"])
    captured = capsys.readouterr().out
    assert "Exported ONNX model to" in captured
    assert out.is_file()

    from vision_model_factory.export.graph_inspection import describe_onnx_graph

    graph = describe_onnx_graph(out)
    assert graph["input"]["shape"] == [1, 3, 640, 640]
    assert graph["output"]["shape"][1] == 6  # 4 + 2 classes


def test_parser_has_no_silent_budget_defaults():
    """Budget flags default to None so the CLI can tell 'unset' from 'zero'."""
    parser = build_parser()
    args = parser.parse_args([
        "validate-spec", "spec.json", "--max-runs", "2",
        "--max-total-seconds", "10", "--per-run-timeout-seconds", "5",
    ])
    assert (args.max_runs, args.max_total_seconds, args.per_run_timeout_seconds) == (2, 10.0, 5.0)
    assert args.fixed_seed is None


class TestPipelineScenario:
    """The SPEC's delivery scenario, driven through the CLI rather than through functions.

    validate-dataset -> export -> evaluate -> publish -> validate-model, on a real ONNX
    graph and real image bytes. Unit tests prove the pieces; this proves the pipeline a
    caller actually runs, including that the artifacts each step writes are accepted by
    the next.
    """

    __test__ = True

    @staticmethod
    def _dataset(tmp_path: Path) -> Path:
        import numpy as np
        from PIL import Image

        from vision_model_factory.contracts.hashing import compute_sha256_file

        ds = tmp_path / "ds"
        (ds / "images").mkdir(parents=True)
        (ds / "task.json").write_text(json.dumps(TASK_JSON), encoding="utf-8")

        layout = [("s-1", "train", "bottle"), ("s-2", "val", "can"),
                  ("s-3", "test", "bottle"), ("s-4", "test", "can")]
        shas = {}
        for i, (sid, split, _cls) in enumerate(layout):
            img_path = ds / "images" / f"{sid}.jpg"
            arr = np.zeros((480, 640, 3), dtype=np.uint8)
            arr[:, :] = 30 + 40 * i
            arr[120:300, 100:500] = 210 - 30 * i
            Image.fromarray(arr).save(img_path)
            shas[sid] = (compute_sha256_file(img_path), split)

        samples = [
            {"sample_id": sid, "file": {"path": f"images/{sid}.jpg", "sha256": shas[sid][0]},
             "width": 640, "height": 480, "group_id": f"g-{sid}", "split": shas[sid][1],
             "annotation_status": "complete_verified"}
            for sid in shas
        ]
        (ds / "samples.jsonl").write_text(
            "\n".join(json.dumps(s) for s in samples) + "\n", encoding="utf-8"
        )
        anns = [
            {"annotation_id": f"a{i}", "sample_id": sid, "class_id": cls,
             "bbox_xyxy": [100.0, 120.0, 500.0, 300.0], "origin": "human",
             "annotator_run_id": "run-ann", "review_state": "human_verified", "score": 1.0}
            for i, (sid, _split, cls) in enumerate(layout)
        ]
        (ds / "annotations.jsonl").write_text(
            "\n".join(json.dumps(a) for a in anns) + "\n", encoding="utf-8"
        )
        (ds / "quality.json").write_text(
            json.dumps({"schema_version": "1.0.0", "annotation_runs": [], "reviewer_runs": [],
                        "audit": {"sample_count": 4}, "gate": {"status": "passed"}}),
            encoding="utf-8",
        )
        (ds / "dataset.json").write_text(
            json.dumps({
                "schema_version": "1.0.0", "dataset_id": "ds-e2e",
                "created_at": "2026-10-03T00:00:00Z",
                "task": {"path": "task.json", "sha256": compute_sha256_file(ds / "task.json")},
                "samples": {"path": "samples.jsonl",
                            "sha256": compute_sha256_file(ds / "samples.jsonl")},
                "annotations": {"path": "annotations.jsonl",
                                "sha256": compute_sha256_file(ds / "annotations.jsonl")},
                "quality": {"path": "quality.json",
                            "sha256": compute_sha256_file(ds / "quality.json")},
            }),
            encoding="utf-8",
        )
        return ds

    def test_evaluate_then_publish_then_validate(self, tmp_path: Path, capsys):
        from vision_model_factory.export.exporter import export_torch_model_to_onnx
        from vision_model_factory.trainers.yolo import TinyYoloMockNet

        ds = self._dataset(tmp_path)
        main(["validate-dataset", str(ds)])
        assert "ds-e2e" in capsys.readouterr().out

        # Export the reference weights' own graph, so parity compares the artifact that
        # will be published against the weights it came from.
        weights = tmp_path / "best.pt"
        import torch

        net = TinyYoloMockNet(num_classes=2, num_anchors=20)
        torch.save(net.state_dict(), weights)
        onnx_path = tmp_path / "model.onnx"
        export_torch_model_to_onnx(TinyYoloMockNet(num_classes=2, num_anchors=20), onnx_path)

        # A mock net decodes nothing meaningful on synthetic fills, so the quality floors
        # are set to zero to exercise the *plumbing*. A policy with real floors correctly
        # fails, and the sibling test below asserts exactly that.
        policy_path = _write_policy(tmp_path / "policy.json",
                                    make_gate_policy(policy_id="e2e-policy-v1",
                                                     min_map50=0.0, min_map50_95=0.0))
        eval_out = tmp_path / "evaluation.json"

        # No --reference-weights, so parity is recorded as failed. This policy does not
        # require parity, so the quality floors still decide the verdict: exit 0 and a
        # publishable report. The sibling test below shows a parity-requiring policy
        # refusing the very same measurements.
        main([
            "evaluate", str(ds), str(onnx_path), str(policy_path),
            "--task-json", str(ds / "task.json"),
            "--run-id", "run-e2e",
            "--class-map-json", CLASS_MAP_JSON,
            "--parity-samples", "2",
            "--benchmark-runs", "3",
            "--output", str(eval_out),
        ])
        assert eval_out.is_file(), "evaluate must write the report it was asked to write"

        report = json.loads(eval_out.read_text(encoding="utf-8"))
        assert report["run_id"] == "run-e2e"
        assert report["dataset"]["dataset_id"] == "ds-e2e"
        # Sections stay separate and measured, never collapsed into one score.
        assert report["test"]["protocol_id"] == "locked_test_per_image_ap_v1"
        assert report["test"]["sample_count"] == 2
        assert report["export_parity"]["status"] == "failed"
        assert report["export_parity"]["method"] == "not_run"
        assert report["gate"]["status"] == "passed"
        assert report["gate"]["policy_id"] == "e2e-policy-v1"
        assert report["gate"]["policy_sha256"] != "0" * 64
        assert report["target_benchmarks"], "a benchmark must be recorded for the measured profile"
        printed = capsys.readouterr().out
        assert "parity=failed" in printed
        assert "gate=passed" in printed

        # Publishing under a policy that *does* require parity must be refused: the report
        # was gated under a different policy, and its digest proves it.
        releases = tmp_path / "releases"
        parity_required = _write_policy(tmp_path / "strict.json",
                                        make_gate_policy(policy_id="e2e-needs-parity",
                                                         min_map50=0.0, min_map50_95=0.0,
                                                         require_export_parity_passed=True))
        with pytest.raises(Exception, match="hashes to"):
            main([
                "publish", "--package-id", "model-e2e-blocked", "--model-path", str(onnx_path),
                "--task-path", str(ds / "task.json"), "--eval-path", str(eval_out),
                "--dataset-id", "ds-e2e",
                "--dataset-manifest-sha", str(report["dataset"]["manifest_sha256"]),
                "--run-id", "run-e2e", "--class-map-json", CLASS_MAP_JSON,
                "--gate-policy", str(parity_required), "--releases-dir", str(releases),
            ])
        assert not (releases / "model-e2e-blocked").exists()

        main([
            "publish", "--package-id", "model-e2e-001", "--model-path", str(onnx_path),
            "--task-path", str(ds / "task.json"), "--eval-path", str(eval_out),
            "--dataset-id", "ds-e2e",
            "--dataset-manifest-sha", str(report["dataset"]["manifest_sha256"]),
            "--run-id", "run-e2e", "--class-map-json", CLASS_MAP_JSON,
            "--gate-policy", str(policy_path), "--releases-dir", str(releases),
        ])
        assert "published atomically" in capsys.readouterr().out

        main(["validate-model", str(releases / "model-e2e-001")])
        assert "model-e2e-001" in capsys.readouterr().out

    def test_evaluate_records_failed_parity_and_blocks_a_parity_requiring_policy(
        self, tmp_path: Path, capsys
    ):
        """Without reference weights, parity is recorded as failed — never as skipped."""
        from vision_model_factory.export.exporter import export_torch_model_to_onnx
        from vision_model_factory.trainers.yolo import TinyYoloMockNet

        ds = self._dataset(tmp_path)
        onnx_path = tmp_path / "model.onnx"
        export_torch_model_to_onnx(TinyYoloMockNet(num_classes=2, num_anchors=20), onnx_path)
        strict = _write_policy(tmp_path / "strict.json",
                              make_gate_policy(policy_id="needs-parity",
                                               require_export_parity_passed=True))

        with pytest.raises(SystemExit) as exit_info:
            main([
                "evaluate", str(ds), str(onnx_path), str(strict),
                "--task-json", str(ds / "task.json"), "--run-id", "run-np",
                "--class-map-json", CLASS_MAP_JSON, "--parity-samples", "2",
                "--benchmark-runs", "2", "--output", str(tmp_path / "e.json"),
            ])
        assert exit_info.value.code == 1
        report = json.loads((tmp_path / "e.json").read_text(encoding="utf-8"))
        assert report["export_parity"]["status"] == "failed"
        assert report["export_parity"]["method"] == "not_run"
        assert report["gate"]["status"] == "failed"
        assert "export_parity_passed" in capsys.readouterr().err

    def test_evaluate_refuses_a_graph_that_contradicts_the_class_map(self, tmp_path: Path, capsys):
        """Measuring the wrong artifact is refused before any number is produced."""
        from vision_model_factory.export.exporter import export_torch_model_to_onnx
        from vision_model_factory.trainers.yolo import TinyYoloMockNet

        ds = self._dataset(tmp_path)
        onnx_path = tmp_path / "model.onnx"
        export_torch_model_to_onnx(TinyYoloMockNet(num_classes=2, num_anchors=20), onnx_path)
        policy_path = _write_policy(tmp_path / "p.json", make_gate_policy(policy_id="p"))

        three_classes = json.dumps([{"index": 0, "class_id": "bottle"},
                                   {"index": 1, "class_id": "can"},
                                   {"index": 2, "class_id": "lid"}])
        task3 = ds / "task.json"
        task3.write_text(json.dumps({
            "schema_version": "1.0.0", "task_id": "t3", "task_type": "object_detection",
            "categories": [
                {"class_id": "bottle", "display_name": "B", "prompt": "b"},
                {"class_id": "can", "display_name": "C", "prompt": "c"},
                {"class_id": "lid", "display_name": "L", "prompt": "l"},
            ],
        }), encoding="utf-8")

        with pytest.raises(SystemExit) as exit_info:
            main([
                "evaluate", str(ds), str(onnx_path), str(policy_path),
                "--task-json", str(task3), "--run-id", "run-bad",
                "--class-map-json", three_classes, "--parity-samples", "2",
                "--benchmark-runs", "1",
            ])
        assert "channels" in str(exit_info.value.args[0]) or "channels" in capsys.readouterr().err
        assert exit_info.value.code != 0


def test_registry_lifecycle_through_the_cli(tmp_path: Path, capsys):
    """`ModelRegistry` is reachable from the CLI, not only by importing it."""
    registry_file = tmp_path / "registry.json"
    pkg = tmp_path / "releases" / "model-x"
    pkg.mkdir(parents=True)
    (pkg / "model.json").write_text(json.dumps({"package_id": "model-x"}), encoding="utf-8")

    base = ["registry", "--registry-file", str(registry_file)]

    main(base + ["--list"])
    assert "empty" in capsys.readouterr().out

    main(base + ["--register", "model-x", "--package-dir", str(pkg)])
    assert "candidate" in capsys.readouterr().out

    # Candidate -> active is refused, and as a clean refusal rather than a traceback.
    with pytest.raises(SystemExit) as exit_info:
        main(base + ["--activate", "model-x"])
    assert "Must be approved first" in str(exit_info.value.args[0])

    # Approval without evidence is refused: the index would claim a verification nobody
    # performed.
    with pytest.raises(SystemExit) as exit_info:
        main(base + ["--approve", "model-x"])
    assert "--evidence" in str(exit_info.value.args[0])

    with pytest.raises(SystemExit):
        main(base + ["--approve", "model-x", "--evidence", "not json"])
    with pytest.raises(SystemExit):
        main(base + ["--approve", "model-x", "--evidence", "[]"])

    main(base + ["--approve", "model-x", "--evidence", json.dumps({"mAP50": 0.93})])
    assert "approved" in capsys.readouterr().out

    main(base + ["--activate", "model-x"])
    assert "active model package" in capsys.readouterr().out

    main(base + ["--active"])
    active = json.loads(capsys.readouterr().out)
    assert active["package_id"] == "model-x"
    assert active["approval_evidence"] == {"mAP50": 0.93}
    # The registered digest is the real manifest digest, not a placeholder.
    registry_state = json.loads(registry_file.read_text(encoding="utf-8"))
    assert registry_state["model-x"]["manifest_sha256"] == hashlib.sha256(
        (pkg / "model.json").read_bytes()
    ).hexdigest()

    with pytest.raises(SystemExit) as exit_info:
        main(base + ["--register", "model-y"])
    assert "--package-dir" in str(exit_info.value.args[0])
