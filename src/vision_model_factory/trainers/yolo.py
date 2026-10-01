"""YOLO object detection trainer adapter for Vision Model Factory."""

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import yaml

from vision_model_factory.contracts.hashing import compute_sha256_file
from vision_model_factory.contracts.models import (
    AnnotationRecord,
    ClassMapItem,
    TaskSpec,
)
from vision_model_factory.contracts.validators import validate_dataset_package
from vision_model_factory.trainers.base import (
    BaseTrainerAdapter,
    TrainerConfig,
    TrainerRunResult,
)

logger = logging.getLogger(__name__)


class TinyYoloMockNet(nn.Module):
    """Tiny PyTorch neural network producing YOLO format outputs [1, 4 + K, N] for fast verification."""

    def __init__(self, num_classes: int = 2, num_anchors: int = 20):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv = nn.Conv2d(3, 4 + num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        channels = 4 + self.num_classes
        feat = self.conv(x)[:, :, 0, : self.num_anchors]
        out = torch.zeros((batch_size, channels, self.num_anchors), dtype=torch.float32, device=x.device)
        out[:, 0, :] = 320.0  # cx
        out[:, 1, :] = 320.0  # cy
        out[:, 2, :] = 100.0  # w
        out[:, 3, :] = 100.0  # h
        out[:, 4, :] = 0.95   # class 0 score
        if self.num_classes > 1:
            out[:, 5:, :] = 0.05
        return out + 0.0 * feat


class YoloTrainerAdapter(BaseTrainerAdapter):
    """Trainer adapter for YOLOv8 object detection."""

    @property
    def adapter_id(self) -> str:
        return "yolo_detection_v1"

    def prepare_dataset(
        self,
        dataset_dir: Path,
        work_dir: Path,
        task: TaskSpec,
        class_map: List[ClassMapItem],
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Convert immutable dataset package to YOLO directory structure.
        Strictly excludes samples marked 'partial', logging exclusion stats.
        """
        manifest, task_spec, samples, annotations = validate_dataset_package(dataset_dir)

        # Mapping from class_id to integer index
        class_to_idx = {item.class_id: item.index for item in class_map}

        yolo_root = work_dir / "yolo_data"
        train_img_dir = yolo_root / "images" / "train"
        val_img_dir = yolo_root / "images" / "val"
        train_lbl_dir = yolo_root / "labels" / "train"
        val_lbl_dir = yolo_root / "labels" / "val"

        for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Index annotations by sample_id
        ann_by_sample: Dict[str, List[AnnotationRecord]] = {}
        for ann in annotations:
            ann_by_sample.setdefault(ann.sample_id, []).append(ann)

        excluded_partial_count = 0
        train_sample_count = 0
        val_sample_count = 0

        for s in samples:
            # Audit and exclude partial annotations
            if s.annotation_status == "partial":
                excluded_partial_count += 1
                logger.info(f"Excluding partial sample '{s.sample_id}' from training dataset")
                continue

            if s.split not in ("train", "val"):
                continue

            target_img_dir = train_img_dir if s.split == "train" else val_img_dir
            target_lbl_dir = train_lbl_dir if s.split == "train" else val_lbl_dir

            if s.split == "train":
                train_sample_count += 1
            else:
                val_sample_count += 1

            # Place image
            src_img_path = (dataset_dir / s.file.path).resolve()
            dst_img_path = target_img_dir / f"{s.sample_id}{src_img_path.suffix}"
            if src_img_path.is_file() and not dst_img_path.exists():
                try:
                    os.symlink(src_img_path, dst_img_path)
                except OSError:
                    shutil.copy2(src_img_path, dst_img_path)
            elif not src_img_path.is_file():
                # If image bytes were not stored, create an empty placeholder for mock mode
                dst_img_path.touch()

            # Write YOLO label: <class_idx> <cx> <cy> <w> <h> (normalized 0..1)
            lbl_file = target_lbl_dir / f"{s.sample_id}.txt"
            lines: List[str] = []
            sample_anns = ann_by_sample.get(s.sample_id, [])

            for a in sample_anns:
                if a.class_id not in class_to_idx:
                    continue
                c_idx = class_to_idx[a.class_id]
                x1, y1, x2, y2 = a.bbox_xyxy

                cx = ((x1 + x2) / 2.0) / s.width
                cy = ((y1 + y2) / 2.0) / s.height
                w = (x2 - x1) / s.width
                h = (y2 - y1) / s.height

                # Ensure values stay in [0, 1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))

                lines.append(f"{c_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            lbl_file.write_text("\n".join(lines), encoding="utf-8")

        # Write data.yaml
        names_dict = {item.index: item.class_id for item in class_map}
        data_yaml_content = {
            "path": str(yolo_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": names_dict,
        }

        yaml_path = yolo_root / "data.yaml"
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data_yaml_content, f, sort_keys=False)

        stats = {
            "train_samples": train_sample_count,
            "val_samples": val_sample_count,
            "excluded_partial_samples": excluded_partial_count,
            "num_classes": len(class_map),
        }
        return yaml_path, stats

    def train(self, config: TrainerConfig) -> TrainerRunResult:
        """Execute training or mock training."""
        t0 = time.time()
        output_dir = config.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare dataset
        task_path = config.dataset_dir / "task.json"
        with task_path.open("r", encoding="utf-8") as f:
            task = TaskSpec.model_validate(json.load(f))

        data_yaml_path, prep_stats = self.prepare_dataset(
            config.dataset_dir, output_dir, task, config.class_map
        )

        if config.mock_mode or config.model_id == "mock_yolo_v1":
            # Deterministic mock training
            mock_model = TinyYoloMockNet(num_classes=len(config.class_map), num_anchors=20)
            ckpt_path = output_dir / "best.pt"
            torch.save(mock_model.state_dict(), ckpt_path)
            sha256 = compute_sha256_file(ckpt_path)

            duration = time.time() - t0
            val_metrics = {
                "mAP50": 0.88,
                "mAP50-95": 0.65,
                "precision": 0.90,
                "recall": 0.85,
                "val_samples": prep_stats["val_samples"],
            }
            return TrainerRunResult(
                status="succeeded",
                duration_seconds=round(duration, 3),
                checkpoint_path=ckpt_path,
                checkpoint_sha256=sha256,
                val_metrics=val_metrics,
                artifacts={"best_checkpoint": ckpt_path, "data_yaml": data_yaml_path},
                diagnostics=[
                    f"Mock training completed. Excluded {prep_stats['excluded_partial_samples']} partial samples."
                ],
            )

        # Real Ultralytics execution
        try:
            from ultralytics import YOLO

            model = YOLO(config.model_id)
            model.train(
                data=str(data_yaml_path),
                epochs=config.params.epochs,
                imgsz=config.params.imgsz,
                batch=config.params.batch,
                seed=config.params.seed,
                lr0=config.params.lr0,
                mosaic=config.params.mosaic,
                project=str(output_dir),
                name="run",
                exist_ok=True,
                verbose=False,
            )

            weights_path = output_dir / "run" / "weights" / "best.pt"
            if not weights_path.is_file():
                weights_path = output_dir / "run" / "weights" / "last.pt"

            if not weights_path.is_file():
                return TrainerRunResult(
                    status="failed",
                    duration_seconds=round(time.time() - t0, 3),
                    diagnostics=["No checkpoint weights found after training"],
                )

            sha256 = compute_sha256_file(weights_path)
            duration = time.time() - t0

            # Val metrics
            val_results = model.val(data=str(data_yaml_path), split="val", verbose=False)
            metrics = {
                "mAP50": float(val_results.box.map50),
                "mAP50-95": float(val_results.box.map),
                "precision": float(val_results.box.mp),
                "recall": float(val_results.box.mr),
                "val_samples": prep_stats["val_samples"],
            }

            return TrainerRunResult(
                status="succeeded",
                duration_seconds=round(duration, 3),
                checkpoint_path=weights_path,
                checkpoint_sha256=sha256,
                val_metrics=metrics,
                artifacts={"best_checkpoint": weights_path, "data_yaml": data_yaml_path},
                diagnostics=[
                    f"Ultralytics training finished. Excluded {prep_stats['excluded_partial_samples']} partial samples."
                ],
            )
        except Exception as e:
            return TrainerRunResult(
                status="failed",
                duration_seconds=round(time.time() - t0, 3),
                diagnostics=[f"Training error: {str(e)}"],
            )
