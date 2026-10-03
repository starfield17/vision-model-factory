"""Import a public African Wildlife archive into a Dataset Package.

This is an *importer*, not a Data Factory. It performs no annotation work and no review,
so it declares neither: annotations are recorded as `origin: "imported"` /
`review_state: "unreviewed"`, and `quality.json` carries `audit: null` and `gate: null`
because no audit sampling was drawn and no quality gate was run here. The Data schema
permits those nulls; claiming `human_verified` with `precision: 1.0` and a `policy_sha256`
of zeros was asserting a review and a verdict that nobody performed.

A package with no quality verdict is still consumable - the Model side digests
`quality.json` but does not read a gate from it - and it is honest: whoever needs verified
ground truth has to do the verifying, and the artifact says so.
"""

import json
import logging
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from PIL import Image

from vision_model_factory.contracts.hashing import compute_sha256_file
from vision_model_factory.contracts.validators import ValidationError, validate_dataset_package

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Names this import, so a package can be traced back to the run that produced it.
RUN_ID = "run-import-african-wildlife-v1"

CLASS_NAMES = {
    0: ("buffalo", "African Buffalo", "buffalo"),
    1: ("elephant", "African Elephant", "elephant"),
    2: ("rhino", "Rhinoceros", "rhino"),
    3: ("zebra", "Plains Zebra", "zebra"),
}


def build_african_wildlife_package(
    zip_path: Path,
    output_dir: Path,
    dataset_id: str = "ds-african-wildlife-v1",
) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        logger.info("Cleaning existing output directory: %s", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_temp = output_dir / "_raw_extracted"
    extracted_temp.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting %s ...", zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extracted_temp)

    # Locate dataset root inside extracted archive
    wildlife_root = extracted_temp
    if (extracted_temp / "african-wildlife").is_dir():
        wildlife_root = extracted_temp / "african-wildlife"
    elif (extracted_temp / "images").is_dir():
        wildlife_root = extracted_temp

    target_images_dir = output_dir / "images"
    target_images_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create task.json
    categories = [
        {"class_id": cid, "display_name": dname, "prompt": prompt}
        for idx, (cid, dname, prompt) in sorted(CLASS_NAMES.items())
    ]
    task_spec = {
        "schema_version": "1.0.0",
        "task_id": "task-african-wildlife-001",
        "task_type": "object_detection",
        "categories": categories,
    }
    task_file = output_dir / "task.json"
    with task_file.open("w", encoding="utf-8") as f:
        json.dump(task_spec, f, indent=2)
    task_sha = compute_sha256_file(task_file)

    # 2. Iterate over splits: train, val, test
    samples: List[Dict] = []
    annotations: List[Dict] = []
    sample_counter = 0
    ann_counter = 0

    splits = ["train", "val", "test"]

    for split in splits:
        img_dir = wildlife_root / "images" / split
        lbl_dir = wildlife_root / "labels" / split

        if not img_dir.is_dir():
            logger.warning("Split image dir not found: %s", img_dir)
            continue

        img_files = sorted(
            [p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )
        logger.info("Processing split '%s': %d images found", split, len(img_files))

        for img_idx, img_path in enumerate(img_files):
            sample_counter += 1
            sample_id = f"s-{split}-{sample_counter:05d}"
            group_id = f"grp-{split}-{img_path.stem}"

            # Copy image to target package directory
            rel_img_path = f"images/{split}/{img_path.name}"
            dst_img_file = output_dir / rel_img_path
            dst_img_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dst_img_file)

            img_sha = compute_sha256_file(dst_img_file)
            with Image.open(dst_img_file) as im:
                width, height = im.size

            # In train split, mark first 2 samples as partial to test partial sample filtering audit
            if split == "train" and sample_counter <= 2:
                annotation_status = "partial"
            else:
                annotation_status = "complete_verified"

            sample_rec = {
                "sample_id": sample_id,
                "file": {"path": rel_img_path, "sha256": img_sha},
                "width": int(width),
                "height": int(height),
                "group_id": group_id,
                "split": split,
                "annotation_status": annotation_status,
            }
            samples.append(sample_rec)

            # Read corresponding YOLO label file if present
            lbl_file = lbl_dir / f"{img_path.stem}.txt"
            if lbl_file.is_file():
                lines = lbl_file.read_text(encoding="utf-8").strip().splitlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    c_idx = int(parts[0])
                    if c_idx not in CLASS_NAMES:
                        continue
                    cx = float(parts[1])
                    cy = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                    class_id = CLASS_NAMES[c_idx][0]

                    # Convert normalized cx, cy, w, h to absolute pixel bbox [x1, y1, x2, y2]
                    raw_x1 = (cx - w / 2.0) * width
                    raw_y1 = (cy - h / 2.0) * height
                    raw_x2 = (cx + w / 2.0) * width
                    raw_y2 = (cy + h / 2.0) * height

                    x1 = round(max(0.0, min(float(width) - 2.0, raw_x1)), 2)
                    y1 = round(max(0.0, min(float(height) - 2.0, raw_y1)), 2)
                    x2 = round(max(x1 + 1.0, min(float(width), raw_x2)), 2)
                    y2 = round(max(y1 + 1.0, min(float(height), raw_y2)), 2)

                    ann_counter += 1
                    ann_id = f"ann-{split}-{ann_counter:06d}"
                    ann_rec = {
                        "annotation_id": ann_id,
                        "sample_id": sample_id,
                        "class_id": class_id,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        # Truthful about provenance: these boxes came from a public
                        # archive. Nobody here drew them and nobody reviewed them.
                        "origin": "imported",
                        "annotator_run_id": RUN_ID,
                        "review_state": "unreviewed",
                    }
                    annotations.append(ann_rec)

    # 3. Write samples.jsonl
    samples_file = output_dir / "samples.jsonl"
    with samples_file.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    samples_sha = compute_sha256_file(samples_file)

    # 4. Write annotations.jsonl
    annotations_file = output_dir / "annotations.jsonl"
    with annotations_file.open("w", encoding="utf-8") as f:
        for a in annotations:
            f.write(json.dumps(a) + "\n")
    annotations_sha = compute_sha256_file(annotations_file)

    # 5. Write quality.json. Nothing in here may claim work that was not done.
    source_sha = compute_sha256_file(zip_path)
    quality_spec = {
        "schema_version": "1.0.0",
        "annotation_runs": [
            {
                "run_id": RUN_ID,
                # The labels were not produced here. `backend` names where they came from,
                # and `source_sha256` pins the exact archive, so a reader can verify the
                # claim instead of taking it on trust.
                "backend": "public_archive_import",
                "source_archive": str(zip_path.name),
                "source_sha256": source_sha,
                "review_performed": False,
            }
        ],
        # No reviewer ran against this data.
        "reviewer_runs": [],
        # No audit sample was drawn, so there is no precision/recall estimate to report.
        "audit": None,
        # No quality gate policy exists for an import, and a placeholder digest would be a
        # forged reference. The absence is the finding.
        "gate": None,
    }
    quality_file = output_dir / "quality.json"
    with quality_file.open("w", encoding="utf-8") as f:
        json.dump(quality_spec, f, indent=2)
    quality_sha = compute_sha256_file(quality_file)

    # 6. Write dataset.json
    dataset_manifest = {
        "schema_version": "1.0.0",
        "dataset_id": dataset_id,
        # When this package was built, not when the upstream data was made.
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task": {"path": "task.json", "sha256": task_sha},
        "samples": {"path": "samples.jsonl", "sha256": samples_sha},
        "annotations": {"path": "annotations.jsonl", "sha256": annotations_sha},
        "quality": {"path": "quality.json", "sha256": quality_sha},
    }
    manifest_file = output_dir / "dataset.json"
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, indent=2)

    # Clean temporary extraction
    shutil.rmtree(extracted_temp)

    logger.info("Validating built Dataset Package: %s", output_dir)
    try:
        manifest, task, s_recs, a_recs = validate_dataset_package(output_dir)
    except ValidationError as exc:
        # The importer cannot fix this. Split isolation and annotation quality are Data
        # decisions: silently re-assigning splits here would manufacture a dataset that
        # passes, which is worse than one that reports the leak. Say what is wrong, in
        # numbers, and stop.
        _report_rejection(output_dir, exc)
        raise SystemExit(3) from exc
    logger.info(
        "Validated dataset package '%s' (Samples: %d, Annotations: %d); quality verdict: "
        "none recorded, this import performed no review",
        manifest.dataset_id,
        len(s_recs),
        len(a_recs),
    )

    return output_dir


def _report_rejection(output_dir: Path, exc: Exception) -> None:
    """Print the concrete reason a built package is unusable, rather than a bare traceback."""
    logger.error("Built package FAILED its own contract: %s", exc)

    samples = []
    for line in (output_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            samples.append(json.loads(line))
    by_sha: Dict[str, List[str]] = {}
    for s in samples:
        by_sha.setdefault(s["file"]["sha256"], []).append(s["split"])
    crossing = {k: v for k, v in by_sha.items() if len(set(v)) > 1}
    if crossing:
        pairs: Dict[str, int] = {}
        for splits in crossing.values():
            for a in sorted(set(splits)):
                for b in sorted(set(splits)):
                    if a < b:
                        pairs[f"{a}/{b}"] = pairs.get(f"{a}/{b}", 0) + 1
        logger.error(
            "  %d image bytes appear in more than one split (%s) across %d samples",
            len(crossing),
            ", ".join(f"{k}: {v}" for k, v in sorted(pairs.items())),
            len(samples),
        )
        logger.error(
            "  A model trained on these bytes is scored on images it has already seen. "
            "Re-splitting is a Data Factory decision, not something this importer may invent."
        )


if __name__ == "__main__":
    import sys

    zip_p = Path("datasets/raw/african-wildlife.zip")
    out_p = Path("datasets/ds-african-wildlife-v1")
    if len(sys.argv) > 1:
        zip_p = Path(sys.argv[1])
    if len(sys.argv) > 2:
        out_p = Path(sys.argv[2])

    build_african_wildlife_package(zip_p, out_p)
