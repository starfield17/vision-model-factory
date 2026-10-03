# Escalations out of this repository

Findings that belong to another owner, recorded with enough detail to act on. Nothing here
is worked around in Model code.

---

## E-1 · `ds-african-wildlife-v1` leaks identical image bytes across train/val/test

**Owner:** Vision Data Factory. **Status:** open, blocking any honest release on this dataset.
**Affects:** `releases/model-african-wildlife-yolo26n-001` (already published; invalid).

`datasets/ds-african-wildlife-v1` (1504 samples) contains image files whose SHA-256 is
identical across splits:

| crossing pair | count |
| --- | --- |
| test ↔ train | 12 |
| train ↔ val | 7 |
| test ↔ val | 1 |
| **total distinct byte-hashes crossing splits** | **20** |

Consequence: a model trained on `train` has literally seen 12 of the images it is scored on
in `test`. mAP50 / mAP50:95 reported for this dataset is not a measurement of generalization.
The `group_id` isolation check passes — the duplicated files carry different `sample_id`s and
different `group_id`s — so **the leakage is invisible to group-level deduplication and only
the byte-hash rule catches it.** That is worth knowing for whatever rule Data Factory adopts.

### Reproduce

```bash
PYTHONPATH=src python scripts/build_dataset_package.py \
  datasets/raw/african-wildlife.zip datasets/_import_check
# exits 3 and prints the table above; the importer rebuilds the package from the same
# archive and re-validates it, so the numbers are reproducible from scratch.
```

or against the existing package:

```bash
PYTHONPATH=src python -m vision_model_factory.cli validate-dataset datasets/ds-african-wildlife-v1
# ValidationError: Identical image bytes cross splits
```

### What is requested

1. A new dataset release in which no image byte-hash appears in more than one split,
   deduplicated before assignment rather than after.
2. Population of the `audit` split. The Data vocabulary already allows
   `split: "audit"` and the Model side accepts it, but this package has zero audit samples
   (`train` 1052, `val` 225, `test` 227, `audit` 0). `audit` is the corpus export parity is
   meant to use, because parity must not spend the locked `test` split and `val` is
   training-visible in this pipeline.
3. A quality verdict that reflects real review. The package currently asserts
   `human_verified` per annotation and `precision: 1.0 / recall: 1.0` with
   `gate.policy_sha256 = "0"*64`. `scripts/build_dataset_package.py` no longer writes that —
   it now records `origin: "imported"`, `review_state: "unreviewed"`, `audit: null`,
   `gate: null` — but the released package on disk still carries the older claims.

### What will not happen in this repository

Re-assigning splits, deleting the duplicated files, or editing `quality.json` to make the
package pass. Under N4 those are Data Factory decisions; a Model-side patch would manufacture
a dataset that validates while hiding the leak, and the next mAP number would be no more
trustworthy than the one it replaced.

**Impact on this repository:** the pipeline is verified end-to-end on synthetic packages
(`tests/test_cli.py::TestPipelineScenario`), but no real release can be regenerated until a
sound dataset exists. The published release 001 stays as-is, and
`python -m vision_model_factory.cli validate-model releases/model-african-wildlife-yolo26n-001`
fails against the current contract (`extensions: None is not of type 'object'`) — which is the
correct outcome for an artifact that should not have been published.
