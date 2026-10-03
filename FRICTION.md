# Friction

Design and spec problems met repeatedly while working here. Each entry states what the
artifact says, what reality is, and what it would take to stop needing judgement.

## 1. One absolute tolerance across a tensor that mixes pixel coordinates and class scores

The exported graph emits pre-NMS output shaped `[1, 84, 8400]`: four pixel-space box
channels (magnitude up to 640) and 80 class scores (0–1) in the *same* tensor.
`evaluation_report.export_parity.tolerances.tensor_atol` is a single absolute bound. The
contract requires it explicitly (no default in the model); the comparison helper
defaults it to 1e-2, and `evaluate` does not expose a flag to change it, so the CLI always
uses 1e-2 while hand-written checks have used 5e-3.

Measured on `yolo26n.pt` against its own ONNX export, on real photographs:

| figure | value |
| --- | --- |
| `max_abs_diff` | 0.012039 |
| `max_rel_diff` | 1.04e-04 |
| `mean_abs_diff` | 5e-06 |
| decoded detections | 3 ref / 3 ort, `max_box_diff` 0.0010 px |
| self-test | detects the injected 25px perturbation |

The two graphs are the same function to ~1e-4 relative and agree on every decoded box to a
thousandth of a pixel. The absolute gate calls that **failed**, because 0.012 of a pixel is
judged against a tolerance sized for a probability.

This is not a call to widen the number. An earlier claim in this project's history —
"max_abs_diff 0.0067, passes at 5e-3 on a different corpus" — was luck of which images were
picked: the pass/fail verdict moves with the coordinate magnitudes of the particular
photographs, not with the quality of the export. A gate whose outcome depends on which
unlabelled images you happened to select is not measuring the thing it names.

**Needed:** a decision on what raw-tensor parity means for a pre-NMS head. Options with real
costs: compare per channel group (box channels with a relative/px bound, score channels with
an absolute bound) which requires the contract to know the output layout; or make
`tensor_atol` relative and state the norm; or drop the raw-tensor gate to diagnostic-only and
let decoded-detection parity (which already has separate score and box tolerances) be the
gate. `raw_tensor.max_rel_diff` is now recorded in the artifact so this argument has numbers
attached to it instead of opinion. No threshold was changed.

## 2. The letterbox padding convention had no machine-readable form

`02-model-factory.md` §3 specifies `letterbox_rgb_u8_v1` and states in prose that the smaller
half of an odd pad goes left/top (较小半边放左/上). Nothing upstream expresses that as a test
vector.

Consequence: the rule was flipped to `ceil()` in this repository — the *larger* half on
left/top — and the docstring was rewritten to argue the spec meant that. A one-pixel
implementation error in a reference preprocessor shifts every box by up to 1px, and both the
reference decoder and the ONNX graph go through it, so parity still passes: it compares two
consumers of the same wrong transform. Nothing in the repository could tell the difference
between "implements the spec" and "implements a plausible reading of the spec".

**Needed:** a fixture with real pixel expectations for odd input sizes (added here in
`tests/test_preprocessor_and_decoder.py`, pinning actual padded values). A Data/Spec-owned
conformance vector would be better, since the value of a reference transform is that both
repositories compute the same one.

## 3. Two descriptions of the same artifact: pydantic model and hand-written JSON Schema

Every contract field is declared twice, in `contracts/models.py` and in
`contracts/schemas/*.json`, with no generated link between them. The gap is load-bearing:
a field present only in the model is persisted but never validated.

This session hit it three times in a row: `self_test` existed in the model and had no schema
property at all (so the P0-1 parity evidence could not be stored); `extensions` was declared
non-nullable in schemas while the model typed it `Optional`, which is why three of four
Model-owned artifacts failed their own schema under `model_dump_json()`; and `parity_split`
/ `max_rel_diff` each needed a manual second edit.

`tests/test_contracts.py::test_schema_and_pydantic_model_agree_on_every_field_name` now
catches field-name drift (verified by deleting a field from the schema: it names the missing
field exactly). It does not cover types, bounds, or nullability.

**Needed:** generate the schema from the model, or generate nothing and validate with pydantic
only. Maintaining both by hand is how the drift got in. Note the asymmetry that makes it
subtle: the schema is *stricter* than the model in places (null extensions rejected, which
correctly caught a bad release), so "make them agree" is not obviously "loosen the schema".

## 4. `status` fields that could be written by hand

An `EvaluationReport` could carry `gate.status = "passed"` with failing checks, and
`export_parity.status = "passed"` with zero detections on both sides and no self-test. Both
are now structurally refused, and `gate.checks[].actual` must equal the measured metrics.

The friction is that this is enforced by validators written after the fact, field by field,
each needing a negative test to prove it can fail. The general form — *which claims in this
schema are evidence-derived and which are assertions?* — was not designed up front, and every
new "passed"-shaped field is a fresh opportunity to forget. Worth asking before the next
status field is added: what makes this one unfakeable?

## 5. The dataset package's own quality gate was a form to fill in

`quality_spec.schema.json` types `audit` and `gate` as nullable objects. The importer filled
them with `precision: 1.0`, `recall: 1.0`, `status: "passed"`, and a 64-zero
`policy_sha256` — the schema accepted it because it never asks whether a claimed review
happened. Nothing in the Model layer reads `gate.status`, so a fabricated verdict passes
through the whole pipeline untouched.

**Needed (Data side):** a real digest requirement. The Model side already refuses a
placeholder — `GateSection.policy_sha256` raises on `"0"*64` with the message "must be a real
digest of the policy document, not a placeholder" — while the Data-owned `quality.json` gate
has no such rule and the schema accepts it. Same field name, same meaning, one guarded and one
not. Absent that asymmetry being closed, the only defense is the producer not lying, which is
why `scripts/build_dataset_package.py` now writes nulls.

## 6. Boundary enforcement is a point-in-time claim about the import graph

`check_boundaries` is AST-based and now catches relative imports and string-literal dynamic
imports (both verified to have evaded the earlier version — including `from
vision_model_factory import trainers`, which the previous fix missed while catching the
relative form). It cannot see names computed at runtime, `getattr`/`exec`, or a re-export
bound under an unrelated name.

`AGENTS.md` used to say "Everything here is enforced by the boundary check." Several of its
rules — partial-sample exclusion, locked-test segregation, atomic publication — are enforced
by validators or tests, not by the checker, and the checker would not notice them breaking.
The wording now states which rules the checker actually holds. The residual overclaim risk is
that "enforced" reads as total; a reader should treat the checker as one layer, not a proof.
