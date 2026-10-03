# Contract schemas

Two ownership groups live here, and the distinction is load-bearing.

## `data_v1/` — Data-owned, read-only mirror

`task_spec`, `dataset_manifest`, `sample_record`, `annotation_record`, `quality_spec`,
`review_job`, `review_report`, `review_submission`.

These are vendored copies of the wire schemas owned by the Data Factory
(`VisionFactory/app.datasets`), pinned by `data_v1/contract-manifest.json` (SHA-256 per
file). The Model Factory consumes these artifacts; it does not define them.

- Never edit a file in `data_v1/` in this repository. A wire change happens on the Data
  side, ships a new `schema_version`, and arrives here as a refreshed mirror plus an
  updated manifest.
- `tests/test_contracts.py::test_data_v1_mirror_matches_contract_manifest` fails if a
  mirrored file and its recorded digest disagree, so a local "quick fix" to a Data-owned
  schema cannot survive the test run.
- There must never be a second copy of a Data-owned schema outside `data_v1/`. Four such
  shadows existed once (`schemas/task_spec.schema.json` and friends); two sources of
  truth for one wire contract drift, and the drift is invisible until a producer and a
  consumer disagree.

## `*.schema.json` at this level — Model-owned

`experiment_spec`, `run_result`, `model_manifest`, `evaluation_report`. Owned here and
validated by `contracts/schema_validation.py` before any semantic validator runs.
Data-owned contracts are validated from `data_v1/`; the file selection lives in
`schema_validation.SCHEMAS`.
