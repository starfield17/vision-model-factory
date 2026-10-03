# Gate policy documents

`evaluate` and `publish` take a gate policy **path**. There is no built-in policy, no
sample threshold applied when one is missing, and no code path that infers a verdict from
"looks reasonable". If the document is absent, the command refuses.

The documents in this directory are **starting points for a human to edit**, not defaults.
Nothing in `src/` reads this directory; a run only uses a policy that an operator named on
the command line. That is deliberate: a release verdict is a decision about a specific
model, task, and deployment target, and a decision somebody else made once, in a file, is
exactly the kind of thing that stops being questioned.

## What a policy is

| field | meaning |
| --- | --- |
| `policy_id` | Names the decision. Recorded in the report and digested, so changing a threshold produces a different `gate.policy_sha256`. |
| `min_map50`, `min_map50_95` | Floors on the locked test split. `min_map50_95 > min_map50` is rejected: mAP50:95 is strictly harder, so an inverted policy would pass ceilings it cannot clear. |
| `critical_classes[]` | Per-class precision/recall floors for classes where a miss has a specific cost. |
| `require_export_parity_passed` | Whether the exported graph must have demonstrated parity. Defaults to `true` in the model; a policy that sets it `false` is stating that it accepts an unverified artifact. |
| `required_target_profiles[]` | Deployment profiles that must have been **measured** before publication. An unmeasured profile fails the gate rather than being assumed. |

## Editing one

Copy a file, change the numbers, and give it a `policy_id` that says what changed and why.
`policy_id` is not decoration: the published package records the digest of the document that
decided its fate, so reusing an id for different thresholds makes a release untraceable to
the standard it was actually held to.

Set thresholds from the cost of a miss in the deployment, not from what the current model
happens to score. A policy written after seeing a result is a record of that result, not a
standard.
