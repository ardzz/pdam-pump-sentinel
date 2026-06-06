# Champion Challenger Promotion Gate

## Status

Accepted.

## Context and Problem Statement

PDAM Pump Sentinel runs a continuous training path that can produce a challenger model after scheduled retraining or drift triggered retraining.
The project needs a quantitative promotion gate because human review for every retraining run doesn't fit the intended MLOps loop.
Manual review is useful for unusual releases, but routine retrains need a deterministic decision that can run inside the pipeline.
Blind automatic promotion is also unsafe.
If a worse challenger becomes champion, operators may see more missed anomalies, more false alarms, or both.
That is a trust problem, not just a metric problem.
Pump operators will stop trusting a sentinel that promotes a noisier model after claiming the retraining process improved it.
The gate therefore has to protect the current champion while still letting better challengers move through MLflow registry workflows.
`docs/presentation/labeling-strategy-notes.md:76` already frames the gate as a project proof point and cites `ml/monitoring/champion_challenger.py:8-32`.
This record turns that implementation into an architecture decision.

## Decision Drivers

1. Automatic promotion safety must block clear regressions without waiting for a reviewer.
2. MLOps automation must support retraining, registry updates, alias promotion, and hot swap workflows.
3. False alarm sensitivity matters as much as F1 because avoidable alarms add operator burden.
4. The rule must be deterministic and auditable.
5. The rule must be explainable in a demo and during model governance review.
6. The MVP should avoid a metric policy that needs long calibration before use.
7. The decision must match the current code contract and tests.

## Considered Options

### Option 1. F1 only gate with margin 0.01 to 0.05

Promote the challenger when its F1 is higher than champion F1 by a fixed margin.
Pros: simple, common, and needs only one metric.
Cons: it can promote a model that improves F1 by creating many more false alarms.
This is too weak for an operator dashboard where alarm noise can reduce trust.

### Option 2. AUC only gate

Promote the challenger when AUC improves over the champion.
Pros: threshold independent and useful during model research.
Cons: AUC doesn't directly describe the deployed alarm threshold.
It can look good while the selected operating point creates too many false positives.
It is also harder to explain when operators ask why the production alarm model changed.

### Option 3. F1 plus FAR guard

Promote only when both conditions hold:

```text
f1_challenger > f1_champion + 0.02 AND far_challenger <= far_champion * 1.05
```

Pros: keeps F1 as the main quality signal, then blocks extra false alarm burden.
Pros: matches the current implementation and the current unit tests.
Cons: still depends on validation sample quality.
Cons: doesn't represent every possible tradeoff across precision, recall, latency, and stability.

### Option 4. Manual review of every retrain

Require a human reviewer before any challenger can become champion.
Pros: catches context that metrics can miss and can include operator feedback.
Cons: high overhead, slow promotion, and less predictable retraining outcomes.
This is better as an exception path than as the default for every retraining run.

### Option 5. Multi metric Pareto gate

Compare the challenger across several metrics and promote only if it is not dominated by the champion.
Pros: more complete than a two metric policy.
Cons: too complex for the MVP and harder to audit quickly.
It should wait until the project has enough production data to calibrate metric priorities.

## Decision Outcome

Chosen option: Option 3, F1 plus FAR guard.

The promotion rule is:

```text
f1_challenger > f1_champion + 0.02 AND far_challenger <= far_champion * 1.05
```

The F1 margin guard is `0.02`.
Prior literature and the SKAB cross group split calibration treat movement around `0.01` F1 as noise, so `0.02` gives a small but meaningful buffer.
The false alarm ratio guard is `1.05`.
That means the challenger may not degrade false alarm rate by more than 5 percent relative to the champion.
The code contract is the signature in `ml/monitoring/champion_challenger.py:8-13`:

```python
def should_promote(
    champion_metrics: Mapping[str, float | None],
    challenger_metrics: Mapping[str, float | None],
    f1_margin: float = 0.02,
    far_guard: float = 1.05,
) -> tuple[bool, str]:
```

The implemented comparison is in `ml/monitoring/champion_challenger.py:24-32`.
It first requires challenger F1 to exceed champion F1 plus the margin.
It then requires challenger `false_alarm_rate` to stay within `champion_far * far_guard`.
If there is no valid incumbent champion, a challenger with valid metrics can become the first champion.

## Consequences

1. Promotion decisions become predictable.
2. The retraining job can decide without waiting for a manual reviewer.
3. MLflow can record runs, metrics, params, tags, artifacts, and alias changes for later audit.
4. The rule ties model quality to detection quality and alarm burden.
5. A challenger with higher F1 but a large false alarm increase is rejected.
6. A challenger with acceptable false alarms but weak F1 gain is rejected.
7. The policy is mostly defensible for the MVP.
8. The policy can be flaky on small validation sets.
9. A small validation sample can trigger spurious promotion when a few windows make F1 look better while hiding uncertainty.
10. The mitigation is to require a minimum validation sample count before `should_promote` is allowed to decide.
11. MLflow run metadata helps audit this failure mode through sample count tags, dataset split tags, metrics, run inputs, and alias update history.
12. The gate doesn't replace later human review for high impact releases.
13. It gives the project a safe default for routine retraining while keeping future policy changes possible.

## Verification

The rule is exercised by `tests/unit/test_ml_monitoring_contract.py:49-62`.
Two verbatim cases from the parametrized test are:

```python
({'f1': 0.8, 'false_alarm_rate': 0.1}, {'f1': 0.83, 'false_alarm_rate': 0.1}, True, 'beats f1'),
({'f1': 0.8, 'false_alarm_rate': 0.1}, {'f1': 0.85, 'false_alarm_rate': 0.2}, False, 'exceeds guard'),
```

The first case proves that a challenger with enough F1 gain and stable FAR promotes.
The second case proves that a challenger with better F1 but too much false alarm degradation is blocked.
The same test calls the implementation through `promoted, reason = should_promote(champion, challenger)` at `tests/unit/test_ml_monitoring_contract.py:58-59`.
The implementation under test is the `should_promote` function exported from `ml.monitoring.champion_challenger`, imported at `tests/unit/test_ml_monitoring_contract.py:15`.

## References

1. `ml/monitoring/champion_challenger.py:8-13`, `should_promote` signature.
2. `ml/monitoring/champion_challenger.py:24-32`, F1 margin and false alarm guard implementation.
3. `tests/unit/test_ml_monitoring_contract.py:49-62`, parametrized promotion contract tests.
4. `docs/presentation/labeling-strategy-notes.md:76`, presentation proof point for the champion challenger gate.
5. MLflow Model Registry alias workflow documentation, especially alias based model selection and alias update flows.
