# ADR 0002: Reject Point-Adjustment Convention for Anomaly Detection Metrics

* Status: Accepted
* Date: 2026-06-06
* Tags: evaluation, anomaly-detection, methodology, point-adjustment

## Context and Problem Statement

Time-series anomaly detection papers often report metrics with the point-adjustment convention.

Point-adjustment means that once a detector flags at least one timestamp inside a true anomaly window, every timestamp in that window is counted as correctly detected.

Under PA, a detector that catches the last sample of a pump fault can receive the same adjusted true-positive credit as a detector that alerted near the first sample.

This convention became common in TSAD benchmark reporting for datasets such as SMD, SMAP, and MSL.

It is easy to compute and gives an event-level story from point labels, but academic literature documents it as a major source of inflated F1 scores.

Kim et al. 2022 show that point-adjustment can produce high adjusted F1 even for random predictions, which makes cross-paper comparison fragile.

PDAM Pump Sentinel needs an evaluation convention that matches operator workflow.

A pump operator asks whether the system raised an alert during the fault episode, how many false alerts happened outside fault episodes, and how long detection took.

SKAB also fits a range-based view of the problem.

The project's labeling notes describe SKAB labels as fault start and end timestamps recorded per injected fault episode, not isolated independent point labels.

That makes the natural evaluation unit a fault window, with detection delay reported separately.

We need a policy that is honest for academic reviewers, clear for future production stakeholders, and stable against metric inflation.

## Decision Drivers

* Match operational reality: an operator wants to detect an anomaly as soon as possible, not merely somewhere inside the window.
* Avoid point-adjustment inflation: Kim et al. 2022, IEEE ICDE, show that point-adjustment can raise random-prediction F1 into the 0.7+ range.
* Keep cross-paper comparison honest by labeling non-PA event metrics clearly.
* Fit SKAB's label structure: SKAB records range-based fault episodes from a controlled water-circulation testbed.
* Preserve operator-facing diagnostics: false alarms and detection delay must stay visible.
* Keep behavior testable through unit contracts, not only report prose.

## Considered Options

### 1. Point-adjustment (PA)

Point-adjustment is the standard convention in many TSAD benchmark papers. If a prediction overlaps any point inside a true anomaly window, the whole window is treated as detected for point-level scoring.

Benefits:

* Easy to compare against older TSAD papers that use SMD, SMAP, or MSL.
* Produces higher headline F1, often closer to published benchmark tables.
* Treats anomaly windows as events rather than isolated timestamps.

Problems:

* Inflates scores by giving full-window credit for one hit.
* Can raise random or adversarial baselines by 20 to 50 percent, with Kim et al. showing adjusted random F1 above 0.7 in some settings.
* Hides detection delay, which is critical in pump monitoring.
* Makes our results harder to defend to reviewers who know the PA critique.

### 2. Point-wise metrics

Point-wise metrics evaluate every timestamp independently. A prediction at time `t` is correct only if the label at time `t` is anomalous.

Benefits:

* Simple, transparent, and familiar from binary classification.
* Does not expand one prediction into a full-window true positive.
* Useful for low-level debugging of thresholds and score distributions.

Problems:

* Penalizes near-miss detections that an operator might still accept.
* Overweights long anomaly windows because they contribute more labeled points.
* Does not answer the operational question, "Did we alert on this fault event?"

### 3. Event-based metrics (chosen)

Event-based metrics treat each contiguous anomaly window as one fault event. A window is detected when at least one prediction overlaps it. Predicted windows outside all true anomaly windows are counted as false-alarm events. Detection delay is reported separately.

Benefits:

* Matches operator workflow: one fault episode should produce one actionable alert.
* Keeps partial detection credit bounded to one event.
* Tracks missed events, false alarms, and delay as separate facts.
* Aligns with SKAB's range-based fault labels.
* Stays stable under random-baseline checks because a single hit does not become full-window point-level success.

Problems:

* Headline F1 is lower than PA-adjusted benchmark numbers.
* Readers need a short explanation before comparing our F1 to older TSAD tables.
* Multiple prediction bursts inside one true event require clear counting rules.

### 4. Volume under surface (VUS-PR / VUS-ROC)

VUS metrics aggregate quality over multiple thresholds and detection ranges. They reduce threshold cherry-picking and can better represent detection quality across a surface of settings.

Benefits:

* Less tied to one threshold.
* Useful for research comparisons across detector families.
* Addresses several limits of single-threshold F1.

Problems:

* Harder for pump operators and project reviewers to interpret.
* Does not directly map to alert counts, missed events, or delay.
* Adds reporting complexity beyond the current MVP scope.

## Decision Outcome

Chosen: event-based metrics for anomaly detection reporting. The project must not use point-adjustment for headline anomaly metrics.

In this checkout, the tested implementation lives in `ml/evaluation/metrics.py`.

The requested `ml/monitoring/event_metrics.py` path is not present in HEAD, so this ADR cites the actual implementation path and function names.

Required reporting convention:

* Use `event_metrics(...)` for event count, event recall, missed events, false-alarm events, and mean detection delay.
* Use `event_precision_recall(...)` and `evaluate_events(...)` for event precision, event recall, and event F1 where an event-level F1 is needed.
* Use point-wise precision and false-alarm rate only when explicitly labeled as point-wise diagnostics.
* Never apply point-adjustment when reporting F1, precision, recall, or model promotion results.

## Consequences

Positive:

* Reproducible cross-paper stance: literature can report both PA and non-PA, but this project reports non-PA event-based results.
* Matches operator expectation: one alert per fault window, with delay shown separately.
* Stable under random-baseline tests because a random hit does not expand into full-window point-level success.
* Honest for academic review: the method makes the PA inflation issue explicit rather than borrowing inflated benchmark conventions.
* Fits SKAB's labels: range-based injected fault windows are evaluated as fault events.

Negative:

* Headline F1 numbers will be lower than many TSAD papers that report PA scores.
* The report and presentation need to educate readers about why PA is rejected.
* Direct comparison to older benchmark tables requires care and clear metric labels.

Neutral:

* Detection delay is reported separately, not folded into F1.
* Threshold-free PR-AUC and ROC-AUC may still be reported as supplementary diagnostics, but they do not replace event metrics.
* Point-wise metrics remain useful for debugging, as long as they are labeled as point-wise and not presented as event-level success.

## Verification

Repository evidence:

* `ml/evaluation/metrics.py:62-79` defines `contiguous_ranges(...)`, which turns binary labels into half-open anomaly windows.
* `ml/evaluation/metrics.py:82-109` defines `event_metrics(...)`, counts true event windows, detected events, missed events, false-alarm events, and mean detection delay without point-adjusting predictions.
* `ml/evaluation/metrics.py:112-129` defines `composite_f_score(...)`, combining point precision with event recall instead of PA-expanded point recall.
* `ml/evaluation/metrics.py:132-150` defines `event_precision_recall(...)` and `evaluate_events(...)`, the event precision, recall, and F1 contract.
* `ml/evaluation/metrics.py:238-253` defines overlap helpers that match ranges by interval overlap, not by expanding predictions across a true window.

Test evidence:

* `tests/unit/test_event_metrics_contract.py:8-29` checks that partial detection of one anomaly segment gives event recall `0.5`, not full adjusted success.
* `tests/unit/test_event_metrics_contract.py:32-47` checks undefined recall when there are no ground-truth anomaly events.
* `tests/unit/test_event_metrics_contract.py:49-64` checks that multiple hits inside one all-anomaly window still count as one detected event.
* `tests/unit/test_event_metrics_contract.py:66-80` checks no-prediction behavior and keeps event precision undefined when there are no predicted events.
* `tests/unit/test_event_metrics_contract.py:83-97` checks perfect predictions score `1.0` across the event contracts.
* `tests/unit/test_skab_metrics_contract.py:45-73` checks half-open ranges, missed events, false-alarm events, and detection delay on SKAB-shaped labels.
* `docs/presentation/labeling-strategy-notes.md:54-59` records SKAB as a water-circulation testbed with range-based fault start and end labels.
* `docs/plans/sprint-remaining.md:89-101` states the honest evaluation spectrum and explicitly says to never use point-adjustment.

## References

* `ml/evaluation/metrics.py`, current event-based metric implementation.
* `tests/unit/test_event_metrics_contract.py`, event metric contract tests.
* `tests/unit/test_skab_metrics_contract.py`, SKAB-shaped metric contract tests.
* Kim et al. 2022, "Towards a Rigorous Evaluation of Time-Series Anomaly Detection", IEEE ICDE, https://arxiv.org/abs/2109.05257.
* Schmidl et al. 2022, "Anomaly Detection in Time Series: A Comprehensive Evaluation", PVLDB, survey and evaluation discussion including PA controversy.
* Doshi et al. 2022, "Reward Once, Penalize Once: Rectifying Time Series Anomaly Detection", https://arxiv.org/abs/2204.11718.
