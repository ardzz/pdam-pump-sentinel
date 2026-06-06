# ADR 0001: Honest Evaluation Split Strategy for Anomaly Detection

* Status: Accepted
* Date: 2026-06-06
* Deciders: PDAM Pump Sentinel team
* Tags: evaluation, anomaly-detection, mlops, methodology

## Context and Problem Statement

PDAM Pump Sentinel uses SKAB as a surrogate dataset for an academic pump predictive maintenance system. SKAB is a controlled water-circulation testbed, not live PDAM data. Its labels come from lab fault injection, with recorded fault windows and categories such as `valve1`, `valve2`, `other`, and `anomaly-free`.

That structure creates a real evaluation choice. The same feature and model stack can look near perfect, merely good, or production-realistic depending on how train and test windows are split.

At one end, a random-window split reports F1 `0.985`. This is not a valid deployment claim because neighboring windows from the same fault episode can land in both train and test. The model can learn episode-local artifacts rather than general fault behavior.

The in-distribution stratified split is the Kaggle-comparable upper bound. It keeps all fault types in both train and test and reports `XGB 0.909 / LGBM 0.905`. This is useful for comparing supervised model capacity, but it assumes labeled examples of every fault type already exist.

That assumption is false for the intended PDAM Day-1 deployment. Operators will begin with mostly normal telemetry, sparse maintenance notes, and no complete labeled inventory of pump failure modes. A supervised in-distribution F1 near `0.91` would therefore overstate what can be deployed first.

At the strict end, cross-group novel-fault evaluation trains on normal-only data, or on labeled fault group A, then tests on fault group B. This gives `0.58` for the unsupervised PCA-spectral champion and `0.60 (AUC 0.937)` for supervised cross-group evaluation. The number is lower, but it matches the operational question: can the system catch a fault pattern it has not already seen with labels?

This ADR decides which split is allowed to support the headline anomaly-detection claim, and how the higher in-distribution numbers may be reported without misleading professors, reviewers, or eventual operators.

## Decision Drivers

* Deployability claim must match operational reality.
* Day-1 PDAM deployment cannot assume labeled examples of every future fault type.
* Industry and academic literature warn that naive evaluation choices inflate time-series anomaly detection scores.
* Point-adjustment and random neighboring windows must not be used to manufacture a better headline number.
* Stakeholders need a clean answer to: "Why not claim the 0.91 XGBoost result?"
* The champion-challenger gate uses F1, so the split behind that F1 must be explicit.
* The presentation should show both model capacity and deployable readiness, not blur them.

## Considered Options

### 1. Random window split

* Result: `F1 0.985`.
* Split semantics: random windows from the same fault episodes can appear in both train and test.
* Pros:
  * Highest headline number.
  * Easy to explain as a generic train/test split.
  * Useful only as a leakage demonstration.
* Cons:
  * Maximum leakage risk.
  * Does not measure novel-fault generalization.
  * Not defensible for production deployment.
  * Would invite a fair objection during Q&A.

### 2. In-distribution stratified, Kaggle-comparable chrono 80/20

* Result: `XGB 0.909 / LGBM 0.905`.
* Split semantics: every fault type is present in both train and test.
* Pros:
  * Good for comparing supervised model families.
  * Shows that XGBoost and LightGBM can learn SKAB fault patterns when labels exist.
  * Comparable to common notebook or competition-style reporting.
* Cons:
  * Requires labeled examples of every fault type.
  * Does not match Day-1 PDAM conditions.
  * Misleading if presented as the deployable anomaly detection score.
  * Can make the PCA-spectral champion look weak for the wrong reason.

### 3. Cross-group novel-fault split

* Result: `supervised 0.60 (AUC 0.937)`, `unsupervised PCA-spectral 0.58`.
* Split semantics: train on normal-only data or one labeled fault group, test on a different fault group.
* Pros:
  * Best match for early PDAM deployment.
  * Tests generalization to fault types without prior labels.
  * Supports the normal-baseline-first roadmap.
  * Makes the PCA-spectral champion defensible as a Day-1 model.
* Cons:
  * Lower and less impressive headline F1.
  * Harder to explain to non-ML audiences.
  * Supervised models lose the benefit of seeing every fault type.
  * Requires slide time to explain split semantics.

### 4. File-level stratified split

* Result: `0.70`.
* Split semantics: files are separated, but fault types are not fully held out by group.
* Pros:
  * Reduces the most obvious window-level leakage.
  * Gives a middle-ground sanity check between random windows and cross-group testing.
  * Easier to run than a strict novel-fault protocol.
* Cons:
  * Still less honest than cross-group for deployment claims.
  * Can mix known fault types across train and test.
  * Ambiguous for a professor asking what happens with unseen fault modes.
  * Not suitable as the single headline number.

## Decision Outcome

Chosen option: **Cross-group novel-fault for the headline number; in-distribution reported only as a clearly labeled upper bound.**

The headline anomaly detection score for the project is the deployable novel-fault result: PCA-spectral champion `F1 0.58`, with supervised cross-group `F1 0.60 (AUC 0.937)` as the strict supervised comparison.

The in-distribution result `XGB 0.909 / LGBM 0.905` may be reported only with its split label and caveat: it requires labeled examples of every fault type in both train and test. It is a capacity upper bound, not the Day-1 deployment claim.

The random-window result `F1 0.985` must not be used as a performance claim. It can appear only as an example of leakage risk.

The file-level stratified result `0.70` may be used as supporting context, but not as the headline. It is more honest than random windows and less strict than cross-group novel-fault.

This choice matches the project story: deploy a normal-baseline anomaly detector first, collect operator feedback later, then let supervised challengers compete when labels become real enough for the champion-challenger gate.

It also gives a defensible Q&A answer. If someone asks why the deck does not headline the `0.91` result, the answer is simple: that number assumes known labeled fault types, while PDAM deployment starts before that inventory exists.

## Consequences

### Positive

* Champion stays PCA-spectral because it can run from Day-1 normal telemetry.
* `F1 0.58` is honestly reported with the cross-group novel-fault label.
* Supervised `0.60 (AUC 0.937)` is still shown as a fair strict comparison.
* The project is defensible against "but in-distribution gets 0.91" questions.
* The evaluation policy aligns with the staged labeling roadmap.

### Negative

* The headline metric is less impressive.
* Extra explanation is needed for non-ML audiences.
* Slide space must be spent on split semantics, not just model names.
* Reviewers may initially compare `0.58` against in-distribution benchmark numbers.

### Neutral

* Both strict and upper-bound numbers are reported in the slide deck.
* Every number must carry its split label.
* Champion promotion can still use F1, but the source split must be recorded with the run.
* Future supervised improvements remain allowed once labeled PDAM data exists.

## Verification

| Claim checked | Evidence |
|---|---|
| SKAB is a controlled water-circulation benchmark with lab fault labels | `docs/presentation/labeling-strategy-notes.md:52-58` |
| PDAM labels will be noisier than SKAB controlled labels | `docs/presentation/labeling-strategy-notes.md:59-70` |
| Academic support covers weak supervision, active learning, pseudo-labels, and TS-AD limits | `docs/presentation/labeling-strategy-notes.md:33-40` |
| Repo already frames the PCA champion as deployable without labeled data | `docs/presentation/labeling-strategy-notes.md:5-8` |
| The deck narrative explicitly distinguishes `0.91` in-distribution from Day-1 deployment | `docs/presentation/labeling-strategy-notes.md:104-110` |
| The honest evaluation spectrum lists all required F1 numbers verbatim | `docs/plans/sprint-remaining.md:89-99` |
| The plan forbids presenting `~0.90` as novel-fault detection | `docs/plans/sprint-remaining.md:101` |
| Champion-challenger promotion is gated by F1 margin and false-alarm guard | `ml/monitoring/champion_challenger.py:8-32` |

## References

* `docs/plans/sprint-remaining.md`: honest evaluation spectrum table and warning against point-adjustment.
* `docs/presentation/labeling-strategy-notes.md`: split semantics, labeling roadmap, and academic support.
* `ml/monitoring/champion_challenger.py:8-32`: `should_promote` uses F1 margin plus false-alarm guard.
* Martínez-Heredia, *WIREs Data Mining and Knowledge Discovery* 2025, weak supervision for predictive maintenance.
* Holtz, *Flexible Services and Manufacturing Journal* 2025, active learning for streaming industrial TS-AD.
* Yoon (SPADE), *TMLR* 2023, pseudo-labeling with one-class detector ensembles.
* Schmidl et al., *PVLDB* 2022, time-series anomaly detection survey.
* Evidently AI, proper test splits for time series: https://www.evidentlyai.com/blog/proper-test-splits-time-series
