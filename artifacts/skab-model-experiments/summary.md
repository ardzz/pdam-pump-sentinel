# SKAB Model Experiment Summary

- Manifest: `/home/reky/college/el-industry/pdam-pump-sentinel/data/skab_split_manifest.json`
- Split protocol: `manifest_train_validation_test_no_point_adjustment`.
- Honest protocol: train-normal fitting, validation-normal calibration, held-out test metrics, no point-adjustment.
- Ranking: completed rows sorted by `test_f1` descending; skipped rows are unranked.
- Deployment framing: `02_pca_spectral` remains the deployment-preferred baseline in this summary; no completed normal-only comparison row reports higher held-out `test_f1` than 0.558331.
- MLflow logging: disabled by default; pass `--log-mlflow` to opt in. Local summaries do not require MLflow.
- Warning: XGBoost/LightGBM are supervised upper-bound experiments only; they require labeled anomaly examples in the training split and must not be described as deployment-safe novel-fault detectors.
- Caution: Train-normal fitting, validation-normal calibration, and held-out test metrics are separate protocol steps.
- Caution: No point-adjustment is applied to thresholded metrics.
- Caution: New unsupervised, calibration, and optional distance-profile rows are comparison evidence, not automatic deployment replacements.
- Expanded row coverage through `12_distance_profile_nearest_normal`:
  - `08_isolation_forest_spectral`: normal-only comparative baseline; Isolation Forest on spectral features, fit on train-normal windows and calibrated on validation-normal windows.
  - `09_oneclass_svm_spectral`: normal-only comparative baseline; One-Class SVM on spectral features, scored as negative decision_function so higher means more abnormal.
  - `10_isolation_forest_conformal`: threshold calibration variant; Conformal wrapper around Isolation Forest scores, not a separately trained model family.
  - `11_forecasting_residual_naive`: normal-only comparative baseline; Naive lag-one residual score with no future rows used for prediction.
  - `12_distance_profile_nearest_normal`: bounded optional comparative baseline; Nearest train-normal window distance with runtime caps and no extra matrix-profile dependency.

| rank | experiment_key | model_family | feature_mode | status | threshold_method | calibration_split | test_f1 | test_precision | test_recall | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 02_pca_spectral | pca | spectral | completed |  |  | 0.5583306581059391 | 0.5487820270099709 | 0.5682174594877156 |  |
| 2 | 10_isolation_forest_conformal | isolation_forest_conformal_threshold | spectral | completed | conformal | validation_normal_rows | 0.5186861313868613 | 0.5874669312169312 | 0.4643230527966545 | Conformal score-threshold wrapper only; source model training is unchanged. |
| 3 | 03_pca_feature_bagging_ensemble | pca_feature_bagging_ensemble | raw_feature_subsets | completed |  |  | 0.5185577007846248 | 0.7968030344080195 | 0.38434396236278096 |  |
| 4 | 08_isolation_forest_spectral | isolation_forest | spectral | completed | validation_normal_quantile | validation_normal_windows | 0.49334060285068926 | 0.6105648737227685 | 0.4138787245164663 | Normal-only train; threshold calibrated on validation-normal windows; held-out test metrics only. |
| 5 | 01_pca_raw | pca | raw | completed |  |  | 0.4897037572254335 | 0.7926900584795321 | 0.3542864610559331 |  |
| 6 | 12_distance_profile_nearest_normal | distance_profile_nearest_normal | raw_window_nearest_normal_distance | completed | validation_normal_quantile | validation_normal_windows | 0.4667339634761862 | 0.7836671802773497 | 0.33233141662310506 | Dependency-light nearest train-normal window distance; reference set is capped for runtime; threshold calibrated on validation-normal windows. |
| 7 | 11_forecasting_residual_naive | forecasting_residual_naive | raw_lag_one_residual | completed | validation_normal_quantile | validation_normal_windows | 0.44308429246224473 | 0.4512806613362244 | 0.4351803450078411 | Naive yhat_t=y_(t-1) residual score; residual scale fit on train-normal windows; threshold calibrated on validation-normal windows. |
| 8 | 04_lstm_ae | lstm_ae | raw | completed |  |  | 0.26039092338800274 | 0.9272 | 0.1514636696288552 |  |
| 9 | 05_lstm_ae_dbscan_threshold | lstm_ae_dbscan_threshold | raw | completed |  |  | 0.22500579911853397 | 1.0 | 0.12676424464192368 | DBSCAN threshold calibrated from validation scores only; fallback is marked in metadata. |
| 10 | 09_oneclass_svm_spectral | oneclass_svm | spectral | completed | validation_normal_quantile | validation_normal_windows | 0.0 | 0.0 | 0.0 | Normal-only train; score=-decision_function; threshold calibrated on validation-normal windows; held-out test metrics only. |
|  | 07_lightgbm | lightgbm | enriched | skipped |  |  |  |  |  | training split must contain both binary classes, got [0]; supervised models require labels in manifest train split |
|  | 06_xgboost | xgboost | enriched | skipped |  |  |  |  |  | training split must contain both binary classes, got [0]; supervised models require labels in manifest train split |
