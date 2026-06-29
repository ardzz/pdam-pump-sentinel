from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image as PILImage
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUTLINE_PATH = ROOT / 'docs/presentation/deck-outline.md'
OUTPUT_PATH = ROOT / 'docs/presentation/pdam-pump-sentinel-deck.pptx'
SCREENSHOT_DIR = ROOT / 'docs/presentation/screenshots'
ASSET_DIR = ROOT / 'docs/presentation/assets/generated'

CANONICAL_SCREENSHOTS = [
    't9-observability-streamlit-observability-snapshot-20260628T0535Z.png',
    't9-observability-grafana-pipeline-observability-20260609T130723Z.png',
]

PRIMARY = RGBColor(42, 42, 115)
INK = RGBColor(28, 30, 56)
PAGE = RGBColor(255, 255, 255)
PANEL = RGBColor(238, 240, 248)
TEAL = RGBColor(0, 128, 124)
CYAN = RGBColor(30, 110, 160)
WHITE = RGBColor(255, 255, 255)
MUTED = RGBColor(108, 116, 140)
AMBER = RGBColor(230, 150, 30)
RED = RGBColor(208, 64, 80)
GREEN = RGBColor(40, 150, 96)


def _ink_for(fill: RGBColor) -> RGBColor:
    luminance = (0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2])
    return INK if luminance > 150 else WHITE


@dataclass(frozen=True)
class SlideSpec:
    title: str
    kicker: str
    bullets: tuple[str, ...]
    kind: str = 'bullets'
    screenshot: str | None = None
    screenshots: tuple[str, ...] = ()
    image: str | None = None
    track: str = ''
    source: str = ''
    table: tuple[tuple[str, ...], ...] = ()
    caption: str = ''


SLIDES: tuple[SlideSpec, ...] = (
    SlideSpec(
        title='PDAM Pump Sentinel',
        kicker='Multivariate anomaly detection on pump sensors — SKAB surrogate, PCA T2/Q champion.',
        bullets=(
            'End-to-end anomaly-detection slice: sensors, multivariate detection, model lifecycle, operator dashboard.',
            'SKAB water-circulation benchmark is used as surrogate academic data, not real PDAM field data.',
            'Focus of this talk: how the system detects faults, and the honest evidence behind it.',
        ),
        kind='title',
        image='slide-overview-pipeline.png',
        source='README.md; docs/plans/design.md',
    ),
    SlideSpec(
        title='Why This Problem Matters',
        kicker='Pump faults are multivariate and gradual, and single-sensor limits miss them.',
        bullets=(
            'Centrifugal pumps are critical assets in water distribution operations.',
            'Faults show up as cross-sensor pattern shifts, not one value crossing a line.',
            'Goal: early multivariate condition monitoring, not just post-failure alarms.',
        ),
        image='slide-problem-threshold.png',
        source='docs/plans/design.md §2',
    ),
    SlideSpec(
        title='Anomaly 1: Point Anomaly',
        kicker='A single sensor reading spikes or drops sharply for an instant.',
        bullets=(
            'A momentary spike or drop in one sensor (general anomaly taxonomy, illustrative).',
            'Min/max only catches it when the value crosses the static limit.',
            'A short spike that stays inside the band still slips through.',
        ),
        kind='figure',
        image='problem-anomaly-point.png',
        source='docs/plans/design.md §2 (taxonomy is illustrative, not SKAB-defined)',
    ),
    SlideSpec(
        title='Anomaly 2: Context Anomaly / Change Point',
        kicker='Normal in another context, but abnormal for the current regime.',
        bullets=(
            'The level shifts to a new baseline without ever touching the static limit.',
            'Min/max has no context memory, so this drift slips through.',
            'SKAB labels separate outliers from change points and collective anomalies.',
        ),
        kind='figure',
        image='problem-anomaly-context.png',
        source='docs/plans/design.md §2; ml/features/windowing.py:54-59',
    ),
    SlideSpec(
        title='Anomaly 3: Pattern / Subsequence Anomaly',
        kicker='A run of points forms an odd shape, for example a stuck or flatline segment.',
        bullets=(
            'A frozen or repeating segment looks calm to min/max (illustrative taxonomy).',
            'What breaks is the time-series shape, not the amplitude.',
            'The 60-sample window makes this short episode visible to the model.',
        ),
        kind='figure',
        image='problem-anomaly-pattern.png',
        source='docs/plans/design.md §2; ml/features/windowing.py:54-59',
    ),
    SlideSpec(
        title='Anomaly 4: Multivariate Relationship Anomaly',
        kicker='Each channel looks fine, but the relationship between sensors breaks.',
        bullets=(
            'Real example: Accelerometer1 vs Accelerometer2 correlation goes from -0.45 (normal baseline) to +0.997 over the fault-file test split.',
            'Per-sensor thresholds never see the relationship across sensors.',
            'PCA T2/Q learns the joint structure of normal sensors and flags the break.',
        ),
        kind='figure',
        image='problem-anomaly-multivariate.png',
        source='train/test correlation_matrix.csv; belajar-pca-t2-q-spectral-manual.md:58-67',
    ),
    SlideSpec(
        title='Why Min/Max Alone Is Not Enough',
        kicker='A static threshold only checks one sensor against one fixed limit.',
        bullets=(
            'A spike that crosses the limit is indeed caught.',
            'A context shift rises to a new regime yet stays inside the min/max band, so it is missed.',
            'Three of the four anomaly families are out of reach for a static limit.',
        ),
        kind='figure',
        image='problem-minmax-limit.png',
        source='docs/plans/design.md §2',
    ),
    SlideSpec(
        title='EDA: What the Data Is',
        kicker='Structural facts about SKAB before any modeling.',
        bullets=(
            '8 sensors at ~1 Hz: 2 accelerometers (vibration), current, pressure, temperature, thermocouple, voltage, volume flow. Missing values: 0%.',
            'No physical units are published with the dataset, so sensors are named by role.',
            'Honest split (per file): train 1 file (anomaly-free, 9405 rows, all normal), validation 16 valve1 files, test 18 valve2+other files.',
            'Windowed at 60 samples / stride 1: train 9346 normal windows; validation 9963 normal / 7253 anomaly; test 10527 / 7652.',
            'The test files are concatenated, so the timeline is not one continuous 1 Hz stream.',
        ),
        kind='detail',
        track='EDA',
        image='slide-data-sensors.png',
        source='skab_loader.py:6-15; data/skab_split_manifest.json; 02_pca_spectral/metrics.json',
    ),
    SlideSpec(
        title='EDA: Sensors Move Together',
        kicker='Sensors are tightly coupled in normal operation; the joint structure shifts under fault.',
        bullets=(
            'Normal baseline shows strong couplings, e.g. Temperature-Thermocouple -0.89 and Thermocouple-Volume Flow +0.83; under fault these change sign or strength.',
            'That coupling is exactly what PCA learns as normal, and what Q/SPE flags when it breaks.',
            'Per-sensor ranges also drift in fault, e.g. Volume Flow p01 121 to 15, Thermocouple max 29.5 to 33.4.',
        ),
        kind='figure',
        image='eda-correlation-heatmap.png',
        source='train/test correlation_matrix.csv; sensor_distributions.csv',
    ),
    SlideSpec(
        title='Candidate Models: Supervised vs Unsupervised',
        kicker='Same dataset, same split — the difference is what each model sees in training.',
        bullets=(
            'Unsupervised — train on normal-only windows, can flag novel faults: PCA T2/Q (spectral champion, raw, feature-bagging), Isolation Forest (+ conformal), One-Class SVM, LSTM autoencoder (+ DBSCAN), plus forecasting-residual and distance-profile baselines.',
            'Supervised — need labeled normal + fault rows, only recognize trained fault types: XGBoost and LightGBM.',
            'Both families emit an identical AnomalyVerdict (anomaly 0/1, score, t2, q); the pipeline cannot tell which model produced it.',
            'Honest Day-1 split has a normal-only train file, so XGBoost/LightGBM are SKIPPED (single class [0]); unsupervised is the only family that can train.',
        ),
        kind='detail',
        track='Modeling',
        image='model-candidates-training-split.png',
        source='ml/inference/pca_inference.py:22-35; ml/inference/supervised_inference.py:29-42; summary.md:31-33',
    ),
    SlideSpec(
        title='Model Inputs: Rows to Windows to Features',
        kicker='One shared contract feeds every candidate model.',
        bullets=(
            'Shared input: 60-sample windows (stride 1), raw or spectral FFT features — identical for every candidate.',
            'What differs is data usage: supervised trains on labeled normal + fault rows; unsupervised (PCA champion, RobustScaler) trains on normal-only windows.',
        ),
        kind='figure',
        track='Modeling',
        image='supervised-vs-unsupervised-data.png',
        source='ml/features/windowing.py:20-61; ml/features/spectral.py:53-82; ml/training/pca_detector.py:30-35',
    ),
    SlideSpec(
        title='Modeling Results: Honest F1 Spectrum',
        kicker='The same model looks very different depending on split honesty.',
        bullets=(),
        kind='table',
        track='Modeling',
        table=(
            ('Model', 'F1', 'Family / note'),
            ('02_pca_spectral', '0.558', 'CHAMPION (deployed) — P 0.549 / R 0.568 — T2 1.0776, Q 6139.43'),
            ('IsoForest-conformal', '0.519', 'unsupervised'),
            ('PCA-bagging', '0.519', 'unsupervised'),
            ('IsoForest-spectral', '0.493', 'unsupervised'),
            ('PCA-raw', '0.490', 'unsupervised'),
            ('distance-profile', '0.467', 'unsupervised baseline'),
            ('forecasting-naive', '0.443', 'baseline'),
            ('LSTM-AE', '0.260', 'unsupervised (deep)'),
            ('LSTM-DBSCAN', '0.225', 'unsupervised (deep)'),
            ('OneClassSVM', '0.000', 'unsupervised (collapsed)'),
        ),
        caption='Context only — NOT the deployed model: supervised cross-group ~0.60 (AUC 0.937), in-distribution XGB 0.909 / LGBM 0.905, random-window 0.985 = leakage. Never present an in-distribution number as novel-fault readiness.',
        source='summary.md:22-33; docs/adr/0001-honest-eval-split-strategy.md; sprint-remaining.md',
    ),
    SlideSpec(
        title='Why PCA-Spectral as Day-1 Champion',
        kicker='Chosen for a continuous-learning loop: cheap, deterministic retraining, not just F1.',
        bullets=(
            'Industrial predictive maintenance retrains on a cadence (scheduled ~30 min + drift-triggered, PCA path), so the detector must be cheap and deterministic.',
            'PCA retrains in a single full-SVD pass with closed-form thresholds: no epochs, no gradient descent, no hyperparameter search, reproducible for the champion-challenger gate.',
            'It also deploys before a labeled fault inventory exists (normal-only cold start), where supervised XGBoost/LightGBM cannot even train.',
            'Fast, decomposable, and operator-explainable; the mathematical proof of its low cost is on the next slide.',
        ),
        kind='detail',
        track='Modeling',
        image='slide-pca-scatter.png',
        source='labeling-strategy-notes.md:5-8; docs/adr/0001-honest-eval-split-strategy.md:16-20; ml/monitoring/scheduler.py',
    ),
    SlideSpec(
        title='PCA Mechanics',
        kicker='One full SVD trains it; everything after is closed-form.',
        bullets=(
            'Train once: fit RobustScaler, then a single full SVD on the normal windows (n=9346 rows, d=64 spectral features); keep components for 90% variance.',
            'Training cost is that SVD: O(n*d^2 + d^3); thresholds are p95 empirical quantiles (one O(n log n) sort). No epochs, no backprop, no hyperparameter search.',
            'Inference per window = project then reconstruct = O(d*k), where k is the kept components; deterministic and CPU-only.',
            'Contrast: LSTM-AE trains ~100 epochs of backprop per retrain; PCA trains in one SVD, so retraining on cadence stays cheap.',
        ),
        kind='detail',
        track='Modeling',
        image='pca-vs-lstm-retrain-cost.png',
        source='ml/training/pca_detector.py:42-64; ml/training/train_lstm_ae.py:38; belajar-pca-t2-q-spectral-manual.md:103-117',
    ),
    SlideSpec(
        title='T2 vs Q/SPE',
        kicker='T2 = extreme along the normal road; Q/SPE = behavior that leaves the road.',
        bullets=(
            'T2 = sum(score^2 / explained_variance): extreme, but still on a pattern PCA already knows.',
            'Q/SPE = squared reconstruction residual: the relationship itself changed.',
            'Decision is OR: anomaly if T2 > threshold OR Q > threshold; thresholds are p95 quantiles of normal.',
            'Reported score = max(T2/T2_thr, Q/Q_thr); the top sensor is the largest residual contributor, not a proven root cause.',
            'Both statistics reuse one projection, so scoring a window is O(d*k): live inference adds negligible cost at the ~1 Hz cadence.',
        ),
        kind='detail',
        track='Modeling',
        image='pca-reconstruct-residual.png',
        source='ml/training/pca_detector.py:55-114; ml/inference/pca_inference.py:186-267',
    ),
    SlideSpec(
        title='Spectral FFT Features',
        kicker='Spectral lets PCA see rhythm, not just level.',
        bullets=(
            'Each window is mean-centered per sensor, then summarized by mean, std, range, FFT band energies, and spectral centroid.',
            'Bands are equal splits of the FFT energy bins (not named Hz bands); feature count = sensors x (n_bands + 4) = 64.',
            'Dominant frequency is taught as intuition only; it is not a live feature column.',
            'Intuition: a rhythm shift can move band energy before the raw level moves (illustrative, not a measured lead time).',
            'Feature build is cheap: the per-window FFT is O(d log d) per channel, so spectral awareness does not change the low-cost profile.',
        ),
        kind='detail',
        track='Modeling',
        image='pca-eigen-scree.png',
        source='ml/features/spectral.py:53-82; ml/inference/pca_inference.py:270-284',
    ),
    SlideSpec(
        title='What Is MLOps, and Why It Helps Us',
        kicker='Operationalizing the model lifecycle: train, deploy, monitor, retrain.',
        bullets=(
            'MLOps keeps a model useful after the notebook: versioning, monitoring, drift, and safe updates.',
            'In our case: deploy a champion on Day-1, watch for drift, retrain, and promote without downtime.',
            'Tools: MLflow (registry), Evidently (drift), Prometheus + Grafana (signals), Streamlit (operator).',
            'Operator actions (ack/mute/note) are captured as evidence, not yet as training labels.',
        ),
        kind='detail',
        track='MLOps',
        source='README.md:13-15,39-42',
    ),
    SlideSpec(
        title='MLflow: Registry + Custom Promotion Gate',
        kicker='One managed registry, plus a custom gate that decides promotions.',
        bullets=(
            'MLflow Model Registry: versioned models with a @champion alias; the app loads the champion and hot-swaps in-process.',
            'A custom champion-challenger gate (project code, not an MLflow feature) makes the decision; MLflow then only moves the alias.',
            'Gate: promote iff F1_challenger > F1_champion + 0.02 AND FAR_challenger <= FAR_champion x 1.05.',
            'Schedulers (drift 15m, retrain 30m, refresh 5m) are off by default; hot-swap is process-local.',
        ),
        kind='detail',
        track='MLOps',
        image='mlops-evidence-loop.png',
        source='ml/monitoring/champion_challenger.py:8-32; docs/adr/0003-champion-challenger-promotion-gate.md:78-87; bootstrap/app.py:199-204',
    ),
    SlideSpec(
        title='RouteMQ: Connect It All Together',
        kicker='One MQTT application framework wires ingest, ML, storage, and the operator view.',
        bullets=(),
        kind='figure',
        track='Platform',
        image='routemq-connect-map.png',
        source='app/routers/telemetry.py; app/controllers/anomaly_controller.py; app/jobs/*; README.md:13-15',
    ),
    SlideSpec(
        title='Layer 1 - Router: Ingest and Dispatch',
        kicker='The entry point of the synchronous path.',
        bullets=(
            'Subscribes the topic factory/skab/{station}/telemetry at QoS 1.',
            'Routing is declarative: the matched topic dispatches to anomaly_controller.ingest.',
            'A per-station wildcard keeps one route for every pump station.',
            'Nothing heavy happens here; the Router only resolves topic to handler.',
        ),
        kind='detail',
        track='Platform',
        source='app/routers/telemetry.py:12-21',
    ),
    SlideSpec(
        title='Layer 2 - Middleware: Validate, Rate-limit, Correlate',
        kicker='An ordered chain runs before the controller and protects inference.',
        bullets=(
            'ValidateTelemetryMiddleware: reject non-JSON payloads and empty sensor maps early.',
            'InMemoryRateLimitMiddleware: cap 50 messages per second per station.',
            'CorrelationLoggingMiddleware: attach a correlation id so a message is traceable end to end.',
            'Order matters: bad or excess traffic is dropped before it reaches the model.',
        ),
        kind='detail',
        track='Platform',
        source='app/middleware/validate_payload.py:9; app/middleware/rate_limit.py:11; app/middleware/correlation.py:10',
    ),
    SlideSpec(
        title='Layer 3 - Controller, Inference, Persistence',
        kicker='The synchronous core: decide, publish, store.',
        bullets=(
            'anomaly_controller calls the inference service, which loads the champion and scores the window (PCA T2/Q).',
            'Inference is in-path and synchronous; it emits an AnomalyVerdict, not a queued job.',
            'It publishes the result to factory/skab/{station}/anomaly at QoS 1.',
            'Persistence is Redis latest-state first, then ClickHouse history for the timeline.',
        ),
        kind='detail',
        track='Platform',
        source='app/controllers/anomaly_controller.py:26-56; app/services/inference.py; app/services/persistence.py',
    ),
    SlideSpec(
        title='Layer 4 - Queue/Worker: MLOps Jobs',
        kicker='The asynchronous lane where continuous learning lives.',
        bullets=(
            'DriftReportJob and RetrainingJob run off the queue, not in the anomaly inference path.',
            'They read ClickHouse/SKAB data and write the MLflow registry plus Redis report keys.',
            'The Controller picks up the new champion from MLflow; promotion is gated by the custom champion-challenger rule.',
            'The scheduler is off by default (env flags), so the loop is opt-in, not always-on.',
        ),
        kind='detail',
        track='Platform',
        source='app/jobs/drift_report_job.py:26-29; app/jobs/retraining_job.py:47-50; ml/monitoring/scheduler.py',
    ),
    SlideSpec(
        title='Operator Dashboard (Streamlit)',
        kicker='Seven operator pages over the same evidence.',
        bullets=(
            'Overview ("PDAM Pump Sentinel"): console-health banner, KPI tiles, station picker.',
            'Live Sensor Monitoring: 5 s autorefresh, anomaly score, sensor status.',
            'Anomaly History: timeline, severity buckets, drilldown.',
            'Model Registry: active champion + version table.',
            'Drift & Retrain Reports; System Health: 10 s autorefresh, probes.',
            'Operator Runbook + actions ack/mute/note that write Redis keys (not yet training labels).',
        ),
        kind='screenshot',
        screenshot='t9-observability-streamlit-observability-snapshot-20260628T0535Z.png',
        source='dashboard/pages/*; dashboard/data.py:213-312',
    ),
    SlideSpec(
        title='What Is Observability',
        kicker='Asking arbitrary questions about a running system from the outside.',
        bullets=(
            'Three pillars: metrics (numbers over time), logs (events), traces (request paths).',
            'Goal: understand system state and behavior without redeploying or guessing.',
            'It turns "is it healthy?" into a question you can answer with data.',
            'It is the basis for alerting, triage, and an operator runbook.',
        ),
        kind='detail',
        track='Platform',
        source='general concept; infra/README.md',
    ),
    SlideSpec(
        title='Observability in Our AIoT Case',
        kicker='What we actually implemented — local evidence, not production SRE.',
        bullets=(
            'Prometheus /metrics joins RouteMQ framework metrics with 15 pumpad_* application metrics.',
            'Grafana dashboards: RouteMQ Observability, MLOps Loop, System Health, MQTT Broker.',
            'Seven local alert rules: telemetry stale, inference/persistence errors, model age, high-severity anomalies.',
            'This proves metrics + dashboards + local alerts; centralized logs/traces and incident routing are future.',
            'All of it is local Docker Compose evidence, not a production cloud claim.',
        ),
        kind='screenshot',
        screenshot='t9-observability-grafana-pipeline-observability-20260609T130723Z.png',
        source='app/observability/metrics.py:33-99; infra/prometheus/rules/pumpad-alerts.yml:1-65; infra/README.md:8',
    ),
    SlideSpec(
        title='Limitations, Roadmap, Close',
        kicker='The MVP is demo-ready because its limits are explicit, not hidden.',
        bullets=(
            'Known limits: synchronous inference, schedulers off by default, scheduled retrain defaults to PCA, process-local hot-swap.',
            'Roadmap: operator labels then supervised promotion, supervised alias setter, incident routing.',
            'Closing thesis: an honest end-to-end anomaly-detection slice, not a standalone notebook.',
        ),
        kind='closing',
        image='slide-roadmap.png',
        source='README.md §Known Limitations; sprint-remaining.md §Architectural follow-ups',
    ),
)


def build_deck(output_path: Path = OUTPUT_PATH) -> None:
    _ensure_assets()
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    for index, spec in enumerate(SLIDES, start=1):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        _paint_background(slide)
        if spec.kind != 'title':
            _add_header(slide, spec, index)

        if spec.kind == 'title':
            _render_title(slide, spec)
        elif spec.kind == 'architecture':
            _render_architecture(slide, spec)
        elif spec.kind == 'timeline':
            _render_timeline(slide, spec)
        elif spec.kind == 'spectrum':
            _render_spectrum(slide, spec)
        elif spec.kind == 'screenshot':
            _render_screenshot(slide, spec)
        elif spec.kind == 'two_screenshots':
            _render_two_screenshots(slide, spec)
        elif spec.kind == 'closing':
            _render_closing(slide, spec)
        elif spec.kind == 'detail':
            _render_detail(slide, spec)
        elif spec.kind == 'figure':
            _render_figure(slide, spec)
        elif spec.kind == 'table':
            _render_table(slide, spec)
        else:
            _render_bullets(slide, spec)

        _add_footer(slide, spec, index)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output_path))


def _ensure_assets() -> None:
    if not OUTLINE_PATH.exists():
        raise FileNotFoundError(f'Missing outline: {OUTLINE_PATH}')
    missing = [name for name in CANONICAL_SCREENSHOTS if not (SCREENSHOT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f'Missing canonical screenshots: {missing}')
    missing_images = [spec.image for spec in SLIDES if spec.image and not (ASSET_DIR / spec.image).exists()]
    if missing_images:
        raise FileNotFoundError(f'Missing generated illustrations: {missing_images}')


def _paint_background(slide) -> None:
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = PAGE
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.22))
    band.fill.solid()
    band.fill.fore_color.rgb = TEAL
    band.line.fill.background()


def _add_header(slide, spec: SlideSpec, index: int) -> None:
    _add_text(slide, f'{index:02d}', 0.72, 0.4, 0.55, 0.3, 11, MUTED, bold=True)
    _add_text(slide, spec.title, 0.72, 0.66, 12.0, 0.62, 28, INK, bold=True)
    _add_text(slide, spec.kicker, 0.74, 1.32, 11.9, 0.4, 14, CYAN)


def _add_footer(slide, spec: SlideSpec, index: int) -> None:
    _add_text(slide, spec.source, 0.72, 7.08, 10.6, 0.26, 7.8, MUTED)
    _add_text(slide, f'{index}/{len(SLIDES)}', 12.15, 7.08, 0.6, 0.26, 8, MUTED, align=PP_ALIGN.RIGHT)


def _render_title(slide, spec: SlideSpec) -> None:
    _add_text(slide, spec.title, 0.82, 1.5, 6.5, 0.82, 42, INK, bold=True)
    _add_text(slide, spec.kicker, 0.86, 2.42, 6.6, 0.5, 16, CYAN)
    _add_bullets(slide, spec.bullets, 0.9, 3.1, 6.3, 2.0, 16)
    _add_metric_card(slide, '40%', 'DevOps + RouteMQ', 0.9, 5.35, TEAL)
    _add_metric_card(slide, '60%', 'AI/ML + MLOps', 2.95, 5.35, AMBER)
    if spec.image:
        _add_flat_image(slide, spec.image, 7.45, 2.0, 5.45, 4.3)


def _render_detail(slide, spec: SlideSpec) -> None:
    color = TEAL if spec.track == 'Modeling' else CYAN
    _add_badge(slide, spec.track, 0.72, 1.9, 1.7, 0.42, color)
    if spec.image:
        _add_bullets(slide, spec.bullets, 0.72, 2.55, 7.55, 4.2, 14)
        _add_flat_image(slide, spec.image, 8.45, 2.55, 4.35, 4.05)
        return
    half = (len(spec.bullets) + 1) // 2
    _add_bullets(slide, spec.bullets[:half], 0.72, 2.55, 6.05, 4.1, 15)
    _add_bullets(slide, spec.bullets[half:], 7.0, 2.55, 5.6, 4.1, 15)


def _render_figure(slide, spec: SlideSpec) -> None:
    if spec.bullets:
        _add_flat_image(slide, spec.image, 0.72, 1.9, 11.9, 3.55)
        _add_bullets(slide, spec.bullets, 0.92, 5.7, 11.4, 0.95, 13.5)
    else:
        _add_flat_image(slide, spec.image, 0.72, 1.9, 11.9, 4.9)


def _render_bullets(slide, spec: SlideSpec) -> None:
    if spec.image:
        _add_bullets(slide, spec.bullets, 0.92, 2.0, 6.3, 4.2, 18)
        _add_flat_image(slide, spec.image, 7.35, 1.9, 5.5, 4.5)
        return
    _add_bullets(slide, spec.bullets, 0.92, 2.0, 7.1, 3.7, 19)
    _add_side_panel(slide, 'Presenter emphasis', _emphasis_for(spec.title), 8.45, 2.0, 3.95, 3.65)


def _render_architecture(slide, spec: SlideSpec) -> None:
    if spec.image:
        _add_flat_image(slide, spec.image, 0.72, 1.95, 11.9, 3.5)
        _add_bullets(slide, spec.bullets, 0.92, 5.7, 11.4, 0.9, 13.5)
        return
    labels = [
        ('DevOps Layer', 'Docker Compose\nPrometheus + Grafana\nHealth checks', TEAL),
        ('RouteMQ App', 'MQTT Router\nMiddleware + Controllers\nRedis + ClickHouse', CYAN),
        ('MLOps Layer', 'Training\nMLflow Registry\nDrift + Retrain', AMBER),
    ]
    x_positions = [0.78, 4.68, 8.58]
    for x, (title, body, color) in zip(x_positions, labels, strict=True):
        _add_layer_card(slide, title, body, x, 2.15, color)
    _add_arrow(slide, 3.6, 3.25, 0.72)
    _add_arrow(slide, 7.5, 3.25, 0.72)
    _add_bullets(slide, spec.bullets, 0.92, 5.58, 11.4, 0.9, 13.5)


def _render_timeline(slide, spec: SlideSpec) -> None:
    events = [
        ('T+0', 'Baseline'),
        ('T+2', 'Anomaly'),
        ('T+4', 'Drift detect'),
        ('T+5', 'Retrain'),
        ('T+7', 'Promote'),
        ('T+8', 'Recover'),
        ('T+9', 'Evidence'),
    ]
    start_x = 0.72
    y = 3.0
    width = 1.55
    for idx, (time, label) in enumerate(events):
        x = start_x + idx * 1.78
        color = TEAL if idx in {0, 5, 6} else AMBER if idx in {2, 3, 4} else RED
        _add_badge(slide, time, x, y, width, 0.45, color)
        _add_text(slide, label, x - 0.12, y + 0.62, width + 0.24, 0.5, 12, INK, align=PP_ALIGN.CENTER)
        if idx < len(events) - 1:
            _add_arrow(slide, x + width + 0.04, y + 0.12, 0.38)
    _add_bullets(slide, spec.bullets, 0.92, 4.55, 11.4, 1.25, 14.5)


def _render_spectrum(slide, spec: SlideSpec) -> None:
    rows = [
        ('PCA-spectral', 'normal-only', 'F1 ≈ 0.58', GREEN),
        ('Supervised cross-group', 'labeled anomalies', 'F1 ≈ 0.60', CYAN),
        ('XGB/LGBM in-distribution', 'known faults in train', 'F1 ≈ 0.90', AMBER),
        ('Random-window leakage', 'not defensible', 'do not claim', RED),
    ]
    for idx, (name, split, metric, color) in enumerate(rows):
        y = 2.0 + idx * 0.92
        _add_badge(slide, name, 0.88, y, 3.35, 0.5, color)
        _add_text(slide, split, 4.58, y + 0.04, 3.45, 0.4, 14, INK)
        _add_text(slide, metric, 9.0, y + 0.04, 2.6, 0.4, 14, color, bold=True)
    _add_side_panel(slide, 'Rule for Q&A', 'Never present an in-distribution number as novel-fault readiness.', 8.55, 5.15, 3.85, 1.18)


def _render_screenshot(slide, spec: SlideSpec) -> None:
    _add_bullets(slide, spec.bullets, 0.78, 1.82, 3.8, 4.5, 13.5)
    assert spec.screenshot is not None
    _add_picture(slide, SCREENSHOT_DIR / spec.screenshot, 4.98, 1.9, 7.35, 4.14)
    _add_text(slide, spec.screenshot, 5.05, 6.12, 7.2, 0.26, 7.5, MUTED)


def _render_two_screenshots(slide, spec: SlideSpec) -> None:
    _add_text(slide, 'Status → metric evidence → runbook step → action history', 0.82, 1.82, 11.7, 0.36, 16, CYAN)
    _add_picture(slide, SCREENSHOT_DIR / spec.screenshots[0], 0.82, 2.5, 5.7, 3.2)
    _add_picture(slide, SCREENSHOT_DIR / spec.screenshots[1], 6.82, 2.5, 5.7, 3.2)
    _add_badge(slide, 'Observability snapshot', 0.82, 5.92, 2.3, 0.42, TEAL)
    _add_badge(slide, 'Metric-driven runbook', 6.82, 5.92, 2.3, 0.42, AMBER)


def _render_closing(slide, spec: SlideSpec) -> None:
    _add_bullets(slide, spec.bullets, 0.92, 1.9, 6.4, 3.2, 17)
    _add_metric_card(slide, 'Now', 'Demo-ready local MLOps slice', 0.92, 5.3, GREEN)
    _add_metric_card(slide, 'Next', 'Operator labels + supervised gates', 2.97, 5.3, AMBER)
    if spec.image:
        _add_flat_image(slide, spec.image, 7.45, 1.95, 5.45, 3.5)
    _add_flat_image(slide, 't9-evidence-storyboard.png', 7.45, 5.0, 5.45, 1.5)
    _add_badge(slide, 'End-to-end platform, not a standalone notebook', 7.7, 6.55, 4.9, 0.55, TEAL)


def _render_table(slide, spec: SlideSpec) -> None:
    _add_badge(slide, spec.track, 0.72, 1.82, 1.7, 0.4, TEAL)
    row_count = len(spec.table)
    col_count = len(spec.table[0])
    left, top, width, row_height = 0.72, 2.3, 11.9, 0.35
    graphic = slide.shapes.add_table(
        row_count, col_count, Inches(left), Inches(top), Inches(width), Inches(row_height * row_count)
    )
    table = graphic.table
    table.first_row = False
    table.first_col = False
    table.horz_banding = False
    table.vert_banding = False
    for idx, col_width in enumerate((4.7, 1.5, 5.7)):
        table.columns[idx].width = Inches(col_width)
    for r in range(row_count):
        table.rows[r].height = Inches(row_height)
        if r == 0:
            fill, ink, bold, size = TEAL, WHITE, True, 13
        elif r == 1:
            fill, ink, bold, size = GREEN, WHITE, True, 12
        else:
            fill, ink, bold, size = (PANEL if r % 2 == 1 else WHITE), INK, False, 12
        for c in range(col_count):
            align = PP_ALIGN.CENTER if c == 1 else PP_ALIGN.LEFT
            _style_cell(table.cell(r, c), spec.table[r][c], size, ink, fill, bold=bold, align=align)
    if spec.caption:
        _add_text(slide, spec.caption, 0.74, top + row_height * row_count + 0.16, 11.9, 0.62, 10.5, RED)


def _style_cell(
    cell,
    text: str,
    size: float,
    color: RGBColor,
    fill: RGBColor,
    *,
    bold: bool = False,
    align: PP_ALIGN = PP_ALIGN.LEFT,
) -> None:
    cell.fill.solid()
    cell.fill.fore_color.rgb = fill
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_left = Inches(0.09)
    cell.margin_right = Inches(0.07)
    cell.margin_top = Inches(0.01)
    cell.margin_bottom = Inches(0.01)
    frame = cell.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.alignment = align
    font = paragraph.font
    font.name = 'Aptos'
    font.size = Pt(size)
    font.bold = bold
    font.color.rgb = color


def _add_text(
    slide,
    text: str,
    left: float,
    top: float,
    width: float,
    height: float,
    size: float,
    color: RGBColor,
    *,
    bold: bool = False,
    align: PP_ALIGN = PP_ALIGN.LEFT,
) -> None:
    shape = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = shape.text_frame
    frame.clear()
    frame.vertical_anchor = MSO_ANCHOR.TOP
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.alignment = align
    font = paragraph.font
    font.name = 'Aptos'
    font.size = Pt(size)
    font.bold = bold
    font.color.rgb = color


def _add_bullets(slide, bullets: tuple[str, ...], left: float, top: float, width: float, height: float, size: float) -> None:
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    for idx, bullet in enumerate(bullets):
        paragraph = frame.paragraphs[0] if idx == 0 else frame.add_paragraph()
        paragraph.text = bullet
        paragraph.level = 0
        paragraph.space_after = Pt(7)
        font = paragraph.font
        font.name = 'Aptos'
        font.size = Pt(size)
        font.color.rgb = INK


def _add_badge(slide, text: str, left: float, top: float, width: float, height: float, color: RGBColor) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.color.rgb = color
    frame = shape.text_frame
    frame.clear()
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.alignment = PP_ALIGN.CENTER
    font = paragraph.font
    font.name = 'Aptos'
    font.size = Pt(12)
    font.bold = True
    font.color.rgb = _ink_for(color)


def _add_metric_card(slide, value: str, label: str, left: float, top: float, color: RGBColor) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(1.7), Inches(1.65))
    shape.fill.solid()
    shape.fill.fore_color.rgb = PANEL
    shape.line.color.rgb = color
    shape.line.width = Pt(2)
    _add_text(slide, value, left + 0.18, top + 0.18, 1.34, 0.52, 26, color, bold=True, align=PP_ALIGN.CENTER)
    _add_text(slide, label, left + 0.16, top + 0.9, 1.38, 0.48, 10.5, INK, align=PP_ALIGN.CENTER)


def _add_layer_card(slide, title: str, body: str, left: float, top: float, color: RGBColor) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(3.0), Inches(2.0))
    shape.fill.solid()
    shape.fill.fore_color.rgb = PANEL
    shape.line.color.rgb = color
    shape.line.width = Pt(2)
    _add_text(slide, title, left + 0.2, top + 0.18, 2.6, 0.36, 16, color, bold=True, align=PP_ALIGN.CENTER)
    _add_text(slide, body, left + 0.28, top + 0.74, 2.44, 0.9, 12.2, INK, align=PP_ALIGN.CENTER)


def _add_side_panel(slide, title: str, body: str, left: float, top: float, width: float, height: float) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = PANEL
    shape.line.color.rgb = TEAL
    _add_text(slide, title, left + 0.26, top + 0.18, width - 0.52, 0.34, 13, TEAL, bold=True)
    _add_text(slide, body, left + 0.26, top + 0.64, width - 0.52, height - 0.82, 13, INK)


def _add_picture(slide, path: Path, left: float, top: float, width: float, height: float) -> None:
    frame = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left - 0.05), Inches(top - 0.05), Inches(width + 0.1), Inches(height + 0.1))
    frame.fill.solid()
    frame.fill.fore_color.rgb = PANEL
    frame.line.color.rgb = TEAL
    frame.line.width = Pt(1.2)
    slide.shapes.add_picture(str(path), Inches(left), Inches(top), width=Inches(width), height=Inches(height))


def _add_flat_image(slide, name: str, box_left: float, box_top: float, box_w: float, box_h: float) -> None:
    path = ASSET_DIR / name
    with PILImage.open(path) as img:
        iw, ih = img.size
    aspect = iw / ih
    if aspect >= box_w / box_h:
        width = box_w
        height = box_w / aspect
    else:
        height = box_h
        width = box_h * aspect
    left = box_left + (box_w - width) / 2
    top = box_top + (box_h - height) / 2
    slide.shapes.add_picture(str(path), Inches(left), Inches(top), width=Inches(width), height=Inches(height))


def _add_arrow(slide, left: float, top: float, width: float) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(left), Inches(top), Inches(width), Inches(0.22))
    shape.fill.solid()
    shape.fill.fore_color.rgb = MUTED
    shape.line.fill.background()


def _emphasis_for(title: str) -> str:
    emphasis = {
        'What Is Implemented Today': 'Use “local evidence” language, not production-cloud language.',
    }
    return emphasis.get(title, 'One clear claim per slide; defer implementation details to Q&A.')


if __name__ == '__main__':
    build_deck()
    print(OUTPUT_PATH)
