from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUTLINE_PATH = ROOT / 'docs/presentation/deck-outline.md'
OUTPUT_PATH = ROOT / 'docs/presentation/pdam-pump-sentinel-deck.pptx'
SCREENSHOT_DIR = ROOT / 'docs/presentation/screenshots'

CANONICAL_SCREENSHOTS = [
    't9-observability-grafana-mlops-observability-20260609T130723Z.png',
    't9-observability-grafana-pipeline-observability-20260609T130723Z.png',
    't9-observability-grafana-slo-health-20260609T130723Z.png',
    't9-observability-streamlit-observability-snapshot-20260628T0535Z.png',
    't9-observability-streamlit-runbook-observability-20260628T0535Z.png',
]

NAVY = RGBColor(12, 22, 37)
NAVY_2 = RGBColor(15, 34, 54)
TEAL = RGBColor(0, 179, 166)
CYAN = RGBColor(70, 208, 220)
WHITE = RGBColor(245, 250, 252)
MUTED = RGBColor(168, 185, 196)
AMBER = RGBColor(255, 196, 87)
RED = RGBColor(255, 100, 116)
GREEN = RGBColor(88, 211, 139)


@dataclass(frozen=True)
class SlideSpec:
    title: str
    kicker: str
    bullets: tuple[str, ...]
    kind: str = 'bullets'
    screenshot: str | None = None
    screenshots: tuple[str, ...] = ()
    source: str = ''


SLIDES: tuple[SlideSpec, ...] = (
    SlideSpec(
        title='PDAM Pump Sentinel',
        kicker='RouteMQ + PCA T²/Q + MLflow + Grafana + Streamlit',
        bullets=(
            'End-to-end AIoT slice for water-pump predictive maintenance.',
            'SKAB water-circulation benchmark is used as surrogate academic data.',
            'Contribution split: DevOps/RouteMQ foundation plus AI/ML MLOps lifecycle.',
        ),
        kind='title',
        source='README.md; docs/plans/design.md',
    ),
    SlideSpec(
        title='Why This Problem Matters',
        kicker='Pump faults are multivariate, gradual, and operationally costly.',
        bullets=(
            'Centrifugal pumps are critical assets in water distribution operations.',
            'Simple min/max thresholds miss contextual, drift, and collective anomalies.',
            'The goal is early condition monitoring, not just post-failure alerting.',
        ),
        source='docs/plans/design.md §2',
    ),
    SlideSpec(
        title='Scope and Honest Data Framing',
        kicker='The demo is evidence-driven, but it does not pretend to be live PDAM production data.',
        bullets=(
            'SKAB is a public controlled water-circulation testbed with injected faults.',
            'MVP excludes real ESP32 sensors, Kubernetes, and production incident routing.',
            'The deck proves pipeline methodology and local demo readiness.',
        ),
        source='README.md; design.md §3; labeling-strategy-notes.md §3',
    ),
    SlideSpec(
        title='Architecture: Three-Layer Slice',
        kicker='DevOps, RouteMQ application, and MLOps are connected as one vertical system.',
        bullets=(
            'DevOps layer: Docker Compose, Prometheus, Grafana, health checks.',
            'RouteMQ layer: MQTT router, middleware, controllers, Redis, ClickHouse.',
            'MLOps layer: training, MLflow registry, Evidently drift, retraining, promotion.',
        ),
        kind='architecture',
        source='README.md; docs/plans/design.md §5',
    ),
    SlideSpec(
        title='What Is Implemented Today',
        kicker='Local observability evidence is implemented; production cloud readiness is roadmap.',
        bullets=(
            'RouteMQ exposes framework and pump-specific metrics on /metrics.',
            'Grafana covers RouteMQ, MLOps, system health, and MQTT broker dashboards.',
            'Streamlit covers overview, live sensors, system health, and runbook triage.',
            'make demo verifies T+0 through T+8, with optional T+9 observability evidence.',
        ),
        source='README.md; screenshot-checklist.md; sprint-remaining.md',
    ),
    SlideSpec(
        title='Demo Storyboard: T+0 to T+9',
        kicker='The live narrative is a controlled lifecycle, not an ad-hoc clickthrough.',
        bullets=(
            'T+0 baseline: champion model and healthy dashboard.',
            'T+1/T+2: normal replay followed by anomalous replay.',
            'T+3/T+4: drift injection and drift detection.',
            'T+5-T+8: retrain, verify challenger, promote alias, hot-swap inference.',
            'T+9: optional observability evidence capture.',
        ),
        kind='timeline',
        source='docs/presentation/screenshot-checklist.md',
    ),
    SlideSpec(
        title='Why PCA-Spectral as Day-1 Champion',
        kicker='PCA is selected for deployability before labeled fault inventory exists.',
        bullets=(
            'Normal-baseline-first matches industrial predictive-maintenance cold starts.',
            'T² tracks distance along learned normal patterns; Q/SPE tracks residuals.',
            'PCA is fast, decomposable, and operator-explainable for the first deployment stage.',
            'Supervised models remain challengers once operator labels mature.',
        ),
        source='labeling-strategy-notes.md; sprint-remaining.md',
    ),
    SlideSpec(
        title='Honest Evaluation Spectrum',
        kicker='The same model family can look very different depending on split honesty.',
        bullets=(
            'PCA-spectral normal-only: deployable novel-fault setting, F1 ≈ 0.58.',
            'Supervised cross-group: labeled anomalies help, but novel-fault constraints remain.',
            'In-distribution supervised: XGB/LGBM ≈ 0.90, requiring known fault types in train.',
            'Random-window leakage numbers are not valid generalization claims.',
        ),
        kind='spectrum',
        source='docs/plans/sprint-remaining.md §Honest evaluation spectrum',
    ),
    SlideSpec(
        title='MLOps Loop Evidence',
        kicker='Registry, champion alias, drift/retrain path, and operator action evidence are visible.',
        bullets=(
            'MLflow model registry supports champion/challenger promotion path.',
            'Champion-challenger gate protects F1 margin and false-alarm-rate guardrails.',
            'Grafana MLOps dashboard surfaces loop signals and operator action evidence.',
        ),
        kind='screenshot',
        screenshot='t9-observability-grafana-mlops-observability-20260609T130723Z.png',
        source='README.md; labeling-strategy-notes.md §4; screenshot-checklist.md',
    ),
    SlideSpec(
        title='Pipeline Observability Evidence',
        kicker='Ingestion, inference, and persistence are observable from the RouteMQ/Grafana layer.',
        bullets=(
            'Dispatch, inference latency, anomaly score, persistence writes, and freshness are measurable.',
            'This is local Docker Compose observability evidence, not a production SRE claim.',
            'The objective is operable demo proof, not decorative monitoring.',
        ),
        kind='screenshot',
        screenshot='t9-observability-grafana-pipeline-observability-20260609T130723Z.png',
        source='README.md §Bukti Observability Portfolio',
    ),
    SlideSpec(
        title='System Health and SLO Evidence',
        kicker='Dependency health and active model freshness are visible before the Q&A starts.',
        bullets=(
            'System-health dashboard tracks scrape health and freshness signals.',
            'Local alert rules cover telemetry freshness, inference errors, persistence errors, and model age.',
            'Runbook readiness exists locally; production escalation remains future work.',
        ),
        kind='screenshot',
        screenshot='t9-observability-grafana-slo-health-20260609T130723Z.png',
        source='README.md; sprint-remaining.md',
    ),
    SlideSpec(
        title='Operator Console and Runbook',
        kicker='The operator surface includes status, observability snapshot, and metric-driven triage.',
        bullets=(
            'Overview shows system status, active model, and freshness.',
            'Runbook guides investigation from metrics to action.',
            'Operator action evidence is stored and reflected in Grafana.',
        ),
        kind='two_screenshots',
        screenshots=(
            't9-observability-streamlit-observability-snapshot-20260628T0535Z.png',
            't9-observability-streamlit-runbook-observability-20260628T0535Z.png',
        ),
        source='README.md; screenshot-checklist.md',
    ),
    SlideSpec(
        title='Limitations, Roadmap, Close',
        kicker='The MVP is demo-ready because its limits are explicit, not hidden.',
        bullets=(
            'Known limits: synchronous inference, env-flag schedulers, PCA-only scheduled retrain, local hot-swap.',
            'Roadmap: operator labels, supervised promotion gates, supervised alias setter, incident routing.',
            'Closing thesis: this is an end-to-end MLOps platform, not a standalone anomaly notebook.',
        ),
        kind='closing',
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


def _paint_background(slide) -> None:
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = NAVY
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.22))
    band.fill.solid()
    band.fill.fore_color.rgb = TEAL
    band.line.fill.background()


def _add_header(slide, spec: SlideSpec, index: int) -> None:
    _add_text(slide, f'{index:02d}', 0.45, 0.42, 0.55, 0.35, 12, MUTED, bold=True)
    _add_text(slide, spec.kicker, 1.08, 0.38, 11.75, 0.42, 14, MUTED)
    _add_text(slide, spec.title, 0.72, 0.87, 12.0, 0.68, 30, WHITE, bold=True)


def _add_footer(slide, spec: SlideSpec, index: int) -> None:
    _add_text(slide, spec.source, 0.72, 7.08, 10.6, 0.26, 7.8, MUTED)
    _add_text(slide, f'{index}/{len(SLIDES)}', 12.15, 7.08, 0.6, 0.26, 8, MUTED, align=PP_ALIGN.RIGHT)


def _render_title(slide, spec: SlideSpec) -> None:
    _add_text(slide, spec.title, 0.82, 1.65, 7.4, 0.82, 44, WHITE, bold=True)
    _add_text(slide, spec.kicker, 0.86, 2.54, 7.4, 0.5, 18, CYAN)
    _add_bullets(slide, spec.bullets, 0.9, 3.35, 7.0, 1.7, 18)
    _add_metric_card(slide, '40%', 'DevOps + RouteMQ', 8.55, 2.0, TEAL)
    _add_metric_card(slide, '60%', 'AI/ML + MLOps', 10.58, 2.0, AMBER)
    _add_badge(slide, 'Surrogate dataset: SKAB', 8.55, 4.55, 4.1, 0.58, CYAN)
    _add_badge(slide, 'First-draft presentation deck', 8.55, 5.28, 4.1, 0.58, GREEN)


def _render_bullets(slide, spec: SlideSpec) -> None:
    _add_bullets(slide, spec.bullets, 0.92, 2.0, 7.1, 3.7, 19)
    _add_side_panel(slide, 'Presenter emphasis', _emphasis_for(spec.title), 8.45, 2.0, 3.95, 3.65)


def _render_architecture(slide, spec: SlideSpec) -> None:
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
        _add_text(slide, label, x - 0.12, y + 0.62, width + 0.24, 0.5, 12, WHITE, align=PP_ALIGN.CENTER)
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
        _add_text(slide, split, 4.58, y + 0.04, 3.45, 0.4, 14, WHITE)
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
    _add_bullets(slide, spec.bullets, 0.92, 1.9, 7.25, 3.9, 18)
    _add_metric_card(slide, 'Now', 'Demo-ready local MLOps slice', 8.6, 2.05, GREEN)
    _add_metric_card(slide, 'Next', 'Operator labels + supervised gates', 10.55, 2.05, AMBER)
    _add_badge(slide, 'End-to-end platform, not a standalone notebook', 8.56, 4.62, 4.08, 0.74, TEAL)


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
        font.color.rgb = WHITE


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
    font.color.rgb = NAVY


def _add_metric_card(slide, value: str, label: str, left: float, top: float, color: RGBColor) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(1.7), Inches(1.65))
    shape.fill.solid()
    shape.fill.fore_color.rgb = NAVY_2
    shape.line.color.rgb = color
    shape.line.width = Pt(2)
    _add_text(slide, value, left + 0.18, top + 0.18, 1.34, 0.52, 26, color, bold=True, align=PP_ALIGN.CENTER)
    _add_text(slide, label, left + 0.16, top + 0.9, 1.38, 0.48, 10.5, WHITE, align=PP_ALIGN.CENTER)


def _add_layer_card(slide, title: str, body: str, left: float, top: float, color: RGBColor) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(3.0), Inches(2.0))
    shape.fill.solid()
    shape.fill.fore_color.rgb = NAVY_2
    shape.line.color.rgb = color
    shape.line.width = Pt(2)
    _add_text(slide, title, left + 0.2, top + 0.18, 2.6, 0.36, 16, color, bold=True, align=PP_ALIGN.CENTER)
    _add_text(slide, body, left + 0.28, top + 0.74, 2.44, 0.9, 12.2, WHITE, align=PP_ALIGN.CENTER)


def _add_side_panel(slide, title: str, body: str, left: float, top: float, width: float, height: float) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = NAVY_2
    shape.line.color.rgb = TEAL
    _add_text(slide, title, left + 0.26, top + 0.18, width - 0.52, 0.34, 13, TEAL, bold=True)
    _add_text(slide, body, left + 0.26, top + 0.64, width - 0.52, height - 0.82, 13, WHITE)


def _add_picture(slide, path: Path, left: float, top: float, width: float, height: float) -> None:
    frame = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left - 0.05), Inches(top - 0.05), Inches(width + 0.1), Inches(height + 0.1))
    frame.fill.solid()
    frame.fill.fore_color.rgb = NAVY_2
    frame.line.color.rgb = TEAL
    frame.line.width = Pt(1.2)
    slide.shapes.add_picture(str(path), Inches(left), Inches(top), width=Inches(width), height=Inches(height))


def _add_arrow(slide, left: float, top: float, width: float) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(left), Inches(top), Inches(width), Inches(0.22))
    shape.fill.solid()
    shape.fill.fore_color.rgb = MUTED
    shape.line.fill.background()


def _emphasis_for(title: str) -> str:
    emphasis = {
        'Why This Problem Matters': 'Move the audience from sensor charts to operational risk.',
        'Scope and Honest Data Framing': 'Say the SKAB caveat early so the rest of the deck feels credible.',
        'What Is Implemented Today': 'Use “local evidence” language, not production-cloud language.',
    }
    return emphasis.get(title, 'One clear claim per slide; defer implementation details to Q&A.')


if __name__ == '__main__':
    build_deck()
    print(OUTPUT_PATH)
