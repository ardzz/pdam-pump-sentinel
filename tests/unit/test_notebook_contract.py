import json
from pathlib import Path

import pytest

NOTEBOOK_PATH = Path(__file__).parent.parent.parent / 'notebooks' / 'pca_artifact_visualization.ipynb'
SKAB_NOTEBOOK_PATH = Path(__file__).parent.parent.parent / 'notebooks' / 'skab_eda_and_data_prep.ipynb'


@pytest.fixture
def notebook():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def skab_notebook():
    with open(SKAB_NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_notebook_is_valid_json(notebook):
    assert isinstance(notebook, dict)
    assert 'cells' in notebook
    assert 'metadata' in notebook
    assert notebook.get('nbformat') == 4
    assert notebook.get('nbformat_minor') == 5


def test_notebook_has_cells(notebook):
    cells = notebook['cells']
    assert isinstance(cells, list)
    assert len(cells) >= 9


def test_notebook_cells_have_no_outputs(notebook):
    for cell in notebook['cells']:
        outputs = cell.get('outputs')
        assert outputs is None or outputs == [], f"Cell has non-empty outputs: {cell.get('cell_type')}"


def test_notebook_code_cells_have_null_execution_count(notebook):
    for cell in notebook['cells']:
        if cell.get('cell_type') == 'code':
            assert cell.get('execution_count') is None, "Code cell execution_count must be null"


def test_notebook_contains_expected_markdown(notebook):
    markdown_text = ''.join(
        ''.join(cell.get('source', []))
        for cell in notebook['cells']
        if cell.get('cell_type') == 'markdown'
    )
    expected_fragments = [
        'Visualisasi Artefak PCA',
        'Dataset Demo',
        'Pelatihan PCA T²/Q',
        'Generasi Visualisasi',
        'Ringkasan Hasil',
        'Grafik Scatter T²/Q',
        'Grafik Timeline',
        'Catatan',
        'skab_tiny.csv',
        'bukan data operasional PDAM nyata',
    ]
    for fragment in expected_fragments:
        assert fragment in markdown_text, f"Missing markdown fragment: {fragment!r}"


def test_notebook_contains_expected_code_references(notebook):
    code_text = ''.join(
        ''.join(cell.get('source', []))
        for cell in notebook['cells']
        if cell.get('cell_type') == 'code'
    )
    expected_fragments = [
        'from ml.training.train_pca import PcaTrainingConfig, train_pca_from_skab',
        'from ml.visualization.pca_artifacts import',
        'PcaArtifactVisualizationConfig',
        'generate_pca_artifact_visualizations',
        'window_size=1',
        'stride=1',
        'threshold_quantile=0.95',
        'log_mlflow=False',
        'find_project_root',
        "PROJECT_ROOT / 'tests' / 'fixtures' / 'skab_tiny.csv'",
        'sys.path.insert(0, str(PROJECT_ROOT))',
        '.to_html(include_plotlyjs=True, full_html=False)',
        'IPython.display import HTML',
        'plotly.io as pio',
        'pdam-pump-sentinel-pca-demo/artifacts',
    ]
    for fragment in expected_fragments:
        assert fragment in code_text, f"Missing code fragment: {fragment!r}"


def test_notebook_first_cell_is_markdown_intro(notebook):
    first = notebook['cells'][0]
    assert first.get('cell_type') == 'markdown'
    source = ''.join(first.get('source', []))
    assert 'Visualisasi Artefak PCA' in source


def test_notebook_training_cell_precedes_visualization_cell(notebook):
    code_texts = [
        ''.join(cell.get('source', []))
        for cell in notebook['cells']
        if cell.get('cell_type') == 'code'
    ]
    train_idx = next(
        (i for i, text in enumerate(code_texts) if 'train_pca_from_skab' in text),
        None,
    )
    viz_idx = next(
        (i for i, text in enumerate(code_texts) if 'generate_pca_artifact_visualizations' in text),
        None,
    )
    assert train_idx is not None, "Missing training code cell"
    assert viz_idx is not None, "Missing visualization code cell"
    assert train_idx < viz_idx, "Training cell must precede visualization cell"


def test_notebook_offline_plotly_cells_exist(notebook):
    code_texts = [
        ''.join(cell.get('source', []))
        for cell in notebook['cells']
        if cell.get('cell_type') == 'code'
    ]
    scatter_found = any('scores_scatter_plotly_json' in text for text in code_texts)
    timeline_found = any('scores_timeline_plotly_json' in text for text in code_texts)
    assert scatter_found, "Missing scatter plot JSON loading cell"
    assert timeline_found, "Missing timeline plot JSON loading cell"


def test_notebook_contains_static_svg_chart_previews(notebook):
    markdown_text = ''.join(
        ''.join(cell.get('source', []))
        for cell in notebook['cells']
        if cell.get('cell_type') == 'markdown'
    )
    assert markdown_text.count('<svg') >= 2
    assert 'Scatter T²/Q' in markdown_text
    assert 'Timeline T²/Q/Score' in markdown_text
    assert 'Preview SVG statis' in markdown_text


def test_skab_notebook_is_valid_json(skab_notebook):
    assert isinstance(skab_notebook, dict)
    assert 'cells' in skab_notebook
    assert 'metadata' in skab_notebook
    assert skab_notebook.get('nbformat') == 4
    assert skab_notebook.get('nbformat_minor') == 5


def test_skab_notebook_has_cells(skab_notebook):
    cells = skab_notebook['cells']
    assert isinstance(cells, list)
    assert len(cells) >= 9


def test_skab_notebook_cells_have_no_outputs(skab_notebook):
    for cell in skab_notebook['cells']:
        outputs = cell.get('outputs')
        assert outputs is None or outputs == [], f"Cell has non-empty outputs: {cell.get('cell_type')}"


def test_skab_notebook_code_cells_have_null_execution_count(skab_notebook):
    for cell in skab_notebook['cells']:
        if cell.get('cell_type') == 'code':
            assert cell.get('execution_count') is None, "Code cell execution_count must be null"


def test_skab_notebook_contains_expected_markdown(skab_notebook):
    markdown_text = ''.join(
        ''.join(cell.get('source', []))
        for cell in skab_notebook['cells']
        if cell.get('cell_type') == 'markdown'
    )
    expected_fragments = [
        'SKAB EDA dan Persiapan Data',
        'Persiapan Data',
        'tests/fixtures/skab_tiny.csv',
        'summary.json',
        'report.md',
        'missing policy',
        'timestamp quality',
        'correlation_matrix.csv',
        'sensor_distributions.csv',
        'rolling_statistics.csv',
        'label_ranges.csv',
        'label_overlay.csv',
        'split manifest metadata/base_dir',
        'held-out test metrics',
        'event metrics',
        'changepoint separation',
        'provenance',
        'bukan data operasional PDAM nyata',
    ]
    for fragment in expected_fragments:
        assert fragment in markdown_text, f"Missing markdown fragment: {fragment!r}"


def test_skab_notebook_contains_expected_code_references(skab_notebook):
    code_text = ''.join(
        ''.join(cell.get('source', []))
        for cell in skab_notebook['cells']
        if cell.get('cell_type') == 'code'
    )
    expected_fragments = [
        'find_project_root',
        'sys.path.insert',
        'validate_sensor_values',
        'generate_skab_eda_report',
        'timestamp_quality.json',
        'missingness.csv',
        'correlation_matrix.csv',
        'sensor_distributions.csv',
        'rolling_statistics.csv',
        'label_ranges.csv',
        'label_overlay.csv',
        'input_timestamp_quality_json',
        'input_missingness_csv',
        'input_correlation_matrix_csv',
        'input_sensor_distributions_csv',
        'input_rolling_statistics_csv',
        'input_label_ranges_csv',
        'input_label_overlay_csv',
        'SENSOR_COLUMNS',
        "'base_dir': 'files'",
        'load_skab_split_manifest',
        'validation_label_ranges_csv',
        'test_timestamp_quality_json',
        'load_skab_csv',
        'build_sensor_windows',
        'window_size=1',
        'stride=1',
        'PcaTrainingConfig',
        'train_pca_from_skab',
        'metric_protocol',
        'test_split_held_out',
        'provenance',
        'test_scores',
    ]
    for fragment in expected_fragments:
        assert fragment in code_text, f"Missing code fragment: {fragment!r}"


def test_skab_notebook_first_cell_is_markdown_intro(skab_notebook):
    first = skab_notebook['cells'][0]
    assert first.get('cell_type') == 'markdown'
    source = ''.join(first.get('source', []))
    assert 'SKAB EDA dan Persiapan Data' in source


def test_skab_notebook_contains_static_svg_chart_previews(skab_notebook):
    markdown_text = ''.join(
        ''.join(cell.get('source', []))
        for cell in skab_notebook['cells']
        if cell.get('cell_type') == 'markdown'
    )
    assert markdown_text.count('<svg') >= 2
