from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _gitignore_lines() -> list[str]:
    return (ROOT / '.gitignore').read_text(encoding='utf-8').splitlines()


def test_local_automation_and_runtime_outputs_are_ignored():
    lines = _gitignore_lines()

    assert '.omo/' in lines
    assert 'mlruns/' in lines
    assert 'mlartifacts/' in lines
    assert 'mlflow.db' in lines
    assert 'drift_reports/' in lines
    assert 'logs/*' in lines
    assert '*.log' in lines


def test_artifacts_policy_keeps_only_selected_experiment_summaries():
    lines = _gitignore_lines()

    assert '/artifacts/**' in lines
    assert '!/artifacts/' in lines
    assert '!/artifacts/skab-model-experiments/' in lines
    assert '!/artifacts/skab-model-experiments/summary.md' in lines
    assert '!/artifacts/skab-model-experiments/summary.csv' in lines

    assert (ROOT / 'artifacts' / 'skab-model-experiments' / 'summary.md').is_file()
    assert (ROOT / 'artifacts' / 'skab-model-experiments' / 'summary.csv').is_file()


def test_curated_report_and_presentation_evidence_stays_tracked_by_policy():
    laporan_assets = sorted((ROOT / 'docs' / 'laporan' / 'assets').glob('*.png'))
    presentation_screenshots = sorted((ROOT / 'docs' / 'presentation' / 'screenshots').glob('*.png'))

    assert laporan_assets, 'expected final report image evidence under docs/laporan/assets'
    assert presentation_screenshots, 'expected curated presentation screenshot evidence'
    assert any(path.name.startswith('t9-observability-') for path in presentation_screenshots)


def test_data_policy_keeps_manifest_docs_but_excludes_bulk_data():
    lines = _gitignore_lines()

    assert 'data/*' in lines
    assert '!data/.gitkeep' in lines
    assert '!data/README.md' in lines
