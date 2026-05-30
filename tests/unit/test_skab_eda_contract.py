from importlib import import_module
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parents[1] / 'fixtures'
SKAB_FIXTURE = FIXTURES_DIR / 'skab_tiny.csv'

SENSOR_COLUMNS = [
    'Accelerometer1RMS',
    'Accelerometer2RMS',
    'Current',
    'Pressure',
    'Temperature',
    'Thermocouple',
    'Voltage',
    'Volume Flow RateRMS',
]


def _skab_eda():
    return import_module('ml.datasets.skab_eda')


class TestSummarizeSkabCsv:
    def test_returns_skab_eda_result_dataclass(self):
        eda = _skab_eda()

        result = eda.summarize_skab_csv(SKAB_FIXTURE)

        assert isinstance(result, eda.SkabEdaResult)

    def test_result_has_expected_fields_for_single_csv(self):
        eda = _skab_eda()

        result = eda.summarize_skab_csv(SKAB_FIXTURE)

        assert result.row_count == 3
        assert result.file_count == 1
        assert result.sensor_columns == SENSOR_COLUMNS
        assert result.anomaly_count == 1
        assert result.changepoint_count == 1
        assert result.plots_available is False
        assert result.source == str(SKAB_FIXTURE)

    def test_result_has_time_range_from_datetime_column(self):
        eda = _skab_eda()

        result = eda.summarize_skab_csv(SKAB_FIXTURE)

        assert result.time_range == {
            'start': '2024-01-01T00:00:00Z',
            'end': '2024-01-01T00:00:02Z',
        }

    def test_result_has_missing_counts_per_column(self):
        eda = _skab_eda()

        result = eda.summarize_skab_csv(SKAB_FIXTURE)

        assert isinstance(result.missing_counts, dict)
        assert all(result.missing_counts.get(col, 0) == 0 for col in SENSOR_COLUMNS)
        assert result.missing_counts.get('datetime', 0) == 0
        assert result.missing_counts.get('anomaly', 0) == 0
        assert result.missing_counts.get('changepoint', 0) == 0

    def test_result_has_sensor_statistics(self):
        eda = _skab_eda()

        result = eda.summarize_skab_csv(SKAB_FIXTURE)

        assert isinstance(result.sensor_statistics, dict)
        assert set(result.sensor_statistics.keys()) == set(SENSOR_COLUMNS)
        for col in SENSOR_COLUMNS:
            stats = result.sensor_statistics[col]
            assert 'mean' in stats
            assert 'std' in stats
            assert 'min' in stats
            assert 'max' in stats

    def test_sensor_statistics_are_numeric(self):
        eda = _skab_eda()

        result = eda.summarize_skab_csv(SKAB_FIXTURE)

        stats = result.sensor_statistics['Accelerometer1RMS']
        assert stats['mean'] == pytest.approx(0.11)
        assert stats['std'] == pytest.approx(0.01)
        assert stats['min'] == pytest.approx(0.10)
        assert stats['max'] == pytest.approx(0.12)

    def test_result_is_json_serializable(self):
        import json

        eda = _skab_eda()

        result = eda.summarize_skab_csv(SKAB_FIXTURE)
        payload = result.to_payload()

        assert json.dumps(payload, sort_keys=True) is not None


class TestSkabEdaConfig:
    def test_config_dataclass_exists_with_plots_enabled(self):
        eda = _skab_eda()

        config = eda.SkabEdaConfig(plots_enabled=True)

        assert config.plots_enabled is True

    def test_config_defaults_plots_enabled_to_false(self):
        eda = _skab_eda()

        config = eda.SkabEdaConfig()

        assert config.plots_enabled is False


class TestWriteSkabEdaReport:
    def test_writes_summary_json_and_report_md(self, tmp_path):
        eda = _skab_eda()

        result = eda.summarize_skab_csv(SKAB_FIXTURE)
        eda.write_skab_eda_report(result, tmp_path)

        summary_path = tmp_path / 'summary.json'
        report_path = tmp_path / 'report.md'
        assert summary_path.exists()
        assert report_path.exists()

    def test_summary_json_is_deterministic_and_sortable(self, tmp_path):
        import json

        eda = _skab_eda()

        result = eda.summarize_skab_csv(SKAB_FIXTURE)
        eda.write_skab_eda_report(result, tmp_path)

        summary_path = tmp_path / 'summary.json'
        text = summary_path.read_text()
        payload = json.loads(text)
        assert json.dumps(payload, sort_keys=True, indent=2) == text

    def test_report_md_contains_title_source_row_count_and_labels(self, tmp_path):
        eda = _skab_eda()

        result = eda.summarize_skab_csv(SKAB_FIXTURE)
        eda.write_skab_eda_report(result, tmp_path)

        report_path = tmp_path / 'report.md'
        text = report_path.read_text()
        assert '# SKAB EDA Report' in text
        assert str(SKAB_FIXTURE) in text
        assert 'Row count' in text
        assert 'Anomalies' in text
        assert 'Changepoints' in text

    def test_report_md_contains_sensor_table(self, tmp_path):
        eda = _skab_eda()

        result = eda.summarize_skab_csv(SKAB_FIXTURE)
        eda.write_skab_eda_report(result, tmp_path)

        report_path = tmp_path / 'report.md'
        text = report_path.read_text()
        assert '| Sensor' in text
        assert 'Accelerometer1RMS' in text
        assert 'Mean' in text or 'mean' in text


class TestGenerateSkabEdaReport:
    def test_writes_single_input_artifacts_and_returns_paths(self, tmp_path):
        eda = _skab_eda()

        artifacts = eda.generate_skab_eda_report(
            input_path=SKAB_FIXTURE,
            split_manifest_path=None,
            output_dir=tmp_path,
            include_plots=False,
        )

        assert artifacts == {
            'input_report_md': tmp_path / 'report.md',
            'input_summary_json': tmp_path / 'summary.json',
        }
        assert (tmp_path / 'summary.json').exists()
        assert (tmp_path / 'report.md').exists()

    def test_writes_split_manifest_artifacts_without_overwriting_input(self, tmp_path):
        import json

        eda = _skab_eda()
        train_path = tmp_path / 'train.csv'
        validation_path = tmp_path / 'validation.csv'
        train_path.write_text(SKAB_FIXTURE.read_text())
        validation_path.write_text(SKAB_FIXTURE.read_text())
        manifest_path = tmp_path / 'manifest.json'
        manifest_path.write_text(
            json.dumps(
                {
                    'train': ['train.csv'],
                    'validation': ['validation.csv'],
                    'test': [],
                }
            )
        )

        artifacts = eda.generate_skab_eda_report(
            input_path=SKAB_FIXTURE,
            split_manifest_path=manifest_path,
            output_dir=tmp_path / 'eda',
            include_plots=False,
        )

        assert artifacts['input_summary_json'] == tmp_path / 'eda' / 'summary.json'
        assert artifacts['train_summary_json'] == tmp_path / 'eda' / 'train' / 'summary.json'
        assert artifacts['validation_report_md'] == tmp_path / 'eda' / 'validation' / 'report.md'
        assert artifacts['test_summary_json'] == tmp_path / 'eda' / 'test' / 'summary.json'
        assert (tmp_path / 'eda' / 'summary.json').exists()
        assert (tmp_path / 'eda' / 'train' / 'summary.json').exists()
        assert (tmp_path / 'eda' / 'validation' / 'report.md').exists()
        assert (tmp_path / 'eda' / 'test' / 'summary.json').exists()


class TestSummarizeSkabManifest:
    def _write_manifest(self, tmp_path: Path, content: dict) -> Path:
        import json

        manifest_path = tmp_path / 'manifest.json'
        manifest_path.write_text(json.dumps(content))
        return manifest_path

    def _make_skab_csv(self, path: Path, rows: int, anomaly_count: int = 0, changepoint_count: int = 0):
        lines = [
            'datetime;Accelerometer1RMS;Accelerometer2RMS;Current;Pressure;Temperature;Thermocouple;Voltage;Volume Flow RateRMS;anomaly;changepoint'
        ]
        for i in range(rows):
            anomaly = 1 if i < anomaly_count else 0
            changepoint = 1 if i < changepoint_count else 0
            lines.append(
                f'2024-01-01T00:00:{i:02d}Z;0.1;0.2;1.0;2.0;30.0;31.0;220.0;10.0;{anomaly};{changepoint}'
            )
        path.write_text('\n'.join(lines) + '\n')

    def test_returns_per_split_results(self, tmp_path):
        eda = _skab_eda()
        self._make_skab_csv(tmp_path / 'train.csv', 5, anomaly_count=1)
        self._make_skab_csv(tmp_path / 'val.csv', 3, changepoint_count=1)
        self._make_skab_csv(tmp_path / 'test.csv', 2)
        manifest_path = self._write_manifest(
            tmp_path,
            {'train': ['train.csv'], 'validation': ['val.csv'], 'test': ['test.csv']},
        )

        results = eda.summarize_skab_manifest(manifest_path)

        assert set(results.keys()) == {'train', 'validation', 'test'}
        assert isinstance(results['train'], eda.SkabEdaResult)
        assert isinstance(results['validation'], eda.SkabEdaResult)
        assert isinstance(results['test'], eda.SkabEdaResult)

    def test_per_split_row_and_file_counts(self, tmp_path):
        eda = _skab_eda()
        self._make_skab_csv(tmp_path / 'a.csv', 4)
        self._make_skab_csv(tmp_path / 'b.csv', 6)
        self._make_skab_csv(tmp_path / 'c.csv', 2)
        manifest_path = self._write_manifest(
            tmp_path,
            {'train': ['a.csv', 'b.csv'], 'validation': ['c.csv'], 'test': []},
        )

        results = eda.summarize_skab_manifest(manifest_path)

        assert results['train'].row_count == 10
        assert results['train'].file_count == 2
        assert results['validation'].row_count == 2
        assert results['validation'].file_count == 1
        assert results['test'].row_count == 0
        assert results['test'].file_count == 0

    def test_per_split_label_counts(self, tmp_path):
        eda = _skab_eda()
        self._make_skab_csv(tmp_path / 'train.csv', 5, anomaly_count=2, changepoint_count=1)
        self._make_skab_csv(tmp_path / 'val.csv', 3, anomaly_count=1, changepoint_count=1)
        manifest_path = self._write_manifest(
            tmp_path,
            {'train': ['train.csv'], 'validation': ['val.csv'], 'test': []},
        )

        results = eda.summarize_skab_manifest(manifest_path)

        assert results['train'].anomaly_count == 2
        assert results['train'].changepoint_count == 1
        assert results['validation'].anomaly_count == 1
        assert results['validation'].changepoint_count == 1

    def test_per_split_plots_available_false_by_default(self, tmp_path):
        eda = _skab_eda()
        self._make_skab_csv(tmp_path / 'a.csv', 2)
        self._make_skab_csv(tmp_path / 'b.csv', 2)
        manifest_path = self._write_manifest(
            tmp_path,
            {'train': ['a.csv'], 'validation': ['b.csv'], 'test': []},
        )

        results = eda.summarize_skab_manifest(manifest_path)

        assert results['train'].plots_available is False
        assert results['validation'].plots_available is False

    def test_manifest_source_reflected_in_result(self, tmp_path):
        eda = _skab_eda()
        self._make_skab_csv(tmp_path / 'a.csv', 2)
        self._make_skab_csv(tmp_path / 'b.csv', 2)
        manifest_path = self._write_manifest(
            tmp_path,
            {'train': ['a.csv'], 'validation': ['b.csv'], 'test': []},
        )

        results = eda.summarize_skab_manifest(manifest_path)

        assert 'manifest' in results['train'].source.lower()
