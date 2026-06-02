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
    def _make_skab_csv(self, path: Path, rows: list[dict[str, object]]):
        header = [
            'datetime',
            *SENSOR_COLUMNS,
            'anomaly',
            'changepoint',
        ]
        lines = [';'.join(header)]
        for row in rows:
            values = [str(row.get(col, '')) for col in header]
            lines.append(';'.join(values))
        path.write_text('\n'.join(lines) + '\n')

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

    def test_result_has_missing_rates_per_column(self):
        eda = _skab_eda()

        result = eda.summarize_skab_csv(SKAB_FIXTURE)

        assert isinstance(result.missing_rates, dict)
        assert all(result.missing_rates.get(col, 1.0) == 0.0 for col in SENSOR_COLUMNS)
        assert result.missing_rates.get('datetime') == 0.0

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

    def test_sensor_statistics_include_research_quantiles(self):
        eda = _skab_eda()

        result = eda.summarize_skab_csv(SKAB_FIXTURE)

        stats = result.sensor_statistics['Accelerometer1RMS']
        for key in ('p01', 'p05', 'p25', 'p50', 'p75', 'p95', 'p99'):
            assert key in stats
        assert stats['p50'] == pytest.approx(0.11)

    def test_result_has_constant_column_flags(self, tmp_path):
        eda = _skab_eda()
        csv_path = tmp_path / 'constant.csv'
        self._make_skab_csv(
            csv_path,
            [
                {
                    'datetime': '2024-01-01T00:00:00Z',
                    **{col: 1.0 for col in SENSOR_COLUMNS},
                    'anomaly': 0,
                    'changepoint': 0,
                },
                {
                    'datetime': '2024-01-01T00:00:01Z',
                    **{col: 1.0 for col in SENSOR_COLUMNS},
                    'anomaly': 0,
                    'changepoint': 0,
                },
            ],
        )

        result = eda.summarize_skab_csv(csv_path)

        assert all(result.constant_column_flags[col] is True for col in SENSOR_COLUMNS)
        assert result.constant_column_flags['datetime'] is False

    def test_result_has_timestamp_quality_for_valid_cadenced_fixture(self):
        eda = _skab_eda()

        result = eda.summarize_skab_csv(SKAB_FIXTURE)

        assert result.timestamp_quality == {
            'column': 'datetime',
            'present': True,
            'row_count': 3,
            'parsed_datetime_count': 3,
            'invalid_timestamp_count': 0,
            'duplicate_timestamp_count': 0,
            'is_monotonic_increasing': True,
            'is_monotonic_decreasing': False,
            'is_sorted_ascending': True,
            'first_valid_timestamp': '2024-01-01T00:00:00Z',
            'last_valid_timestamp': '2024-01-01T00:00:02Z',
            'cadence_mode_seconds': 1.0,
            'max_gap_seconds': 1.0,
        }

    def test_timestamp_quality_counts_invalid_duplicate_and_unsorted_rows(self, tmp_path):
        eda = _skab_eda()
        csv_path = tmp_path / 'bad_timestamps.csv'
        base_sensors = {col: 1.0 for col in SENSOR_COLUMNS}
        self._make_skab_csv(
            csv_path,
            [
                {'datetime': '2024-01-01T00:00:02Z', **base_sensors, 'anomaly': 0, 'changepoint': 0},
                {'datetime': 'not-a-date', **base_sensors, 'anomaly': 0, 'changepoint': 0},
                {'datetime': '2024-01-01T00:00:01Z', **base_sensors, 'anomaly': 0, 'changepoint': 0},
                {'datetime': '2024-01-01T00:00:01Z', **base_sensors, 'anomaly': 0, 'changepoint': 0},
            ],
        )

        result = eda.summarize_skab_csv(csv_path)

        assert result.timestamp_quality['parsed_datetime_count'] == 3
        assert result.timestamp_quality['invalid_timestamp_count'] == 1
        assert result.timestamp_quality['duplicate_timestamp_count'] == 1
        assert result.timestamp_quality['is_monotonic_increasing'] is False
        assert result.timestamp_quality['is_sorted_ascending'] is False
        assert result.timestamp_quality['cadence_mode_seconds'] == 1.0

    def test_result_separates_anomaly_ranges_and_changepoints(self):
        eda = _skab_eda()

        result = eda.summarize_skab_csv(SKAB_FIXTURE)

        assert result.label_ranges['anomaly'] == [
            {
                'start_index': 1,
                'end_index': 1,
                'length': 1,
                'start_timestamp': '2024-01-01T00:00:01Z',
                'end_timestamp': '2024-01-01T00:00:01Z',
            }
        ]
        assert result.label_ranges['changepoint'] == [
            {
                'start_index': 2,
                'end_index': 2,
                'length': 1,
                'start_timestamp': '2024-01-01T00:00:02Z',
                'end_timestamp': '2024-01-01T00:00:02Z',
            }
        ]
        assert result.changepoint_timestamps == ['2024-01-01T00:00:02Z']

    def test_result_has_correlation_distribution_rolling_and_overlay_data(self):
        eda = _skab_eda()

        result = eda.summarize_skab_csv(SKAB_FIXTURE)

        assert set(result.correlation_matrix.keys()) == set(SENSOR_COLUMNS)
        assert set(result.correlation_matrix['Accelerometer1RMS'].keys()) == set(SENSOR_COLUMNS)
        assert result.correlation_matrix['Accelerometer1RMS']['Current'] == pytest.approx(1.0)

        distribution = result.sensor_distributions['Accelerometer1RMS']
        assert distribution['count'] == 3
        assert distribution['quantiles']['p50'] == pytest.approx(0.11)
        assert distribution['iqr'] == pytest.approx(0.01)

        rolling = result.rolling_statistics['Accelerometer1RMS']
        assert rolling['window'] == 3
        assert rolling['rolling_mean_last'] == pytest.approx(0.11)
        assert rolling['rolling_std_max'] == pytest.approx(0.01)

        assert len(result.label_overlay) == 3
        assert result.label_overlay[1]['label'] == 'anomaly'
        assert result.label_overlay[1]['Accelerometer1RMS'] == pytest.approx(0.11)
        assert result.label_overlay[2]['label'] == 'changepoint'

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

    def test_writes_research_grade_csv_and_json_artifacts(self, tmp_path):
        import csv
        import json

        eda = _skab_eda()

        result = eda.summarize_skab_csv(SKAB_FIXTURE)
        eda.write_skab_eda_report(result, tmp_path)

        sensor_stats_path = tmp_path / 'sensor_statistics.csv'
        missingness_path = tmp_path / 'missingness.csv'
        timestamp_quality_path = tmp_path / 'timestamp_quality.json'
        label_ranges_path = tmp_path / 'label_ranges.csv'
        correlation_path = tmp_path / 'correlation_matrix.csv'
        distributions_path = tmp_path / 'sensor_distributions.csv'
        rolling_path = tmp_path / 'rolling_statistics.csv'
        overlay_path = tmp_path / 'label_overlay.csv'

        assert sensor_stats_path.exists()
        assert missingness_path.exists()
        assert timestamp_quality_path.exists()
        assert label_ranges_path.exists()
        assert correlation_path.exists()
        assert distributions_path.exists()
        assert rolling_path.exists()
        assert overlay_path.exists()

        sensor_rows = list(csv.DictReader(sensor_stats_path.open(newline='')))
        assert sensor_rows[0]['sensor'] == 'Accelerometer1RMS'
        assert sensor_rows[0]['p50'] == '0.11'

        missing_rows = list(csv.DictReader(missingness_path.open(newline='')))
        assert missing_rows[0] == {'column': 'datetime', 'missing_count': '0', 'missing_rate': '0.0'}

        timestamp_quality = json.loads(timestamp_quality_path.read_text())
        assert timestamp_quality['cadence_mode_seconds'] == 1.0

        label_rows = list(csv.DictReader(label_ranges_path.open(newline='')))
        assert label_rows == [
            {
                'label': 'anomaly',
                'start_index': '1',
                'end_index': '1',
                'length': '1',
                'start_timestamp': '2024-01-01T00:00:01Z',
                'end_timestamp': '2024-01-01T00:00:01Z',
            },
            {
                'label': 'changepoint',
                'start_index': '2',
                'end_index': '2',
                'length': '1',
                'start_timestamp': '2024-01-01T00:00:02Z',
                'end_timestamp': '2024-01-01T00:00:02Z',
            },
        ]

        correlation_rows = list(csv.DictReader(correlation_path.open(newline='')))
        assert len(correlation_rows) == len(SENSOR_COLUMNS)
        assert set(correlation_rows[0]) == {'sensor', *SENSOR_COLUMNS}
        assert float(correlation_rows[0]['Current']) == pytest.approx(1.0)

        distribution_rows = list(csv.DictReader(distributions_path.open(newline='')))
        assert distribution_rows[0]['sensor'] == 'Accelerometer1RMS'
        assert distribution_rows[0]['p50'] == '0.11'
        assert float(distribution_rows[0]['iqr']) == pytest.approx(0.01)

        rolling_rows = list(csv.DictReader(rolling_path.open(newline='')))
        assert rolling_rows[0]['sensor'] == 'Accelerometer1RMS'
        assert rolling_rows[0]['window'] == '3'
        assert float(rolling_rows[0]['rolling_mean_last']) == pytest.approx(0.11)

        overlay_rows = list(csv.DictReader(overlay_path.open(newline='')))
        assert overlay_rows[1]['label'] == 'anomaly'
        assert overlay_rows[2]['label'] == 'changepoint'

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

        assert artifacts['input_report_md'] == tmp_path / 'report.md'
        assert artifacts['input_summary_json'] == tmp_path / 'summary.json'
        assert artifacts['input_sensor_statistics_csv'] == tmp_path / 'sensor_statistics.csv'
        assert artifacts['input_missingness_csv'] == tmp_path / 'missingness.csv'
        assert artifacts['input_timestamp_quality_json'] == tmp_path / 'timestamp_quality.json'
        assert artifacts['input_label_ranges_csv'] == tmp_path / 'label_ranges.csv'
        assert artifacts['input_correlation_matrix_csv'] == tmp_path / 'correlation_matrix.csv'
        assert artifacts['input_sensor_distributions_csv'] == tmp_path / 'sensor_distributions.csv'
        assert artifacts['input_rolling_statistics_csv'] == tmp_path / 'rolling_statistics.csv'
        assert artifacts['input_label_overlay_csv'] == tmp_path / 'label_overlay.csv'
        assert 'input_label_overlay_plot_json' not in artifacts
        assert 'input_label_overlay_plot_html' not in artifacts
        assert 'input_correlation_heatmap_png' not in artifacts
        assert (tmp_path / 'summary.json').exists()
        assert (tmp_path / 'report.md').exists()
        assert (tmp_path / 'sensor_statistics.csv').exists()
        assert (tmp_path / 'missingness.csv').exists()
        assert (tmp_path / 'timestamp_quality.json').exists()
        assert (tmp_path / 'label_ranges.csv').exists()
        assert (tmp_path / 'correlation_matrix.csv').exists()
        assert (tmp_path / 'sensor_distributions.csv').exists()
        assert (tmp_path / 'rolling_statistics.csv').exists()
        assert (tmp_path / 'label_overlay.csv').exists()
        assert not (tmp_path / 'correlation_heatmap.png').exists()

    def test_include_plots_true_writes_optional_overlay_plot_artifacts(self, tmp_path):
        import json

        eda = _skab_eda()

        artifacts = eda.generate_skab_eda_report(
            input_path=SKAB_FIXTURE,
            split_manifest_path=None,
            output_dir=tmp_path,
            include_plots=True,
        )

        assert artifacts['input_label_overlay_plot_json'] == tmp_path / 'label_overlay_plot.json'
        assert artifacts['input_label_overlay_plot_html'] == tmp_path / 'label_overlay_plot.html'
        assert artifacts['input_correlation_heatmap_png'] == tmp_path / 'correlation_heatmap.png'
        assert artifacts['plots_note'] == 'plots available'

        plot_json_path = tmp_path / 'label_overlay_plot.json'
        plot_html_path = tmp_path / 'label_overlay_plot.html'
        heatmap_path = tmp_path / 'correlation_heatmap.png'
        assert plot_json_path.exists()
        assert plot_html_path.exists()
        assert heatmap_path.exists()

        plot_payload = json.loads(plot_json_path.read_text())
        assert [trace['name'] for trace in plot_payload['data']] == [
            'Accelerometer1RMS',
            'anomaly',
            'changepoint',
        ]
        assert plot_payload['layout']['title']['text'] == 'SKAB anomaly/changepoint overlay'
        assert 'generated_at' not in plot_json_path.read_text()
        assert 'Plotly.newPlot' in plot_html_path.read_text()

        first_heatmap_bytes = heatmap_path.read_bytes()
        assert first_heatmap_bytes.startswith(b'\x89PNG\r\n\x1a\n')
        assert len(first_heatmap_bytes) > 0

        result = eda.summarize_skab_csv(SKAB_FIXTURE)
        eda.write_skab_eda_report(result, tmp_path, include_plots=True)
        assert heatmap_path.read_bytes() == first_heatmap_bytes

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
        assert artifacts['train_sensor_statistics_csv'] == tmp_path / 'eda' / 'train' / 'sensor_statistics.csv'
        assert artifacts['train_correlation_matrix_csv'] == tmp_path / 'eda' / 'train' / 'correlation_matrix.csv'
        assert artifacts['validation_report_md'] == tmp_path / 'eda' / 'validation' / 'report.md'
        assert artifacts['validation_timestamp_quality_json'] == tmp_path / 'eda' / 'validation' / 'timestamp_quality.json'
        assert artifacts['validation_label_overlay_csv'] == tmp_path / 'eda' / 'validation' / 'label_overlay.csv'
        assert artifacts['test_summary_json'] == tmp_path / 'eda' / 'test' / 'summary.json'
        assert artifacts['test_label_ranges_csv'] == tmp_path / 'eda' / 'test' / 'label_ranges.csv'
        assert artifacts['test_rolling_statistics_csv'] == tmp_path / 'eda' / 'test' / 'rolling_statistics.csv'
        assert (tmp_path / 'eda' / 'summary.json').exists()
        assert (tmp_path / 'eda' / 'train' / 'summary.json').exists()
        assert (tmp_path / 'eda' / 'train' / 'sensor_statistics.csv').exists()
        assert (tmp_path / 'eda' / 'train' / 'correlation_matrix.csv').exists()
        assert (tmp_path / 'eda' / 'validation' / 'report.md').exists()
        assert (tmp_path / 'eda' / 'validation' / 'timestamp_quality.json').exists()
        assert (tmp_path / 'eda' / 'validation' / 'label_overlay.csv').exists()
        assert (tmp_path / 'eda' / 'test' / 'summary.json').exists()
        assert (tmp_path / 'eda' / 'test' / 'label_ranges.csv').exists()
        assert (tmp_path / 'eda' / 'test' / 'rolling_statistics.csv').exists()


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

    def test_manifest_results_include_split_level_aggregation(self, tmp_path):
        eda = _skab_eda()
        self._make_skab_csv(tmp_path / 'a.csv', 4, anomaly_count=2, changepoint_count=1)
        self._make_skab_csv(tmp_path / 'b.csv', 6, anomaly_count=1, changepoint_count=1)
        manifest_path = self._write_manifest(
            tmp_path,
            {'train': ['a.csv', 'b.csv'], 'validation': ['a.csv'], 'test': []},
        )

        with pytest.raises(ValueError, match='duplicate file across splits'):
            eda.summarize_skab_manifest(manifest_path)

        manifest_path = self._write_manifest(
            tmp_path,
            {'train': ['a.csv', 'b.csv'], 'validation': [], 'test': []},
        )

        with pytest.raises(ValueError, match='validation split must be non-empty'):
            eda.summarize_skab_manifest(manifest_path)

        self._make_skab_csv(tmp_path / 'val.csv', 2)
        manifest_path = self._write_manifest(
            tmp_path,
            {'train': ['a.csv', 'b.csv'], 'validation': ['val.csv'], 'test': []},
        )

        results = eda.summarize_skab_manifest(manifest_path)

        split_summary = results['train'].split_aggregation
        assert split_summary['split'] == 'train'
        assert split_summary['file_count'] == 2
        assert split_summary['row_count'] == 10
        assert split_summary['anomaly_count'] == 3
        assert split_summary['changepoint_count'] == 2
        assert split_summary['source_files'] == [str(tmp_path / 'a.csv'), str(tmp_path / 'b.csv')]
        assert split_summary['rows_by_file'] == {str(tmp_path / 'a.csv'): 4, str(tmp_path / 'b.csv'): 6}
