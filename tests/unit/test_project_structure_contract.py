import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_current_package_roots_remain_explicit_before_src_migration():
    pyproject = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))

    assert pyproject['tool']['hatch']['build']['targets']['wheel']['packages'] == ['app', 'bootstrap', 'ml']
    assert not (ROOT / 'src').exists()


def test_current_top_level_layout_keeps_runtime_boundaries():
    expected_directories = {
        'app',
        'bootstrap',
        'dashboard',
        'docs',
        'infra',
        'ml',
        'scripts',
        'tests',
    }

    for directory in expected_directories:
        assert (ROOT / directory).is_dir(), f'missing top-level directory: {directory}'

    bootstrap = (ROOT / 'bootstrap' / 'app.py').read_text(encoding='utf-8')
    assert "create_dynamic_router('app.routers')" in bootstrap
    assert "router_directory='app.routers'" in bootstrap
    assert "discover_and_register_jobs('app.jobs')" in bootstrap


def test_readme_documents_the_current_layout_contract():
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')

    assert '## Struktur Project' in readme
    assert 'app/            # RouteMQ application' in readme
    assert 'ml/             # ML/MLOps domain' in readme
    assert 'bootstrap/      # RouteMQ bootstrap entry' in readme
    assert 'src/' not in readme
