from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

def main(argv: Sequence[str] | None = None) -> dict[str, Path]:
    from ml.visualization.pca_artifacts import (
        PcaArtifactVisualizationConfig,
        generate_pca_artifact_visualizations,
    )

    parser = argparse.ArgumentParser(description='Generate PCA artifact visualizations from training outputs.')
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--no-test', action='store_true')
    args = parser.parse_args(argv)

    config = PcaArtifactVisualizationConfig(
        artifact_dir=args.artifact_dir,
        output_dir=args.output_dir,
        include_test=not args.no_test,
    )

    artifacts = generate_pca_artifact_visualizations(config)
    print(json.dumps({name: str(path) for name, path in sorted(artifacts.items())}, indent=2))
    return artifacts


if __name__ == '__main__':
    main()
