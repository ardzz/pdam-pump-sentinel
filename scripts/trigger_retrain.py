from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from routemq.queue import dispatch  # type: ignore[reportMissingImports]

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.jobs.retraining_job import RetrainingJob  # noqa: E402


async def _dispatch() -> None:
    await dispatch(RetrainingJob())


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Dispatch a PumpAD retraining job.')
    parser.parse_args(argv)
    asyncio.run(_dispatch())


if __name__ == '__main__':
    main()
