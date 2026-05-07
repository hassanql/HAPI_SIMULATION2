"""End-to-end smoke test. Run after a fresh checkout to confirm the project is healthy.

Equivalent to:
  pytest -q  &&  python run.py --stage bootstrap

Useful in CI and as a one-line "did everything install correctly?" check.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent

    print("==> Running unit tests")
    result = subprocess.run(["pytest", "-q"], cwd=repo_root)
    if result.returncode != 0:
        print("FAIL: unit tests did not pass.")
        return result.returncode

    print("==> Running bootstrap stage")
    result = subprocess.run([sys.executable, "run.py", "--stage", "bootstrap"], cwd=repo_root)
    if result.returncode != 0:
        print("FAIL: bootstrap stage did not pass.")
        return result.returncode

    print("==> Validation complete: project is healthy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
