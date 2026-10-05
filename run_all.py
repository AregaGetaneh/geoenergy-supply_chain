"""Entry point. Draws every published figure and writes the supplementary tables.

    python run_all.py           draw everything
    python run_all.py --check   draw everything, then verify against the published output
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="compare the drawn figures with the published versions")
    args = ap.parse_args()

    import figures

    start = time.time()
    for fn in figures.PUBLISHED:
        fn()
        print("  drew %s" % fn.__name__)
    print("completed in %.0f s" % (time.time() - start))
    print("figures and tables are in %s" % (ROOT / "outputs"))

    if args.check:
        import verify
        return verify.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
