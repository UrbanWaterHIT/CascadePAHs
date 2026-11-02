"""Run every numbered script in order.

    python scripts/run_all.py --config configs/fast.yaml          # demo (≈ 1.5–2 h on 2 cores)
    python scripts/run_all.py --config configs/default.yaml       # full protocol (many hours)
    python scripts/run_all.py --config configs/fast.yaml --only 04 09
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(HERE.parent / "configs" / "fast.yaml"))
    p.add_argument("--only", nargs="*", help="script number prefixes to run, e.g. 04 09")
    p.add_argument("--skip-data", action="store_true", help="skip the demo-data step (00)")
    a = p.parse_args()
    scripts = sorted(s for s in HERE.glob("[0-9][0-9]_*.py"))
    if a.only:
        scripts = [s for s in scripts if s.name[:2] in a.only]
    if a.skip_data:
        scripts = [s for s in scripts if not s.name.startswith("00")]
    for s in scripts:
        t = time.time()
        print(f"\n{'=' * 78}\n{s.name}\n{'=' * 78}", flush=True)
        r = subprocess.run([sys.executable, str(s), "--config", a.config])
        if r.returncode != 0:
            sys.exit(f"{s.name} failed with exit code {r.returncode}")
        print(f"[{s.name}] finished in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
