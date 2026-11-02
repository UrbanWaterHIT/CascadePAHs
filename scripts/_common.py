"""Shared command-line handling for the numbered scripts."""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*number of unique classes.*")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cascadepahs.config import load_config  # noqa: E402


def get_config(description: str = ""):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(ROOT / "configs" / "fast.yaml"),
                   help="YAML configuration (configs/fast.yaml for a demo, configs/default.yaml for the full protocol)")
    args = p.parse_args()
    cfg = load_config(args.config)
    print(f"[config] {cfg['_config_path']}  tune={cfg.get('tune')}  n_jobs={cfg.get('n_jobs')}")
    return cfg


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.time()
        print(f"[start] {self.label}", flush=True)
        return self

    def __exit__(self, *a):
        print(f"[done ] {self.label} ({time.time() - self.t:.0f} s)", flush=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=float)
