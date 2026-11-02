"""YAML configuration loading with simple inheritance (``extends:`` key)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config. A config may contain ``extends: other.yaml``."""
    if path is None:
        path = REPO_ROOT / "configs" / "default.yaml"
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("extends", None)
    if parent:
        parent_cfg = load_config(path.parent / parent)
        cfg = _deep_update(parent_cfg, cfg)
    cfg["_config_path"] = str(path)
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    """Resolve ``cfg['paths'][key]`` relative to the repository root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else REPO_ROOT / p


def grid_from_cfg(cfg: dict, name: str) -> dict[str, list]:
    """Return the hyper-parameter grid of an algorithm (YAML ``null`` → None)."""
    grid = cfg["grids"].get(name, {})
    out = {}
    for k, v in grid.items():
        vals = []
        for x in v:
            if isinstance(x, list):          # hidden_layer_sizes written as lists in YAML
                x = tuple(x)
            vals.append(x)
        out[k] = vals
    return out
