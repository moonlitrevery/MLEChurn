"""Load YAML training configuration (single source of defaults)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_train_config(path: Path) -> dict[str, Any]:
    """Return parsed YAML dict, or empty dict if file is missing."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw if isinstance(raw, dict) else {}


def deep_get(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
