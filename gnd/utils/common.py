"""Shared utilities: seeding, device selection, config I/O, result bookkeeping."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
FIGURE_DIR = REPO_ROOT / "figures"
CONFIG_DIR = REPO_ROOT / "configs"


def set_seed(seed: int) -> None:
    """Seed every RNG we touch.  Also pins torch to deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(prefer: str = "cpu") -> torch.device:
    """Resolve a device string.

    We default to CPU: the models here are small (<1M parameters) and the
    gauge field relies on ``torch.matrix_exp`` / ``torch.linalg.lstsq``,
    neither of which is implemented for the MPS backend.
    """
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str | Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_jsonable(obj), fh, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def aggregate_seeds(records: list[dict]) -> dict:
    """Collapse a list of per-seed metric dicts into mean / sem / n.

    Non-numeric entries are dropped.  Returns ``{key: {"mean", "sem", "std",
    "n", "values"}}``.
    """
    keys: set[str] = set()
    for rec in records:
        keys |= {k for k, v in rec.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    out: dict[str, dict] = {}
    for key in sorted(keys):
        vals = np.array(
            [rec[key] for rec in records if isinstance(rec.get(key), (int, float))],
            dtype=float,
        )
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        out[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "sem": float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0,
            "n": int(vals.size),
            "values": vals.tolist(),
        }
    return out


def fmt_mean_sem(entry: Mapping[str, float] | None, digits: int = 3) -> str:
    """LaTeX-ready ``mean ± sem`` string."""
    if entry is None:
        return "--"
    return f"{entry['mean']:.{digits}f} $\\pm$ {entry['sem']:.{digits}f}"


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
