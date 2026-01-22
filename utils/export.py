from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
import numpy as np


def find_repo_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "src").exists():
            return cand
    return p


def make_run_id(prefix: str) -> str:
    # microseconds -> no colisiones al re-ejecutar
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def ensure_out_dirs(root: Path) -> tuple[Path, Path]:
    fig_dir = root / "figures"
    res_dir = root / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, res_dir


def save_json(path: Path, obj: dict) -> None:
    def default(o):
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")



def save_npz(path: Path, **arrays) -> None:
    np.savez_compressed(path, **arrays)

__all__ = [
    "find_repo_root",
    "make_run_id",
    "ensure_out_dirs",
    "save_json",
    "save_npz",
]
