"""JSON output with NaN -> null and numpy -> lists."""
import json

import numpy as np


def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(to_jsonable(data), f)
    print("written", path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def nan2none(a):
    # rows of a 2D/3D array -> nested lists, with None for every entry that is not entirely finite (seed-grid format)
    return [[None if not np.all(np.isfinite(v)) else (v.tolist() if np.ndim(v) else float(v)) for v in row] for row in a]
