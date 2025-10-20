from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import casadi as ca
import numpy as np

from .training import NormalizationStats


def load_weights(npz_path: Path) -> Dict[str, np.ndarray]:
    """Load a weight NPZ exported by export_state_dict."""
    data = np.load(npz_path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def _activation(expr: ca.SX, name: str) -> ca.SX:
    if name == "tanh":
        return ca.tanh(expr)
    if name == "relu":
        return ca.fmax(expr, 0)
    if name == "softplus":
        return ca.log(1 + ca.exp(expr))
    raise ValueError(f"Unsupported activation '{name}'")


def build_casadi_mlp(
    weights: Dict[str, np.ndarray],
    activation: str,
    feature_stats: NormalizationStats,
    target_stats: NormalizationStats,
    input_name: str = "psi_features",
) -> Tuple[ca.SX, ca.SX]:
    """
    Construct a CasADi computational graph equivalent to the trained MLP.

    Returns the input SX symbol and the scalar output expression (denormalised power).
    """
    n_features = feature_stats.mean.size
    x = ca.SX.sym(input_name, n_features)

    z = (x - feature_stats.mean) / feature_stats.std

    # Extract linear layer parameters in order
    layer_indices: List[int] = sorted(
        {
            int(name.split(".")[1])
            for name in weights.keys()
            if name.endswith("weight")
        }
    )

    current = z
    for idx in layer_indices:
        weight = weights[f"net.{idx}.weight"]
        bias = weights[f"net.{idx}.bias"]
        current = ca.mtimes(weight, current) + bias
        is_last = idx == layer_indices[-1]
        if not is_last:
            current = _activation(current, activation)

    y_norm = current
    power = y_norm * target_stats.std[0] + target_stats.mean[0]
    return x, power
