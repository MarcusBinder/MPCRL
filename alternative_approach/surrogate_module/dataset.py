from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

import numpy as np

ALT_ROOT = Path(__file__).resolve().parents[1]
if str(ALT_ROOT) not in sys.path:
    sys.path.insert(0, str(ALT_ROOT))

from nmpc_windfarm_acados_fixed import Farm, build_pywake_model, pywake_farm_power


def build_default_farm() -> Farm:
    """Return the canonical 4-turbine farm layout used across examples."""
    D = 178.0
    x = np.array([0.0, 7 * D, 14 * D, 21 * D])
    y = np.zeros_like(x)
    return Farm(x=x, y=y, D=D)


def generate_power_dataset(
    n_samples: int,
    farm: Farm,
    wind_speed_range: Tuple[float, float],
    wind_direction_range: Tuple[float, float],
    yaw_range: Tuple[float, float],
    turbulence: float = 0.06,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Sample yaw configurations and compute the corresponding PyWake farm power.

    Features are ordered as [yaw_t0, ..., yaw_t{N-1}, wind_speed, wind_direction].
    Targets contain a single column: total farm power in Watts.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if wind_speed_range[0] <= 0:
        raise ValueError("Wind speed must be positive")

    rng = np.random.default_rng(seed)
    N_turbines = len(farm.x)

    feature_names = [f"yaw_t{i}" for i in range(N_turbines)] + [
        "wind_speed",
        "wind_direction",
    ]
    X = np.zeros((n_samples, len(feature_names)), dtype=np.float64)
    y = np.zeros((n_samples,), dtype=np.float64)

    wf_model, layout = build_pywake_model(farm.x, farm.y, farm.D, ti=turbulence)

    yaw_min, yaw_max = yaw_range
    ws_min, ws_max = wind_speed_range
    wd_min, wd_max = wind_direction_range

    for i in range(n_samples):
        yaw = rng.uniform(yaw_min, yaw_max, size=N_turbines)
        wind_speed = rng.uniform(ws_min, ws_max)
        wind_dir = rng.uniform(wd_min, wd_max)
        wind_dir_mod = np.mod(wind_dir, 360.0)

        power = pywake_farm_power(
            wf_model,
            layout,
            wind_speed,
            wind_dir_mod,
            yaw,
        )

        X[i, :N_turbines] = yaw
        X[i, -2] = wind_speed
        X[i, -1] = wind_dir_mod
        y[i] = power

    metadata = {
        "n_samples": n_samples,
        "features": feature_names,
        "target": "farm_power",
        "farm": {
            "x": farm.x.tolist(),
            "y": farm.y.tolist(),
            "D": farm.D,
        },
        "wind_speed_range": list(wind_speed_range),
        "wind_direction_range": list(wind_direction_range),
        "yaw_range": list(yaw_range),
        "turbulence": turbulence,
        "seed": seed,
    }

    return {
        "features": X,
        "targets": y,
        "feature_names": np.array(feature_names),
        "metadata": metadata,
    }


def save_dataset_npz(dataset: Dict[str, np.ndarray], output_path: Path) -> None:
    """Persist dataset to NPZ plus JSON metadata for reproducibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=dataset["features"],
        targets=dataset["targets"],
        feature_names=dataset["feature_names"],
    )
    meta_path = output_path.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(dataset["metadata"], fh, indent=2)


def load_dataset_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load dataset saved via save_dataset_npz."""
    data = np.load(path, allow_pickle=False)
    meta_path = path.with_suffix(".json")
    meta = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
    return {
        "features": data["features"],
        "targets": data["targets"],
        "feature_names": data["feature_names"],
        "metadata": meta,
    }
