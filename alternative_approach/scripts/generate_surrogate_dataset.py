#!/usr/bin/env python3
"""
Generate a surrogate-training dataset by sampling yaw configurations
and evaluating farm power with PyWake.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
import importlib.util

import numpy as np

ALT_ROOT = Path(__file__).resolve().parents[1]
SURROGATE_PATH = ALT_ROOT / "surrogate_module" / "dataset.py"
spec = importlib.util.spec_from_file_location(
    "alternative_surrogate.dataset", SURROGATE_PATH
)
dataset_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(dataset_module)

build_default_farm = dataset_module.build_default_farm
generate_power_dataset = dataset_module.generate_power_dataset
save_dataset_npz = dataset_module.save_dataset_npz


def parse_range(value: str, name: str) -> Tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"{name} must be provided as 'min,max' (got '{value}')"
        )
    low, high = map(float, parts)
    if low > high:
        raise argparse.ArgumentTypeError(f"{name} requires min ≤ max (got '{value}')")
    return low, high


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/surrogate/power_dataset.npz"),
        help="Output NPZ file (metadata saved alongside as JSON).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Number of random yaw samples to generate.",
    )
    parser.add_argument(
        "--yaw-range",
        type=lambda v: parse_range(v, "yaw-range"),
        default=(-25.0, 25.0),
        help="Yaw bounds in degrees as 'min,max'.",
    )
    parser.add_argument(
        "--wind-speed",
        type=lambda v: parse_range(v, "wind-speed"),
        default=(6.0, 12.0),
        help="Wind speed range in m/s as 'min,max'.",
    )
    parser.add_argument(
        "--wind-direction",
        type=lambda v: parse_range(v, "wind-direction"),
        default=(0.0, 0.0),
        help="Wind direction in meteorological degrees as 'min,max'.",
    )
    parser.add_argument(
        "--turbulence",
        type=float,
        default=0.06,
        help="Ambient turbulence intensity used in PyWake.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    farm = build_default_farm()
    dataset = generate_power_dataset(
        n_samples=args.n_samples,
        farm=farm,
        wind_speed_range=args.wind_speed,
        wind_direction_range=args.wind_direction,
        yaw_range=args.yaw_range,
        turbulence=args.turbulence,
        seed=args.seed,
    )
    save_dataset_npz(dataset, args.output)

    print(f"Wrote dataset with {args.n_samples} samples to {args.output}")
    print(f"Feature order: {', '.join(dataset['feature_names'])}")
    print(
        "Metadata path:",
        args.output.with_suffix(".json"),
    )


if __name__ == "__main__":
    main()
