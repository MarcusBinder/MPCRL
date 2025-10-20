#!/usr/bin/env python3
"""
Train a neural-network surrogate on a generated power dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence
import importlib.util

ALT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ALT_ROOT / "surrogate_module" / "training.py"
spec = importlib.util.spec_from_file_location(
    "alternative_surrogate.training", TRAIN_PATH
)
training_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(training_module)

train_surrogate = training_module.train_surrogate


def parse_hidden_layers(value: str) -> Sequence[int]:
    try:
        layers = [int(v.strip()) for v in value.split(",") if v.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("hidden-layers must be comma-separated integers") from exc
    if not layers:
        raise argparse.ArgumentTypeError("hidden-layers cannot be empty")
    if any(width <= 0 for width in layers):
        raise argparse.ArgumentTypeError("hidden layer sizes must be positive")
    return layers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/surrogate/power_dataset.npz"),
        help="Path to the NPZ dataset generated via generate_surrogate_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/surrogate"),
        help="Directory where trained model artifacts will be saved.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=parse_hidden_layers,
        default=(128, 128, 128),
        help="Comma-separated hidden layer sizes, e.g. '128,128,64'.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="tanh",
        choices=("tanh", "relu", "softplus"),
        help="Activation function for hidden layers.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="L2 regularisation strength.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of data used for validation.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Fraction of data used for hold-out testing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Training device, e.g. 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data splits.",
    )
    args = parser.parse_args()

    result = train_surrogate(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        hidden_layers=args.hidden_layers,
        activation=args.activation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        device=args.device,
    )

    print("Training complete.")
    print(f"Model weights: {result['model_path']}")
    print(f"Exported NPZ: {result['weights_path']}")
    print(f"Stats JSON:  {result['stats_path']}")
    metrics = result["metrics"]
    print(f"Test MAE (W): {metrics['mae_W']:.2f}")
    print(f"Test MAPE    : {metrics['mape']*100:.3f}%")


if __name__ == "__main__":
    main()
