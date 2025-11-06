"""
Test Surrogate Model Accuracy
==============================

Validate trained surrogate model against PyWake on test set.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt

from surrogate_module.model import PowerSurrogate
from nmpc_windfarm_acados_fixed import build_pywake_model, pywake_farm_power


def load_model(model_path: str) -> PowerSurrogate:
    """Load trained model."""
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint['model_config']
    model = PowerSurrogate(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set normalization
    norm = checkpoint['normalization']
    model.set_normalization(
        np.array(norm['input_mean']),
        np.array(norm['input_std']),
        np.array(norm['output_mean']),
        np.array(norm['output_std'])
    )

    model.eval()
    return model


def test_accuracy(model: PowerSurrogate, test_dataset_path: str, n_samples: int = 1000):
    """Test model accuracy on test set."""

    print("=" * 70)
    print("Surrogate Model Accuracy Test")
    print("=" * 70)

    # Load test data
    print(f"\nLoading test data from {test_dataset_path}...")
    with h5py.File(test_dataset_path, 'r') as f:
        yaw = f['yaw'][:n_samples]
        wind_speed = f['wind_speed'][:n_samples]
        wind_direction = f['wind_direction'][:n_samples]
        power_true = f['power'][:n_samples]

    # Prepare inputs
    X = np.column_stack([yaw, wind_speed, wind_direction])
    X_torch = torch.tensor(X, dtype=torch.float32)

    # Predict
    print("Predicting with surrogate model...")
    with torch.no_grad():
        power_pred = model(X_torch).numpy().flatten()

    # Compute metrics
    errors = power_pred - power_true
    abs_errors = np.abs(errors)
    rel_errors = abs_errors / (np.abs(power_true) + 1e-8)

    mae = abs_errors.mean()
    rmse = np.sqrt((errors ** 2).mean())
    max_error = abs_errors.max()

    mean_power = power_true.mean()
    mae_percent = mae / mean_power * 100

    # R² score
    ss_res = ((power_true - power_pred) ** 2).sum()
    ss_tot = ((power_true - power_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    # Print results
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"\nTested on {n_samples:,} samples")
    print(f"\nAccuracy Metrics:")
    print(f"  MAE:  {mae/1e3:.2f} kW ({mae_percent:.3f}%)")
    print(f"  RMSE: {rmse/1e3:.2f} kW")
    print(f"  Max error: {max_error/1e3:.2f} kW")
    print(f"  R² score: {r2:.6f}")

    # Check if meets targets
    print(f"\n Target Achievement:")
    if mae_percent < 1.0:
        print(f"  ✅ MAE < 1% of mean power")
    else:
        print(f"  ❌ MAE > 1% (target: <1%)")

    if r2 > 0.99:
        print(f"  ✅ R² > 0.99")
    else:
        print(f"  ⚠️  R² < 0.99 (target: >0.99)")

    # Plot
    plot_results(power_true, power_pred, errors)

    return {
        'mae': mae,
        'rmse': rmse,
        'mae_percent': mae_percent,
        'r2': r2
    }


def plot_results(y_true, y_pred, errors):
    """Plot prediction results."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Parity plot
    ax = axes[0]
    ax.scatter(y_true / 1e6, y_pred / 1e6, alpha=0.3, s=1)
    ax.plot([y_true.min() / 1e6, y_true.max() / 1e6],
            [y_true.min() / 1e6, y_true.max() / 1e6], 'r--', label='Perfect')
    ax.set_xlabel('True Power (MW)')
    ax.set_ylabel('Predicted Power (MW)')
    ax.set_title('Parity Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error distribution
    ax = axes[1]
    ax.hist(errors / 1e3, bins=50, alpha=0.7)
    ax.axvline(0, color='r', linestyle='--', label='Zero error')
    ax.set_xlabel('Prediction Error (kW)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Relative error distribution
    ax = axes[2]
    rel_errors = np.abs(errors) / (np.abs(y_true) + 1e-8) * 100
    ax.hist(rel_errors, bins=50, alpha=0.7)
    ax.axvline(1.0, color='r', linestyle='--', label='1% threshold')
    ax.set_xlabel('Relative Error (%)')
    ax.set_ylabel('Count')
    ax.set_title('Relative Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('surrogate_accuracy_test.png', dpi=150)
    print(f"\n  ✅ Plot saved to surrogate_accuracy_test.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='models/power_surrogate.pth')
    parser.add_argument('--test_dataset', type=str,
                        default='data/surrogate_dataset_test.h5')
    parser.add_argument('--n_samples', type=int, default=1000)

    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    # Test
    results = test_accuracy(model, args.test_dataset, args.n_samples)

    print("\n" + "=" * 70)
    if results['mae_percent'] < 1.0 and results['r2'] > 0.99:
        print("✅ Surrogate model meets accuracy targets!")
    else:
        print("⚠️  Surrogate model needs improvement")
    print("=" * 70)


if __name__ == '__main__':
    main()
