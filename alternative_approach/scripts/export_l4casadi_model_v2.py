"""
Export trained surrogate model to l4casadi format (Version 2 - Fixed)
=======================================================================

This version creates a simple wrapper without conditional logic
to ensure TorchScript can properly trace the normalization.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import casadi as ca
import pickle
import time

# Try to import l4casadi
try:
    import l4casadi as l4c
    L4CASADI_AVAILABLE = True
except ImportError:
    L4CASADI_AVAILABLE = False
    print("Warning: l4casadi not available. Install with: pip install l4casadi")

from alternative_approach.surrogate_module.model import PowerSurrogate


class SimplePowerSurrogate(nn.Module):
    """
    Simplified wrapper for l4casadi export.

    Removes conditional logic and explicitly applies normalization
    in every forward pass. This ensures TorchScript can properly trace it.
    """

    def __init__(self, original_model: PowerSurrogate):
        super().__init__()

        # Copy the network
        self.network = original_model.network

        # Copy normalization parameters as regular tensors (not buffers)
        # This ensures they are traced by TorchScript
        self.input_mean = original_model.input_mean.clone()
        self.input_std = original_model.input_std.clone()
        self.output_mean = original_model.output_mean.clone()
        self.output_std = original_model.output_std.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple forward pass without conditionals.
        Always normalizes input and denormalizes output.
        """
        # Normalize input: (x - mean) / std
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)

        # Forward through network
        y_norm = self.network(x_norm)

        # Denormalize output: y * std + mean
        y = y_norm * self.output_std + self.output_mean

        return y


def load_model(checkpoint_path: str):
    """Load trained model from checkpoint."""

    print(f"Loading model from {checkpoint_path}...")

    # Check if checkpoint exists
    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        print(f"\n❌ ERROR: Checkpoint not found at {checkpoint_path}")
        print("\nYou need to train the model first!")
        print("\nRun these steps in order:")
        print("  1. python scripts/generate_dataset_v2.py      # Generate training data")
        print("  2. python scripts/train_surrogate_v2.py       # Train model")
        print("  3. python scripts/export_l4casadi_model_v2.py # Export (you are here)")
        print("\nSee FULL_PIPELINE.md for detailed instructions.")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create original model
    model_config = checkpoint['model_config'].copy()
    model_config.pop('n_parameters', None)

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

    # Create simple wrapper
    simple_model = SimplePowerSurrogate(model)
    simple_model.eval()

    print(f"  ✅ Model loaded and wrapped")
    print(f"  Parameters: {model.count_parameters():,}")

    return simple_model, checkpoint


def validate_wrapper(original_model: PowerSurrogate, simple_model: SimplePowerSurrogate, n_tests: int = 100):
    """Validate that simple wrapper matches original model."""

    print("\nValidating wrapper...")

    # Generate random test inputs
    np.random.seed(42)
    yaw_test = np.random.uniform(-30, 30, (n_tests, 4))
    ws_test = np.random.uniform(6, 12, n_tests)
    wd_test = np.random.uniform(260, 280, n_tests)
    X_test = np.column_stack([yaw_test, ws_test, wd_test])

    # Test
    with torch.no_grad():
        X_torch = torch.tensor(X_test, dtype=torch.float32)

        y_original = original_model(X_torch).numpy()
        y_simple = simple_model(X_torch).numpy()

    # Compare
    abs_diff = np.abs(y_original - y_simple)
    rel_diff = abs_diff / (np.abs(y_original) + 1e-8)

    print(f"  Max absolute difference: {abs_diff.max()/1e3:.2f} kW")
    print(f"  Mean absolute difference: {abs_diff.mean()/1e3:.2f} kW")

    if abs_diff.max() < 1:  # Less than 1 W
        print(f"  ✅ Wrapper matches original perfectly!")
        return True
    else:
        print(f"  ❌ Warning: Wrapper doesn't match original")
        return False


def export_l4casadi(model: SimplePowerSurrogate, output_path: str):
    """Export simple wrapper model to l4casadi format."""

    if not L4CASADI_AVAILABLE:
        raise ImportError("l4casadi is required. Install with: pip install l4casadi")

    print("\nExporting to l4casadi...")

    # Wrap model with l4casadi
    l4c_model = l4c.L4CasADi(model, name='power_surrogate')

    # Create CasADi function
    x = ca.SX.sym('x', 6)
    y_raw = l4c_model(x)
    y = y_raw[0] if y_raw.shape[0] > 0 else y_raw
    power_func = ca.Function('power_surrogate', [x], [y])

    print("  ✅ CasADi function created")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump({
            'l4c_model': l4c_model,
            'power_func': power_func,
            'pytorch_model': model,  # Save the simple wrapper
        }, f)

    print(f"  ✅ Saved to {output_path}")

    return power_func


def validate_export(pytorch_model: SimplePowerSurrogate, casadi_func: ca.Function, n_tests: int = 100):
    """Validate that CasADi function matches PyTorch wrapper."""

    print("\nValidating export...")
    print(f"  Testing {n_tests} random samples...")

    # Generate random test inputs
    np.random.seed(42)
    yaw_test = np.random.uniform(-30, 30, (n_tests, 4))
    ws_test = np.random.uniform(6, 12, n_tests)
    wd_test = np.random.uniform(260, 280, n_tests)
    X_test = np.column_stack([yaw_test, ws_test, wd_test])

    # PyTorch predictions
    with torch.no_grad():
        X_torch = torch.tensor(X_test, dtype=torch.float32)
        y_pytorch = pytorch_model(X_torch).numpy()

    # CasADi predictions
    y_casadi = np.zeros((n_tests, 1))
    for i in range(n_tests):
        result = casadi_func(X_test[i])
        y_casadi[i] = np.array(result).flatten()[0]

    # Compare
    abs_diff = np.abs(y_pytorch - y_casadi)
    rel_diff = abs_diff / (np.abs(y_pytorch) + 1e-8)

    print(f"\n  Results:")
    print(f"    Max absolute difference: {abs_diff.max()/1e3:.2f} kW")
    print(f"    Mean absolute difference: {abs_diff.mean()/1e3:.2f} kW")
    print(f"    Max relative difference: {rel_diff.max()*100:.4f}%")
    print(f"    Mean relative difference: {rel_diff.mean()*100:.4f}%")

    if abs_diff.max() < 1e3:  # Less than 1 kW
        print(f"  ✅ Validation passed!")
        return True
    else:
        print(f"  ❌ Warning: Large differences detected")
        return False


def main():
    """Main export script."""

    print("="*70)
    print("L4CasADi Model Export (V2 - Fixed)")
    print("="*70)

    # Paths
    checkpoint_path = 'checkpoints/power_surrogate_best.ckpt'
    output_path = 'models/power_surrogate_casadi.pkl'

    # Load
    simple_model, checkpoint = load_model(checkpoint_path)

    # Load original model for validation
    original_model_config = checkpoint['model_config'].copy()
    original_model_config.pop('n_parameters', None)
    original_model = PowerSurrogate(**original_model_config)
    original_model.load_state_dict(checkpoint['model_state_dict'])
    norm = checkpoint['normalization']
    original_model.set_normalization(
        np.array(norm['input_mean']),
        np.array(norm['input_std']),
        np.array(norm['output_mean']),
        np.array(norm['output_std'])
    )
    original_model.eval()

    # Validate wrapper matches original
    validate_wrapper(original_model, simple_model)

    # Export
    power_func = export_l4casadi(simple_model, output_path)

    # Validate export
    validate_export(simple_model, power_func)

    print("\n" + "="*70)
    print("✅ Export complete!")
    print("="*70)
    print(f"\nSaved to: {output_path}")
    print("\nNext steps:")
    print("  1. Run: python validate_normalization.py")
    print("  2. Run: python nmpc_surrogate_casadi.py")


if __name__ == '__main__':
    main()
