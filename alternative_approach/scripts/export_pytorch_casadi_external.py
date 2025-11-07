"""
Export trained surrogate model as CasADi external function (no l4casadi)
=========================================================================

This approach:
- Wraps PyTorch model as CasADi external function
- Evaluates PyTorch at each call (slower but guaranteed correct)
- No TorchScript tracing issues!
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import casadi as ca
import pickle

from alternative_approach.surrogate_module.model import PowerSurrogate


class PyTorchCasADiExternal:
    """
    Wrapper to use PyTorch model as CasADi external function.

    Evaluates PyTorch directly - slower but guaranteed correct.
    """

    def __init__(self, pytorch_model):
        self.pytorch_model = pytorch_model
        self.pytorch_model.eval()

    def evaluate(self, x):
        """Evaluate PyTorch model (called by CasADi)."""
        # x is numpy array [yaw_0, yaw_1, yaw_2, yaw_3, wind_speed, wind_direction]
        x_torch = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y_torch = self.pytorch_model(x_torch)
        return float(y_torch.item())

    def create_casadi_function(self):
        """Create CasADi function that calls PyTorch."""

        # Create symbolic input
        x_sym = ca.MX.sym('x', 6)

        # Create external function
        # This will call self.evaluate() at each evaluation
        power_func = ca.Function('power_surrogate', [x_sym], [x_sym[0]])  # Placeholder

        # We'll use callback instead
        return self._create_callback_function()

    def _create_callback_function(self):
        """Create CasADi callback function."""

        class PyTorchCallback(ca.Callback):
            def __init__(self, pytorch_wrapper, name='pytorch_callback'):
                ca.Callback.__init__(self)
                self.pytorch_wrapper = pytorch_wrapper
                self.construct(name, {'enable_fd': False})

            def get_n_in(self): return 1
            def get_n_out(self): return 1

            def get_sparsity_in(self, i):
                return ca.Sparsity.dense(6, 1)

            def get_sparsity_out(self, i):
                return ca.Sparsity.dense(1, 1)

            def eval(self, arg):
                # arg[0] is the input vector
                x = np.array(arg[0]).flatten()
                y = self.pytorch_wrapper.evaluate(x)
                return [y]

        callback = PyTorchCallback(self)
        x = ca.MX.sym('x', 6)
        return ca.Function('power_surrogate', [x], [callback(x)])


def load_model(checkpoint_path: str = "models/power_surrogate.pth"):
    """Load trained model from checkpoint."""

    print(f"Loading model from {checkpoint_path}...")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"\n❌ ERROR: Checkpoint not found at {checkpoint_path}")
        print("\nTrain the model first:")
        print("  python scripts/train_surrogate_v2.py --max_epochs 100")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model config
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

    print(f"  ✅ Model loaded")
    print(f"  Parameters: {model.count_parameters():,}")

    return model


def export_casadi_external(model: PowerSurrogate, output_path: str):
    """Export model as CasADi external function."""

    print("\nExporting to CasADi external function...")

    # Create wrapper
    pytorch_casadi = PyTorchCasADiExternal(model)

    # Create CasADi function
    power_func = pytorch_casadi.create_callback_function()

    print("  ✅ CasADi function created")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump({
            'pytorch_model': model,
            'power_func': power_func,
            'pytorch_casadi': pytorch_casadi,  # Keep reference
        }, f)

    print(f"  ✅ Saved to {output_path}")

    return power_func


def validate_export(pytorch_model: PowerSurrogate, casadi_func: ca.Function, n_tests: int = 100):
    """Validate that CasADi function matches PyTorch model."""

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
        y_casadi[i] = float(np.array(result).flatten()[0])

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
    print("PyTorch → CasADi External Function Export")
    print("="*70)
    print("\nThis approach:")
    print("  - Uses PyTorch model directly (no TorchScript)")
    print("  - Evaluates PyTorch at each CasADi call")
    print("  - Slower but guaranteed correct!")
    print("="*70)

    # Paths
    checkpoint_path = 'models/power_surrogate.pth'
    output_path = 'models/power_surrogate_casadi.pkl'

    # Load
    model = load_model(checkpoint_path)

    # Export
    power_func = export_casadi_external(model, output_path)

    # Validate
    validate_export(model, power_func)

    print("\n" + "="*70)
    print("✅ Export complete!")
    print("="*70)
    print(f"\nSaved to: {output_path}")
    print("\nNext steps:")
    print("  1. Run: python validate_normalization.py")
    print("  2. Run: python nmpc_surrogate_casadi.py")
    print("\nNote: This uses PyTorch callbacks (slower than l4casadi)")
    print("      But it's guaranteed to match PyTorch exactly!")


if __name__ == '__main__':
    main()
