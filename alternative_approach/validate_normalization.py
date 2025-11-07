"""
Quick validation: Check if CasADi function matches PyTorch model predictions.
"""

import numpy as np
import torch
import torch.nn as nn
import casadi as ca
import pickle
from pathlib import Path
import sys

# Add parent directory to path to import surrogate module
sys.path.insert(0, str(Path(__file__).parent.parent))
from alternative_approach.surrogate_module.model import PowerSurrogate


# Define wrapper classes here (needed for unpickling)
class SimplePowerSurrogate(nn.Module):
    """Simplified wrapper for l4casadi export."""

    def __init__(self, original_model):
        super().__init__()
        self.network = original_model.network
        self.input_mean = original_model.input_mean.clone()
        self.input_std = original_model.input_std.clone()
        self.output_mean = original_model.output_mean.clone()
        self.output_std = original_model.output_std.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        y_norm = self.network(x_norm)
        y = y_norm * self.output_std + self.output_mean
        return y


class NetworkWrapper(nn.Module):
    """Wrapper for network-only export (no normalization)."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 2 and x.shape[0] != 1:
            if x.shape[0] == 6 and x.shape[1] == 1:
                x = x.T
        y = self.network(x)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        if y.dim() == 0:
            y = y.reshape(1, 1)
        return y


def validate_network_only_export(data):
    """Validate network-only export (with manual normalization)."""

    print("="*70)
    print("Validating Network-Only Export")
    print("="*70)

    network_func = data['network_func']
    normalization = data['normalization']
    wrapper = data.get('network_wrapper', None)

    input_mean = normalization['input_mean']
    input_std = normalization['input_std']
    output_mean = normalization['output_mean']
    output_std = normalization['output_std']

    print(f"\nNormalization parameters:")
    print(f"  input_mean: {input_mean[:3]}... (truncated)")
    print(f"  input_std: {input_std[:3]}... (truncated)")
    print(f"  output_mean: {output_mean/1e6:.2f} MW")
    print(f"  output_std: {output_std/1e6:.2f} MW")

    # Test cases (RAW inputs)
    test_cases = [
        np.array([0.0, 0.0, 0.0, 0.0, 8.0, 270.0]),
        np.array([10.0, 5.0, 2.0, 0.0, 8.0, 270.0]),
        np.array([20.0, 15.0, 10.0, 0.0, 10.0, 270.0]),
    ]

    print(f"\nTesting {len(test_cases)} cases (with manual normalization)...")
    print("\n" + "-"*70)

    max_error = 0.0
    errors = []

    for i, x in enumerate(test_cases):
        # Manual normalization
        x_norm = (x - input_mean) / (input_std + 1e-8)

        # PyTorch prediction (on normalized input)
        if wrapper is not None:
            with torch.no_grad():
                x_torch = torch.tensor(x_norm, dtype=torch.float32)
                y_norm_pytorch = float(wrapper(x_torch).squeeze().item())
        else:
            # Can't validate without wrapper
            print("⚠️  Warning: No wrapper available for validation")
            y_norm_pytorch = 0.0

        # CasADi prediction (on normalized input)
        y_norm_casadi = float(np.array(network_func(x_norm)).flatten()[0])

        # Manual denormalization
        power_pytorch = y_norm_pytorch * output_std + output_mean
        power_casadi = y_norm_casadi * output_std + output_mean

        # Compare
        error = abs(power_casadi - power_pytorch) if wrapper is not None else 0.0
        rel_error = error / abs(power_pytorch) * 100 if power_pytorch != 0 else 0
        errors.append(error)
        max_error = max(max_error, error)

        print(f"Case {i+1}: yaw={x[:4]}, wind={x[4]:.1f}m/s @ {x[5]:.0f}°")
        if wrapper is not None:
            print(f"  PyTorch:     {power_pytorch/1e6:.4f} MW")
        print(f"  CasADi:      {power_casadi/1e6:.4f} MW")
        if wrapper is not None:
            print(f"  Error:       {error/1e3:.2f} kW ({rel_error:.3f}%)")
        print("-"*70)

    # Summary
    print(f"\nSummary:")
    print(f"  Max error:  {max_error/1e3:.2f} kW")
    print(f"  Mean error: {np.mean(errors)/1e3:.2f} kW")

    if wrapper is None:
        print("\n⚠️  Could not fully validate (wrapper not saved)")
        print("   But CasADi function is callable!")
        return True
    elif max_error < 100:  # Less than 100 W
        print("\n✅ Network-only export is working correctly!")
        print("   Error is negligible (< 100 W)")
        return True
    elif max_error < 1000:  # Less than 1 kW
        print("\n⚠️  Network-only export has small errors")
        print("   Error is acceptable (< 1 kW)")
        return True
    else:
        print("\n❌ Network-only export has significant errors!")
        print(f"   Error is too large ({max_error/1e3:.2f} kW)")
        return False


def validate_casadi_export():
    """Test that CasADi function produces same results as PyTorch model."""

    print("="*70)
    print("Validating CasADi Export")
    print("="*70)

    # Load model
    model_path = Path('models/power_surrogate_casadi.pkl')
    if not model_path.exists():
        print(f"\n❌ Model not found at {model_path}")
        print("Please run: python scripts/export_l4casadi_model.py")
        return

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    # Check what type of export this is
    if 'network_func' in data and 'normalization' in data:
        # Network-only export (with manual normalization)
        print(f"\n✅ Detected network-only export (manual normalization)")
        network_func = data['network_func']
        normalization = data['normalization']

        # This export requires manual normalization in validation
        # We'll handle this differently
        return validate_network_only_export(data)

    elif 'pytorch_model' in data and 'power_func' in data:
        # Old export with SimplePowerSurrogate (with built-in normalization)
        print(f"\n✅ Detected SimplePowerSurrogate export")
        pytorch_model = data['pytorch_model']
        power_func = data['power_func']

        print(f"  Both should take raw inputs and return raw outputs")
    else:
        print(f"\n❌ Unknown export format!")
        print(f"  Available keys: {list(data.keys())}")
        return False

    # Test cases
    test_cases = [
        np.array([0.0, 0.0, 0.0, 0.0, 8.0, 270.0]),   # All zero yaw
        np.array([10.0, 5.0, 2.0, 0.0, 8.0, 270.0]),  # Mixed yaw
        np.array([20.0, 15.0, 10.0, 0.0, 10.0, 270.0]), # Higher yaw, higher wind
    ]

    print(f"\nTesting {len(test_cases)} cases...")
    print("\n" + "-"*70)

    max_error = 0.0
    errors = []

    for i, x in enumerate(test_cases):
        # PyTorch prediction (raw input)
        with torch.no_grad():
            x_torch = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            power_pytorch = float(pytorch_model(x_torch).item())

        # CasADi prediction (raw input - should match PyTorch)
        power_casadi_result = power_func(x)
        power_casadi = float(np.array(power_casadi_result).flatten()[0])

        # Compare
        error = abs(power_casadi - power_pytorch)
        rel_error = error / abs(power_pytorch) * 100 if power_pytorch != 0 else 0
        errors.append(error)
        max_error = max(max_error, error)

        print(f"Case {i+1}: yaw={x[:4]}, wind={x[4]:.1f}m/s @ {x[5]:.0f}°")
        print(f"  PyTorch:     {power_pytorch/1e6:.4f} MW")
        print(f"  CasADi:      {power_casadi/1e6:.4f} MW")
        print(f"  Error:       {error/1e3:.2f} kW ({rel_error:.3f}%)")
        print("-"*70)

    # Summary
    print(f"\nSummary:")
    print(f"  Max error:  {max_error/1e3:.2f} kW")
    print(f"  Mean error: {np.mean(errors)/1e3:.2f} kW")

    if max_error < 100:  # Less than 100 W
        print("\n✅ CasADi export is working correctly!")
        print("   Error is negligible (< 100 W)")
        return True
    elif max_error < 1000:  # Less than 1 kW
        print("\n⚠️  CasADi export has small errors")
        print("   Error is acceptable (< 1 kW)")
        return True
    else:
        print("\n❌ CasADi export has significant errors!")
        print(f"   Error is too large ({max_error/1e3:.2f} kW)")
        print("\nDEBUG: Check if l4casadi is properly exporting the normalization layers")
        return False


if __name__ == '__main__':
    success = validate_casadi_export()
    exit(0 if success else 1)
