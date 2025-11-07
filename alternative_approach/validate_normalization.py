"""
Quick validation: Check if CasADi function matches PyTorch model predictions.
"""

import numpy as np
import torch
import casadi as ca
import pickle
from pathlib import Path

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

    pytorch_model = data['pytorch_model']
    power_func = data['power_func']

    print(f"\nLoaded PyTorch model and CasADi function")
    print(f"  Both should take raw inputs and return raw outputs")

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
