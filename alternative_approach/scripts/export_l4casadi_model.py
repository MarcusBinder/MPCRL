"""
Export PyTorch Model to l4casadi
=================================

Convert trained PyTorch model to CasADi format using l4casadi for use in acados.

Usage:
    python export_l4casadi_model.py --model models/power_surrogate.pth
"""

import argparse
from pathlib import Path
import pickle
import time

import numpy as np
import torch
import casadi as ca

try:
    import l4casadi as l4c
    L4CASADI_AVAILABLE = True
except ImportError:
    L4CASADI_AVAILABLE = False
    print("⚠️  Warning: l4casadi not installed. Install with: pip install l4casadi")

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from surrogate_module.model import PowerSurrogate


def load_model(model_path: str) -> PowerSurrogate:
    """Load trained PyTorch model."""

    print(f"Loading model from {model_path}...")

    checkpoint = torch.load(model_path, map_location='cpu')

    # Create model
    model_config = checkpoint['model_config']
    model = PowerSurrogate(**model_config)

    # Load weights
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
    print(f"  Architecture: {model_config}")

    return model, checkpoint


def export_l4casadi(model: PowerSurrogate, output_path: str):
    """Export model to l4casadi format."""

    if not L4CASADI_AVAILABLE:
        raise ImportError("l4casadi is required. Install with: pip install l4casadi")

    print("\nExporting to l4casadi...")

    # Wrap model with l4casadi
    # Note: model already handles normalization internally
    l4c_model = l4c.L4CasADi(
        model,
        model_expects_batch_dim=True,
        name='power_surrogate'
    )

    # Create CasADi function
    # Input: [yaw_0, yaw_1, yaw_2, yaw_3, wind_speed, wind_direction]
    x = ca.SX.sym('x', 6)

    # Wrap in batch dimension for model
    x_batched = ca.reshape(x, (1, 6))

    # Evaluate
    y_batched = l4c_model(x_batched)

    # Remove batch dimension
    y = ca.reshape(y_batched, (1, 1))

    # Create function
    power_func = ca.Function('power_surrogate', [x], [y])

    print("  ✅ CasADi function created")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump({
            'l4c_model': l4c_model,
            'power_func': power_func,
            'pytorch_model': model,
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
        y_casadi[i] = float(casadi_func(X_test[i]))

    # Compare
    abs_diff = np.abs(y_pytorch - y_casadi)
    rel_diff = abs_diff / (np.abs(y_pytorch) + 1e-8)

    print(f"\n  Results:")
    print(f"    Max absolute difference: {abs_diff.max()/1e3:.2f} kW")
    print(f"    Mean absolute difference: {abs_diff.mean()/1e3:.2f} kW")
    print(f"    Max relative difference: {rel_diff.max()*100:.4f}%")
    print(f"    Mean relative difference: {rel_diff.mean()*100:.4f}%")

    if abs_diff.max() < 1e3:  # Less than 1 kW difference
        print(f"  ✅ Validation passed!")
    else:
        print(f"  ⚠️  Warning: Large differences detected")

    return abs_diff, rel_diff


def benchmark_performance(casadi_func: ca.Function, n_evals: int = 1000):
    """Benchmark CasADi function performance."""

    print("\nBenchmarking performance...")

    # Random inputs
    np.random.seed(42)
    X_test = np.random.randn(n_evals, 6) * 10

    # Warm up
    for i in range(10):
        _ = casadi_func(X_test[i])

    # Benchmark
    start_time = time.time()
    for i in range(n_evals):
        _ = casadi_func(X_test[i])
    elapsed = time.time() - start_time

    avg_time = elapsed / n_evals * 1000  # ms

    print(f"  Average evaluation time: {avg_time:.4f} ms")
    print(f"  Throughput: {n_evals/elapsed:.0f} evaluations/second")

    if avg_time < 1.0:
        print(f"  ✅ Fast enough for real-time MPC (<1ms)")
    else:
        print(f"  ⚠️  Slower than target (want <1ms)")

    return avg_time


def test_gradients(casadi_func: ca.Function):
    """Test gradient computation."""

    print("\nTesting gradients...")

    # Create symbolic input
    x = ca.SX.sym('x', 6)

    # Evaluate function
    y = casadi_func(x)

    # Compute Jacobian
    J = ca.jacobian(y, x)
    jacobian_func = ca.Function('jacobian', [x], [J])

    # Test at a point
    x_test = np.array([10, 5, 5, 0, 8, 270])
    jac = jacobian_func(x_test)

    print(f"  Test point: yaw={x_test[:4]}, ws={x_test[4]}, wd={x_test[5]}")
    print(f"  Jacobian shape: {jac.shape}")
    print(f"  Jacobian: {jac}")

    # Check if gradients are reasonable
    grad_norm = float(ca.norm_2(jac))
    print(f"  Gradient norm: {grad_norm:.2f}")

    if grad_norm > 0 and grad_norm < 1e10:
        print(f"  ✅ Gradients look reasonable")
    else:
        print(f"  ⚠️  Warning: Unusual gradient magnitude")

    return jacobian_func


def main():
    parser = argparse.ArgumentParser(description='Export model to l4casadi')
    parser.add_argument('--model', type=str,
                        default='models/power_surrogate.pth',
                        help='Trained PyTorch model path')
    parser.add_argument('--output', type=str,
                        default='models/power_surrogate_casadi.pkl',
                        help='Output l4casadi model path')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation tests')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run performance benchmark')

    args = parser.parse_args()

    print("=" * 70)
    print("Export PyTorch Model to l4casadi")
    print("=" * 70)

    # Load model
    pytorch_model, checkpoint = load_model(args.model)

    # Export
    casadi_func = export_l4casadi(pytorch_model, args.output)

    # Validate
    if args.validate:
        validate_export(pytorch_model, casadi_func)

    # Test gradients
    jacobian_func = test_gradients(casadi_func)

    # Benchmark
    if args.benchmark:
        benchmark_performance(casadi_func)

    print("\n" + "=" * 70)
    print("✅ Export complete!")
    print("=" * 70)

    print("\nNext steps:")
    print("  1. Use in nonlinear MPC:")
    print(f"     from nmpc_surrogate import SurrogateMPC")
    print(f"     controller = SurrogateMPC(model_path='{args.output}')")


if __name__ == '__main__':
    main()
