"""
Export ONLY the neural network (no normalization) to l4casadi
==============================================================

Strategy:
1. Export only model.network (the nn.Sequential part)
2. Save normalization parameters separately
3. MPC handles normalization manually

This avoids TorchScript normalization tracing issues!
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
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


def load_model(checkpoint_path: str = "models/power_surrogate.pth"):
    """Load trained model from checkpoint."""

    print(f"Loading model from {checkpoint_path}...")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"\n❌ ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model_config = checkpoint['model_config'].copy()
    model_config.pop('n_parameters', None)

    model = PowerSurrogate(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])

    norm = checkpoint['normalization']
    model.set_normalization(
        np.array(norm['input_mean']),
        np.array(norm['input_std']),
        np.array(norm['output_mean']),
        np.array(norm['output_std'])
    )

    model.eval()

    print(f"  ✅ Model loaded")

    # Extract just the network (without normalization wrapper)
    network = model.network

    # Extract normalization parameters
    normalization = {
        'input_mean': model.input_mean.cpu().numpy(),
        'input_std': model.input_std.cpu().numpy(),
        'output_mean': float(model.output_mean.cpu().item()),
        'output_std': float(model.output_std.cpu().item()),
    }

    return network, normalization


def export_l4casadi(network: torch.nn.Module, normalization: dict, output_path: str):
    """Export network to l4casadi."""

    if not L4CASADI_AVAILABLE:
        raise ImportError("l4casadi is required. Install with: pip install l4casadi")

    print("\nExporting to l4casadi...")
    print("  Exporting ONLY the network (no normalization)")
    print("  Normalization will be handled manually in MPC")

    # Create a wrapper that handles batching properly
    class NetworkWrapper(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, x):
            # Ensure x has batch dimension
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Add batch dimension: (6,) -> (1, 6)
            elif x.dim() == 2 and x.shape[0] != 1:
                # If shape is (6, 1), transpose to (1, 6)
                if x.shape[0] == 6 and x.shape[1] == 1:
                    x = x.T

            # Forward through network
            y = self.network(x)

            # Ensure output is 2D: (1, 1) as required by l4casadi
            if y.dim() == 1:
                y = y.unsqueeze(-1)  # (batch,) -> (batch, 1)
            if y.dim() == 0:
                y = y.reshape(1, 1)  # scalar -> (1, 1)

            return y

    wrapper = NetworkWrapper(network)
    wrapper.eval()

    # Wrap with l4casadi
    l4c_model = l4c.L4CasADi(wrapper, name='power_network')

    # Create CasADi function
    # Input: NORMALIZED [yaw_0, yaw_1, yaw_2, yaw_3, wind_speed, wind_direction]
    # Output: NORMALIZED power
    x_norm = ca.SX.sym('x_norm', 6)
    y_norm = l4c_model(x_norm)

    # Extract scalar if needed
    if hasattr(y_norm, 'shape') and y_norm.shape[0] > 1:
        y = y_norm[0]
    else:
        y = y_norm

    network_func = ca.Function('power_network', [x_norm], [y])

    print("  ✅ CasADi function created")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump({
            'network_func': network_func,
            'normalization': normalization,
            'l4c_model': l4c_model,
            'network_wrapper': wrapper,  # Keep reference
        }, f)

    print(f"  ✅ Saved to {output_path}")

    return network_func, normalization, wrapper


def validate_export(network: torch.nn.Module, network_func: ca.Function,
                   normalization: dict, wrapper=None, n_tests: int = 100):
    """Validate that CasADi matches PyTorch."""

    print("\nValidating export...")
    print(f"  Testing {n_tests} random NORMALIZED samples...")

    # Use wrapper if provided, otherwise use network directly
    eval_model = wrapper if wrapper is not None else network

    # Generate random NORMALIZED inputs (mean=0, std=1)
    np.random.seed(42)
    X_norm_test = np.random.randn(n_tests, 6) * 0.5  # Keep within reasonable range

    # PyTorch predictions (on normalized input)
    with torch.no_grad():
        X_torch = torch.tensor(X_norm_test, dtype=torch.float32)
        y_norm_pytorch = eval_model(X_torch).numpy()
        if y_norm_pytorch.ndim == 1:
            y_norm_pytorch = y_norm_pytorch.reshape(-1, 1)

    # CasADi predictions (on normalized input)
    y_norm_casadi = np.zeros((n_tests, 1))
    for i in range(n_tests):
        result = network_func(X_norm_test[i])
        y_norm_casadi[i] = np.array(result).flatten()[0]

    # Compare NORMALIZED outputs
    abs_diff = np.abs(y_norm_pytorch - y_norm_casadi)
    rel_diff = abs_diff / (np.abs(y_norm_pytorch) + 1e-8)

    print(f"\n  Results (on normalized data):")
    print(f"    Max absolute difference: {abs_diff.max():.6f}")
    print(f"    Mean absolute difference: {abs_diff.mean():.6f}")
    print(f"    Max relative difference: {rel_diff.max()*100:.4f}%")
    print(f"    Mean relative difference: {rel_diff.mean()*100:.4f}%")

    # Also test on RAW inputs (with manual normalization)
    print(f"\n  Testing {n_tests} random RAW samples (with manual normalization)...")

    yaw_test = np.random.uniform(-30, 30, (n_tests, 4))
    ws_test = np.random.uniform(6, 12, n_tests)
    wd_test = np.random.uniform(260, 280, n_tests)
    X_raw_test = np.column_stack([yaw_test, ws_test, wd_test])

    input_mean = normalization['input_mean']
    input_std = normalization['input_std']
    output_mean = normalization['output_mean']
    output_std = normalization['output_std']

    # PyTorch: manual normalize, predict, manual denormalize
    X_norm_pytorch = (X_raw_test - input_mean) / (input_std + 1e-8)
    with torch.no_grad():
        X_torch = torch.tensor(X_norm_pytorch, dtype=torch.float32)
        y_norm = eval_model(X_torch).numpy()
        if y_norm.ndim == 1:
            y_norm = y_norm.reshape(-1, 1)
        y_pytorch = y_norm * output_std + output_mean

    # CasADi: manual normalize, predict, manual denormalize
    y_casadi = np.zeros((n_tests, 1))
    for i in range(n_tests):
        x_norm = (X_raw_test[i] - input_mean) / (input_std + 1e-8)
        y_norm = np.array(network_func(x_norm)).flatten()[0]
        y_casadi[i] = y_norm * output_std + output_mean

    # Compare RAW outputs (in MW)
    abs_diff_raw = np.abs(y_pytorch - y_casadi)
    rel_diff_raw = abs_diff_raw / (np.abs(y_pytorch) + 1e-8)

    print(f"\n  Results (on raw data with manual norm/denorm):")
    print(f"    Max absolute difference: {abs_diff_raw.max()/1e3:.2f} kW")
    print(f"    Mean absolute difference: {abs_diff_raw.mean()/1e3:.2f} kW")
    print(f"    Max relative difference: {rel_diff_raw.max()*100:.4f}%")
    print(f"    Mean relative difference: {rel_diff_raw.mean()*100:.4f}%")

    if abs_diff_raw.max() < 1e3:  # Less than 1 kW
        print(f"  ✅ Validation passed!")
        return True
    else:
        print(f"  ❌ Warning: Large differences detected")
        return False


def main():
    """Main export script."""

    print("="*70)
    print("Network-Only Export (Manual Normalization)")
    print("="*70)

    checkpoint_path = 'models/power_surrogate.pth'
    output_path = 'models/power_surrogate_casadi.pkl'

    # Load
    network, normalization = load_model(checkpoint_path)

    print(f"\nNormalization parameters:")
    print(f"  input_mean: {normalization['input_mean']}")
    print(f"  input_std: {normalization['input_std']}")
    print(f"  output_mean: {normalization['output_mean']:.1f}")
    print(f"  output_std: {normalization['output_std']:.1f}")

    # Export
    network_func, normalization, wrapper = export_l4casadi(network, normalization, output_path)

    # Validate
    validate_export(network, network_func, normalization, wrapper=wrapper)

    print("\n" + "="*70)
    print("✅ Export complete!")
    print("="*70)
    print(f"\nSaved to: {output_path}")
    print("\nIMPORTANT: MPC must handle normalization manually:")
    print("  1. Normalize inputs: x_norm = (x - mean) / std")
    print("  2. Call network: y_norm = network(x_norm)")
    print("  3. Denormalize output: y = y_norm * std + mean")
    print("\nNext steps:")
    print("  1. Run: python validate_normalization.py")
    print("  2. Run: python nmpc_surrogate_casadi.py")


if __name__ == '__main__':
    main()
