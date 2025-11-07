"""
Diagnostic: Test SimplePowerSurrogate wrapper before l4casadi export
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from alternative_approach.surrogate_module.model import PowerSurrogate

# SimplePowerSurrogate definition
class SimplePowerSurrogate(torch.nn.Module):
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


def test_wrapper():
    print("="*70)
    print("Testing SimplePowerSurrogate Wrapper")
    print("="*70)

    # Load original model
    checkpoint_path = Path("models/power_surrogate.pth")

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model_config = checkpoint['model_config'].copy()
    model_config.pop('n_parameters', None)

    original_model = PowerSurrogate(**model_config)
    original_model.load_state_dict(checkpoint['model_state_dict'])

    norm = checkpoint['normalization']
    original_model.set_normalization(
        np.array(norm['input_mean']),
        np.array(norm['input_std']),
        np.array(norm['output_mean']),
        np.array(norm['output_std'])
    )
    original_model.eval()

    # Create wrapper
    simple_model = SimplePowerSurrogate(original_model)
    simple_model.eval()

    print("\n✅ Models loaded")
    print(f"Original model normalization:")
    print(f"  input_mean: {original_model.input_mean}")
    print(f"  input_std: {original_model.input_std}")
    print(f"  output_mean: {original_model.output_mean}")
    print(f"  output_std: {original_model.output_std}")

    print(f"\nSimple model normalization:")
    print(f"  input_mean: {simple_model.input_mean}")
    print(f"  input_std: {simple_model.input_std}")
    print(f"  output_mean: {simple_model.output_mean}")
    print(f"  output_std: {simple_model.output_std}")

    # Test cases
    test_cases = [
        np.array([0.0, 0.0, 0.0, 0.0, 8.0, 270.0]),
        np.array([10.0, 5.0, 2.0, 0.0, 8.0, 270.0]),
        np.array([20.0, 15.0, 10.0, 0.0, 10.0, 270.0]),
    ]

    print("\n" + "-"*70)
    print("Testing 3 cases...")
    print("-"*70)

    max_error = 0.0
    all_match = True

    for i, x in enumerate(test_cases):
        x_torch = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            # Original model
            power_original = float(original_model(x_torch).item())

            # Simple wrapper
            power_simple = float(simple_model(x_torch).item())

        error = abs(power_original - power_simple)
        max_error = max(max_error, error)

        match = error < 1.0  # Less than 1 W
        all_match = all_match and match
        status = "✅" if match else "❌"

        print(f"\n{status} Case {i+1}: yaw={x[:4]}, wind={x[4]:.1f}m/s @ {x[5]:.0f}°")
        print(f"  Original:  {power_original/1e6:.6f} MW")
        print(f"  Wrapper:   {power_simple/1e6:.6f} MW")
        print(f"  Error:     {error:.2f} W")

    print("\n" + "="*70)
    print("Summary:")
    print(f"  Max error: {max_error:.2f} W")

    if all_match:
        print("\n✅ SimplePowerSurrogate wrapper works correctly!")
        print("   The issue must be in l4casadi export.")
        return True
    else:
        print("\n❌ SimplePowerSurrogate wrapper has errors!")
        print("   Fix the wrapper before exporting to l4casadi.")
        return False


if __name__ == '__main__':
    success = test_wrapper()
    exit(0 if success else 1)
