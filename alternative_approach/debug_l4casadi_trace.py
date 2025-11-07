"""
Debug: What is l4casadi actually tracing?
"""

import torch
import numpy as np
import pickle
from pathlib import Path

# Load the model
model_path = Path('models/power_surrogate_casadi.pkl')
with open(model_path, 'rb') as f:
    data = pickle.load(f)

pytorch_model = data['pytorch_model']
power_func = data['power_func']

print("="*70)
print("Debugging l4casadi Trace")
print("="*70)

# Check normalization buffers
print("\nNormalization buffers in PyTorch model:")
print(f"  input_mean: {pytorch_model.input_mean}")
print(f"  input_std: {pytorch_model.input_std}")
print(f"  output_mean: {pytorch_model.output_mean}")
print(f"  output_std: {pytorch_model.output_std}")

# Test with a simple case
x_raw = np.array([0.0, 0.0, 0.0, 0.0, 8.0, 270.0])

print(f"\nTest input (raw): {x_raw}")

# PyTorch prediction
with torch.no_grad():
    x_torch = torch.tensor(x_raw, dtype=torch.float32).unsqueeze(0)

    # Try calling forward directly
    power_pytorch = float(pytorch_model(x_torch).item())
    print(f"\nPyTorch forward(): {power_pytorch/1e6:.4f} MW")

    # Try calling with normalized=True (should give wrong result)
    try:
        power_pytorch_norm = float(pytorch_model(x_torch, normalized=True).item())
        print(f"PyTorch forward(normalized=True): {power_pytorch_norm/1e6:.4f} MW (should be wrong)")
    except:
        print("Cannot call with normalized=True")

    # Manual normalization test
    x_normalized = pytorch_model.normalize_input(x_torch)
    print(f"\nManually normalized input: {x_normalized[0].numpy()}")

    y_network = pytorch_model.network(x_normalized)
    print(f"Network output (normalized): {y_network[0].item():.6f}")

    y_denormalized = pytorch_model.denormalize_output(y_network)
    print(f"Denormalized output: {y_denormalized[0].item()/1e6:.4f} MW")

# CasADi prediction
power_casadi = float(np.array(power_func(x_raw)).flatten()[0])
print(f"\nCasADi function: {power_casadi/1e6:.4f} MW")

# Compare
error = abs(power_casadi - power_pytorch)
print(f"\nError: {error/1e6:.4f} MW ({error/power_pytorch*100:.1f}%)")

# Theory: l4casadi might be tracing with normalized=False but not capturing the buffers
print("\n" + "="*70)
print("HYPOTHESIS:")
print("l4casadi might be tracing the model but not properly capturing")
print("the normalize_input() and denormalize_output() operations")
print("because they use register_buffer tensors.")
print("="*70)

# Check if we can trace manually with TorchScript
print("\nTrying manual TorchScript trace...")
try:
    traced_model = torch.jit.trace(pytorch_model, x_torch)
    power_traced = float(traced_model(x_torch).item())
    print(f"TorchScript traced model: {power_traced/1e6:.4f} MW")

    # Check the traced graph
    print("\nTraced graph:")
    print(traced_model.code)
except Exception as e:
    print(f"Error tracing: {e}")

print("\nDone!")
