"""
Quick test: Does the surrogate give reasonable predictions?
"""

import numpy as np
import pickle
import torch

# Load surrogate
print("Loading surrogate...")
with open('models/power_surrogate_casadi.pkl', 'rb') as f:
    data = pickle.load(f)

pytorch_model = data['pytorch_model']
power_func = data['power_func']

print("Testing surrogate predictions...")

# Test a few yaw angles
test_cases = [
    {"yaw": [0, 0, 0, 0], "ws": 8.0, "wd": 270.0, "expected": "~5 MW (baseline)"},
    {"yaw": [10, 10, 10, 0], "ws": 8.0, "wd": 270.0, "expected": "slightly higher"},
    {"yaw": [20, 20, 20, 0], "ws": 8.0, "wd": 270.0, "expected": "much higher (~15% gain)"},
    {"yaw": [-10, -10, -10, 0], "ws": 8.0, "wd": 270.0, "expected": "slightly higher"},
]

print("\nPyTorch model predictions:")
for i, case in enumerate(test_cases):
    yaw = case["yaw"]
    ws = case["ws"]
    wd = case["wd"]

    x = torch.tensor(yaw + [ws, wd], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        power = pytorch_model(x).item()

    print(f"{i}. Yaw={yaw}, Power={power/1e6:.3f} MW ({case['expected']})")

print("\nCasADi model predictions:")
for i, case in enumerate(test_cases):
    yaw = case["yaw"]
    ws = case["ws"]
    wd = case["wd"]

    x = np.array(yaw + [ws, wd])
    power_ca = float(np.array(power_func(x)).flatten()[0])

    print(f"{i}. Yaw={yaw}, Power={power_ca/1e6:.3f} MW ({case['expected']})")

print("\n" + "="*70)
print("Analysis:")
print("  - Baseline (0°) should be ~5 MW")
print("  - Optimal (~20°) should be ~5.7 MW (+15%)")
print("  - If predictions are way off, normalization is broken")
print("="*70)
