import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nmpc_windfarm_acados_fixed import build_pywake_model, pywake_farm_power


def evaluate_layout():
    D = 178.0
    spacing = 6 * D
    x = np.array([0.0, spacing, 2 * spacing, 3 * spacing])
    y = np.zeros_like(x)
    wf, layout = build_pywake_model(x, y, D, ti=0.06)

    combos = []
    for yaw in np.linspace(0, 25, 6):
        combos.append(np.array([yaw, yaw, yaw, 0.0]))

    results = []
    for yaw in combos:
        power = pywake_farm_power(wf, layout, 10.0, 270.0, yaw) / 1e6
        results.append((yaw, power))

    baseline = pywake_farm_power(wf, layout, 10.0, 270.0, np.zeros(4)) / 1e6
    return baseline, results


if __name__ == "__main__":
    baseline, results = evaluate_layout()
    print(f"Baseline (0° yaw): {baseline:.3f} MW")
    for yaw, power in results:
        yaw_str = ", ".join(f"{v:.1f}" for v in yaw)
        delta = (power - baseline) * 1e3
        print(f"Yaw [{yaw_str}] => {power:.3f} MW (Δ {delta:+.1f} kW)")
