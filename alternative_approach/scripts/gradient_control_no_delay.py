import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nmpc_windfarm_acados_fixed import (
    build_pywake_model,
    pywake_farm_power,
    finite_diff_gradient,
)


def run_gradient_ascent(step_deg: float = 2.0, iters: int = 15):
    D = 178.0
    spacing = 6 * D
    x = np.array([0.0, spacing, 2 * spacing, 3 * spacing])
    y = np.zeros_like(x)
    wf, layout = build_pywake_model(x, y, D, ti=0.06)

    psi = np.zeros(4)
    history = []
    for k in range(iters):
        power = pywake_farm_power(wf, layout, 10.0, 270.0, psi) / 1e6
        history.append((k, psi.copy(), power))

        _, grad = finite_diff_gradient(wf, layout, 10.0, 270.0, psi)
        if np.allclose(grad, 0.0):
            grad = np.array([1.0, 1.0, 1.0, 0.0])
        step = step_deg * np.sign(grad)
        step[-1] = 0.0
        psi = np.clip(psi + step, -25, 25)

    power = pywake_farm_power(wf, layout, 10.0, 270.0, psi) / 1e6
    history.append((iters, psi.copy(), power))
    return history


if __name__ == "__main__":
    hist = run_gradient_ascent()
    for k, psi, power in hist:
        yaw_str = ", ".join(f"{v:.1f}" for v in psi)
        print(f"iter {k:02d}: yaw [{yaw_str}] -> {power:.3f} MW")
