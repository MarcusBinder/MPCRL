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
    Farm,
    Wind,
    compute_delays,
)


def delayed_gradient_control(step_deg: float = 2.0, dt: float = 10.0, iters: int = 60):
    D = 178.0
    spacing = 6 * D
    x = np.array([0.0, spacing, 2 * spacing, 3 * spacing])
    y = np.zeros_like(x)

    wf, layout = build_pywake_model(x, y, D, ti=0.06)
    farm = Farm(x=x, y=y, D=D)
    wind = Wind(U=10.0, theta=270.0)
    _, tau = compute_delays(farm, wind, dt=dt)
    delay_steps = np.max(tau, axis=0).astype(int)
    history = [np.zeros(4)]

    psi = history[0].copy()
    records = []
    for k in range(iters):
        effective = np.zeros_like(psi)
        for i in range(len(psi)):
            idx = min(delay_steps[i], len(history) - 1)
            effective[i] = history[idx][i]

        power = pywake_farm_power(wf, layout, 10.0, 270.0, effective) / 1e6
        records.append((k, psi.copy(), effective.copy(), power))

        _, grad = finite_diff_gradient(wf, layout, 10.0, 270.0, effective)
        if np.allclose(grad, 0.0):
            grad = np.array([1.0, 1.0, 1.0, 0.0])
        step = step_deg * np.sign(grad)
        step[-1] = 0.0
        psi = np.clip(psi + step, -25, 25)

        history.insert(0, psi.copy())
        if len(history) > delay_steps.max() + 5:
            history = history[: delay_steps.max() + 5]

    effective = np.zeros_like(psi)
    for i in range(len(psi)):
        idx = min(delay_steps[i], len(history) - 1)
        effective[i] = history[idx][i]
    power = pywake_farm_power(wf, layout, 10.0, 270.0, effective) / 1e6
    records.append((iters, psi.copy(), effective.copy(), power))
    return records


if __name__ == "__main__":
    recs = delayed_gradient_control()
    cmd_hist = []
    eff_hist = []
    power_hist = []
    for k, commanded, effective, power in recs:
        cmd_str = ", ".join(f"{v:.1f}" for v in commanded)
        eff_str = ", ".join(f"{v:.1f}" for v in effective)
        print(f"step {k:02d}: cmd [{cmd_str}] | eff [{eff_str}] -> {power:.3f} MW")
        cmd_hist.append(commanded)
        eff_hist.append(effective)
        power_hist.append(power)

    out = Path("results")
    out.mkdir(exist_ok=True)
    np.savez(
        out / "gradient_delayed_reference.npz",
        command=np.array(cmd_hist),
        effective=np.array(eff_hist),
        power=np.array(power_hist),
    )
    print(f"Saved reference trajectory to {out / 'gradient_delayed_reference.npz'}")
