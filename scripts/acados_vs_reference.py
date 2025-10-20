import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nmpc_windfarm_acados_fixed import AcadosYawMPC, Farm, Wind, Limits, MPCConfig


def compare_to_reference(data_path: Path):
    data = np.load(data_path)
    command = data["command"]
    effective = data["effective"]
    power = data["power"]

    D = 178.0
    spacing = 6 * D
    x = np.array([0.0, spacing, 2 * spacing, 3 * spacing])
    y = np.zeros_like(x)
    farm = Farm(x=x, y=y, D=D)
    wind = Wind(U=10.0, theta=270.0, TI=0.06)
    limits = Limits(yaw_min=-25, yaw_max=25, yaw_rate_max=1.0)
    cfg = MPCConfig(
        dt=10.0,
        N_h=40,
        lam_move=0.05,
        trust_region_weight=0.01,
        trust_region_step=10.0,
        max_gradient_scale=float("inf"),
        max_quadratic_weight=1e4,
        direction_bias=0.0,
        initial_bias=0.0,
        target_weight=500.0,
        coarse_yaw_step=5.0,
        grad_clip=5e4,
    )
    controller = AcadosYawMPC(farm, wind, limits, cfg)

    acados_moves = []
    for k in range(len(effective) - 1):
        hist = effective[: k + 1]
        hist_reversed = hist[::-1]
        controller.set_history(hist_reversed)
        psi_plan, solve_time, grad_time, grad = controller.solve_step()
        next_yaw = psi_plan[1] if psi_plan.shape[0] > 1 else psi_plan[0]
        acados_moves.append((k, controller.psi_current.copy(), next_yaw.copy(), command[k + 1]))

    return acados_moves, command, effective, power


if __name__ == "__main__":
    data_path = Path("results/gradient_delayed_reference.npz")
    moves, command, effective, power = compare_to_reference(data_path)
    for k, current, plan_next, ref_next in moves:
        curr_str = ", ".join(f"{v:.1f}" for v in current)
        plan_str = ", ".join(f"{v:.1f}" for v in plan_next)
        ref_str = ", ".join(f"{v:.1f}" for v in ref_next)
        print(f"step {k:02d}: current [{curr_str}] | acados next [{plan_str}] | ref next [{ref_str}]")
