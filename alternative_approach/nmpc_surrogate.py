"""
Nonlinear MPC with Surrogate Model
===================================

Wind farm yaw control using acados MPC with learned surrogate model as cost function.

This replaces the linearized cost with a nonlinear neural network surrogate,
allowing the MPC to optimize over the full nonlinear landscape.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pickle
import time

import torch
import casadi as ca

try:
    from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
    ACADOS_AVAILABLE = True
except ImportError:
    ACADOS_AVAILABLE = False
    print("Warning: acados not available")

try:
    import l4casadi as l4c
    L4CASADI_AVAILABLE = True
except ImportError:
    L4CASADI_AVAILABLE = False
    print("Warning: l4casadi not available")

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from nmpc_windfarm_acados_fixed import Farm, Wind, Limits, MPCConfig


@dataclass
class SurrogateMPCConfig:
    """Configuration for surrogate-based MPC."""
    dt: float = 10.0                    # Time step (s)
    N_h: int = 20                       # Horizon length
    lam_move: float = 10.0              # Control penalty
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    nlp_solver: str = "SQP"             # Use SQP for nonlinear
    qp_solver_iter_max: int = 100
    nlp_solver_max_iter: int = 100
    tol: float = 1e-6


class SurrogateMPC:
    """
    MPC controller using neural network surrogate as cost function.

    Uses l4casadi to integrate PyTorch model into acados.
    """

    def __init__(
        self,
        model_path: str,
        farm: Farm,
        wind: Wind,
        limits: Limits = None,
        config: SurrogateMPCConfig = None
    ):
        if not ACADOS_AVAILABLE:
            raise ImportError("acados is required")

        if not L4CASADI_AVAILABLE:
            raise ImportError("l4casadi is required. Install with: pip install l4casadi")

        self.farm = farm
        self.wind = wind
        self.limits = limits or Limits()
        self.config = config or SurrogateMPCConfig()

        # Load surrogate model
        print(f"Loading surrogate model from {model_path}...")
        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        self.pytorch_model = data['pytorch_model']
        self.l4c_model = data['l4c_model']
        self.power_func = data['power_func']

        print("  ✅ Surrogate model loaded")

        # Build acados solver
        print("Building acados OCP...")
        self.solver = self._build_acados_ocp()
        print("  ✅ acados solver ready")

        # State
        self.n_turbines = len(farm.x)
        self.current_yaw = np.zeros(self.n_turbines)

    def _build_acados_ocp(self) -> AcadosOcpSolver:
        """Build acados OCP with surrogate cost function."""

        # Create OCP
        ocp = AcadosOcp()

        # Model
        model = self._create_acados_model()
        ocp.model = model

        # Dimensions
        nx = self.n_turbines  # State: yaw angles
        nu = self.n_turbines  # Control: yaw rates

        # Horizon
        ocp.dims.N = self.config.N_h

        # Cost (nonlinear external cost)
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        # Cost function is defined in the model (power_cost)

        # Constraints
        # State bounds
        ocp.constraints.lbx = np.full(nx, np.deg2rad(self.limits.yaw_min))
        ocp.constraints.ubx = np.full(nx, np.deg2rad(self.limits.yaw_max))
        ocp.constraints.idxbx = np.arange(nx)

        # Control bounds
        u_max = np.deg2rad(self.limits.yaw_rate_max)
        ocp.constraints.lbu = np.full(nu, -u_max)
        ocp.constraints.ubu = np.full(nu, u_max)
        ocp.constraints.idxbu = np.arange(nu)

        # Initial state constraint
        ocp.constraints.x0 = np.zeros(nx)

        # Solver options
        ocp.solver_options.qp_solver = self.config.qp_solver
        ocp.solver_options.nlp_solver_type = self.config.nlp_solver
        ocp.solver_options.qp_solver_iter_max = self.config.qp_solver_iter_max
        ocp.solver_options.nlp_solver_max_iter = self.config.nlp_solver_max_iter
        ocp.solver_options.tol = self.config.tol
        ocp.solver_options.tf = self.config.N_h * self.config.dt

        # Create solver
        solver = AcadosOcpSolver(ocp, json_file='acados_ocp_surrogate.json')

        return solver

    def _create_acados_model(self) -> AcadosModel:
        """Create acados model with surrogate cost."""

        model_name = 'windfarm_surrogate'

        # State and control
        nx = self.n_turbines
        nu = self.n_turbines

        x = ca.SX.sym('x', nx)  # Yaw angles (rad)
        u = ca.SX.sym('u', nu)  # Yaw rates (rad/s)

        # Parameters: wind conditions
        p = ca.SX.sym('p', 2)  # [wind_speed, wind_direction]

        # Dynamics: simple integrator
        x_dot = ca.SX.sym('x_dot', nx)
        f_expl = u
        f_impl = x_dot - f_expl

        # Cost function using surrogate
        # Input to surrogate: [yaw_0, yaw_1, yaw_2, yaw_3, wind_speed, wind_direction]
        # Note: Convert yaw from radians to degrees for surrogate
        yaw_deg = x * 180.0 / ca.pi

        # Construct surrogate input
        surrogate_input = ca.vertcat(yaw_deg, p[0], p[1])

        # Evaluate surrogate (power in Watts)
        power = self.power_func(surrogate_input)

        # Cost: minimize negative power (maximize power) + control penalty
        # Normalize control penalty by max rate
        u_normalized = u / np.deg2rad(self.limits.yaw_rate_max)
        control_penalty = self.config.lam_move * ca.dot(u_normalized, u_normalized)

        # Stage cost
        cost = -power[0] + control_penalty

        # Terminal cost (just power, no control)
        cost_e = -power[0]

        # Create model
        model = AcadosModel()
        model.name = model_name
        model.x = x
        model.xdot = x_dot
        model.u = u
        model.p = p
        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.cost_expr_ext_cost = cost
        model.cost_expr_ext_cost_e = cost_e

        return model

    def set_state(self, yaw: np.ndarray):
        """Set current yaw angles."""
        self.current_yaw = yaw.copy()

        # Set initial condition in solver
        yaw_rad = np.deg2rad(yaw)
        self.solver.set(0, 'lbx', yaw_rad)
        self.solver.set(0, 'ubx', yaw_rad)

    def step(self, warm_start: bool = True) -> Dict:
        """
        Solve one MPC step.

        Args:
            warm_start: Use previous solution as initial guess

        Returns:
            Dictionary with solution and diagnostics
        """

        # Set wind conditions as parameters
        wind_params = np.array([self.wind.U, self.wind.theta])

        for i in range(self.config.N_h + 1):
            self.solver.set(i, 'p', wind_params)

        # Solve
        start_time = time.time()
        status = self.solver.solve()
        solve_time = time.time() - start_time

        # Get solution
        yaw_plan_rad = np.array([self.solver.get(i, 'x') for i in range(self.config.N_h + 1)])
        control_plan_rad = np.array([self.solver.get(i, 'u') for i in range(self.config.N_h)])

        # Convert to degrees
        yaw_plan = np.rad2deg(yaw_plan_rad)
        control_plan = np.rad2deg(control_plan_rad)

        # Evaluate power at first step
        with torch.no_grad():
            x_torch = torch.tensor(
                np.concatenate([yaw_plan[0], [self.wind.U, self.wind.theta]]),
                dtype=torch.float32
            ).unsqueeze(0)
            power = float(self.pytorch_model(x_torch).item())

        # Results
        result = {
            'status': status,
            'psi_plan': yaw_plan,
            'control_plan': control_plan,
            'power': power,
            'solve_time': solve_time,
            'iterations': self.solver.get_stats('sqp_iter'),
        }

        # Update state for next step
        self.current_yaw = yaw_plan[0]

        return result

    def reset(self):
        """Reset solver and state."""
        self.current_yaw = np.zeros(self.n_turbines)
        self.solver.reset()


def demo_surrogate_mpc():
    """Demo of surrogate MPC."""

    print("=" * 70)
    print("Surrogate-Based Nonlinear MPC Demo")
    print("=" * 70)

    # Check if model exists
    model_path = Path('models/power_surrogate_casadi.pkl')
    if not model_path.exists():
        print(f"\n❌ Error: Model not found at {model_path}")
        print("\nPlease train and export the model first:")
        print("  1. python scripts/generate_dataset_v2.py --n_samples 100000")
        print("  2. python scripts/train_surrogate_v2.py")
        print("  3. python scripts/export_l4casadi_model.py")
        return

    # Setup
    D = 178.0
    x = np.array([0.0, 5*D, 10*D, 15*D])
    y = np.zeros_like(x)

    farm = Farm(x=x, y=y, D=D)
    wind = Wind(U=8.0, theta=270.0)
    limits = Limits()
    config = SurrogateMPCConfig(N_h=20, lam_move=10.0)

    # Create controller
    print("\nInitializing controller...")
    controller = SurrogateMPC(
        model_path=str(model_path),
        farm=farm,
        wind=wind,
        limits=limits,
        config=config
    )

    # Initial state (start from zero)
    initial_yaw = np.array([0.0, 0.0, 0.0, 0.0])
    controller.set_state(initial_yaw)

    print("\nRunning MPC...")
    print(f"  Initial yaw: {initial_yaw}")
    print(f"  Wind: {wind.U} m/s @ {wind.theta}°")

    # Run a few steps
    for step in range(10):
        result = controller.step()

        print(f"\nStep {step}:")
        print(f"  Status: {result['status']}")
        print(f"  Yaw: {result['psi_plan'][0]}")
        print(f"  Power: {result['power']/1e6:.3f} MW")
        print(f"  Solve time: {result['solve_time']*1000:.1f} ms")

    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print("=" * 70)


if __name__ == '__main__':
    demo_surrogate_mpc()
