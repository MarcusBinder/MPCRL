"""
Nonlinear MPC with Surrogate Model (CasADi/ipopt version)
==========================================================

Simpler version using CasADi's built-in optimizer instead of acados.
Works immediately with l4casadi without linking issues.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
import pickle
import time

import torch
import casadi as ca

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from nmpc_windfarm_acados_fixed import Farm, Wind, Limits


@dataclass
class SurrogateMPCConfig:
    """Configuration for surrogate-based MPC."""
    dt: float = 10.0                    # Time step (s)
    N_h: int = 20                       # Horizon length
    lam_move: float = 10.0              # Control penalty
    yaw_rate_max: float = 0.25          # Max yaw rate (deg/s)
    max_iter: int = 100                 # Max solver iterations


class SurrogateMPCCasADi:
    """
    MPC controller using neural network surrogate as cost function.
    Uses CasADi's built-in optimizer (ipopt) instead of acados.
    """

    def __init__(
        self,
        model_path: str,
        farm: Farm,
        wind: Wind,
        limits: Limits = None,
        config: SurrogateMPCConfig = None
    ):
        self.farm = farm
        self.wind = wind
        self.limits = limits or Limits()
        self.config = config or SurrogateMPCConfig()

        # Set dimensions
        self.n_turbines = len(farm.x)

        # Load surrogate model
        print(f"Loading surrogate model from {model_path}...")
        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        self.pytorch_model = data['pytorch_model']
        self.power_func = data['power_func']

        print("  ✅ Surrogate model loaded")
        print(f"  Model handles normalization internally via l4casadi")

        # Build MPC optimization problem
        print("Building MPC optimization problem...")
        self._build_mpc()
        print("  ✅ MPC ready")

        # State
        self.current_yaw = np.zeros(self.n_turbines)

    def _build_mpc(self):
        """Build MPC optimization problem using CasADi."""

        N = self.config.N_h
        dt = self.config.dt
        n = self.n_turbines

        # Decision variables: [yaw(0), ..., yaw(N), u(0), ..., u(N-1)]
        # yaw: n x (N+1), u: n x N
        n_states = n * (N + 1)
        n_controls = n * N
        n_vars = n_states + n_controls

        # Create symbolic variables
        X = ca.SX.sym('X', n_vars)

        # Extract state and control trajectories
        def get_state(k):
            """Get yaw angles at timestep k"""
            return X[k*n:(k+1)*n]

        def get_control(k):
            """Get yaw rates at timestep k"""
            return X[n_states + k*n:n_states + (k+1)*n]

        # Build cost function
        J = 0

        for k in range(N):
            # Current state (yaw angles in degrees)
            yaw_deg = get_state(k)

            # Input: [yaw_0, yaw_1, yaw_2, yaw_3, wind_speed, wind_direction]
            surrogate_input = ca.vertcat(yaw_deg, self.wind.U, self.wind.theta)

            # Power from surrogate (handles normalization internally)
            power = self.power_func(surrogate_input)

            # Control (yaw rate in deg/s)
            u = get_control(k)

            # Normalize control for penalty
            u_normalized = u / self.config.yaw_rate_max

            # Stage cost: -power + control penalty
            control_penalty = self.config.lam_move * ca.dot(u_normalized, u_normalized)
            J += -power + control_penalty

        # Terminal cost (just power)
        yaw_deg_N = get_state(N)
        surrogate_input_N = ca.vertcat(yaw_deg_N, self.wind.U, self.wind.theta)
        power_N = self.power_func(surrogate_input_N)
        J += -power_N

        # Constraints
        g = []
        lbg = []
        ubg = []

        # Dynamics constraints: yaw(k+1) = yaw(k) + u(k)*dt
        for k in range(N):
            yaw_k = get_state(k)
            yaw_kp1 = get_state(k+1)
            u_k = get_control(k)

            # Dynamics
            dynamics = yaw_kp1 - (yaw_k + u_k * dt)
            g.append(dynamics)
            lbg.extend([0.0] * n)
            ubg.extend([0.0] * n)

        # Variable bounds
        lbx = []
        ubx = []

        # State bounds: yaw_min <= yaw <= yaw_max
        for k in range(N+1):
            lbx.extend([self.limits.yaw_min] * n)
            ubx.extend([self.limits.yaw_max] * n)

        # Control bounds: -yaw_rate_max <= u <= yaw_rate_max
        for k in range(N):
            lbx.extend([-self.config.yaw_rate_max] * n)
            ubx.extend([self.config.yaw_rate_max] * n)

        # Convert to CasADi types
        g = ca.vertcat(*g) if g else ca.SX.sym('empty', 0, 1)

        # Create NLP
        nlp = {
            'x': X,
            'f': J,
            'g': g
        }

        # Solver options
        opts = {
            'ipopt.max_iter': self.config.max_iter,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.sb': 'yes',  # Suppress banner
        }

        # Create solver
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg
        self.n_vars = n_vars
        self.n_states = n_states
        self.n_controls = n_controls

    def set_state(self, yaw: np.ndarray):
        """Set current yaw angles."""
        self.current_yaw = yaw.copy()

    def step(self) -> Dict:
        """
        Solve one MPC step.

        Returns:
            Dictionary with solution and diagnostics
        """

        N = self.config.N_h
        n = self.n_turbines

        # Initial guess: stay at current yaw
        x0 = np.zeros(self.n_vars)

        # State guess
        for k in range(N+1):
            x0[k*n:(k+1)*n] = self.current_yaw

        # Control guess: zero
        # (already initialized to zero)

        # Update bounds for initial state
        lbx = self.lbx.copy()
        ubx = self.ubx.copy()

        # Fix initial state
        for i in range(n):
            lbx[i] = self.current_yaw[i]
            ubx[i] = self.current_yaw[i]

        # Solve
        start_time = time.time()
        sol = self.solver(x0=x0, lbx=lbx, ubx=ubx, lbg=self.lbg, ubg=self.ubg)
        solve_time = time.time() - start_time

        # Extract solution
        X_opt = np.array(sol['x']).flatten()

        # Extract state trajectory
        yaw_plan = np.zeros((N+1, n))
        for k in range(N+1):
            yaw_plan[k, :] = X_opt[k*n:(k+1)*n]

        # Extract control trajectory
        control_plan = np.zeros((N, n))
        for k in range(N):
            control_plan[k, :] = X_opt[self.n_states + k*n:self.n_states + (k+1)*n]

        # Evaluate power at first step
        with torch.no_grad():
            x_torch = torch.tensor(
                np.concatenate([yaw_plan[0], [self.wind.U, self.wind.theta]]),
                dtype=torch.float32
            ).unsqueeze(0)
            power = float(self.pytorch_model(x_torch).item())

        # Check solver status
        stats = self.solver.stats()
        success = stats['success']

        # Results
        result = {
            'status': 0 if success else 1,
            'psi_plan': yaw_plan,
            'control_plan': control_plan,
            'power': power,
            'solve_time': solve_time,
            'iterations': stats['iter_count'],
            'success': success
        }

        # Update state for next step
        self.current_yaw = yaw_plan[0]

        return result

    def reset(self):
        """Reset state."""
        self.current_yaw = np.zeros(self.n_turbines)


def demo_surrogate_mpc():
    """Demo of surrogate MPC."""

    print("=" * 70)
    print("Surrogate-Based Nonlinear MPC Demo (CasADi/ipopt)")
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
    controller = SurrogateMPCCasADi(
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
        print(f"  Success: {result['success']}")
        print(f"  Yaw: {result['psi_plan'][0]}")
        print(f"  Power: {result['power']/1e6:.3f} MW")
        print(f"  Solve time: {result['solve_time']*1000:.1f} ms")
        print(f"  Iterations: {result['iterations']}")

        if not result['success']:
            print("  ⚠️  Solver did not converge!")
            break

    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print("=" * 70)


if __name__ == '__main__':
    demo_surrogate_mpc()
