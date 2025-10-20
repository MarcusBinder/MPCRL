"""
Wind Farm Yaw Control using acados MPC (FIXED - Fast Version)

This is the corrected version that uses acados parameters to update the gradient
instead of rebuilding the solver at every timestep.

Key fix: Build solver once, update gradient via parameters (10-50ms per solve)
instead of rebuilding (3000ms per solve).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import time
from itertools import product

# PyWake imports
from py_wake.site import UniformSite
from py_wake.examples.data.dtu10mw import DTU10MW as wind_turbine
from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.turbulence_models import CrespoHernandez
from py_wake.deflection_models.jimenez import JimenezWakeDeflection

try:
    from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
    import casadi as ca
    ACADOS_AVAILABLE = True
except ImportError:
    ACADOS_AVAILABLE = False
    print("Warning: acados not available")

# ============================================================================
# Data structures
# ============================================================================

@dataclass
class Farm:
    x: np.ndarray
    y: np.ndarray
    D: float

@dataclass
class Wind:
    U: float
    theta: float
    TI: float = 0.06

@dataclass
class Limits:
    yaw_min: float = -30.0
    yaw_max: float = 30.0
    yaw_rate_max: float = 0.25

@dataclass
class MPCConfig:
    dt: float = 10.0
    N_h: int = 20
    lam_move: float = 0.5
    trust_region_weight: float = 0.2
    trust_region_step: float = 5.0
    max_gradient_scale: float = float("inf")
    max_quadratic_weight: float = 1e4
    direction_bias: float = 0.0
    initial_bias: float = 0.0
    target_weight: float = 0.0
    coarse_yaw_step: float = 5.0
    grad_clip: float | None = 5e4
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"

# ============================================================================
# PyWake setup
# ============================================================================

def build_pywake_model(x: np.ndarray, y: np.ndarray, D: float, ti: float = 0.06):
    site = UniformSite(ti=ti)
    wt = wind_turbine()
    wf_model = Blondel_Cathelain_2020(
        site, wt,
        turbulenceModel=CrespoHernandez(),
        deflectionModel=JimenezWakeDeflection()
    )
    layout = dict(x=x, y=y, D=D)
    return wf_model, layout

def pywake_farm_power(wf_model, layout, U: float, theta_deg: float, psi_deg: np.ndarray) -> float:
    x, y = layout["x"], layout["y"]
    N = len(x)
    wd = np.array([theta_deg], dtype=float)
    ws = np.array([U], dtype=float)
    yaw_ilk = psi_deg.reshape(N, 1, 1)

    sim_res = wf_model(x=x, y=y, wd=wd, ws=ws, yaw=yaw_ilk, tilt=0)
    P_ilk = sim_res.Power.values
    return float(P_ilk.sum())

def finite_diff_gradient(
        wf_model,
        layout,
        U: float,
        theta_deg: float,
        psi: np.ndarray,
        eps: float = 1e-2,
        return_hessian: bool = False,
) -> Tuple[float, np.ndarray] | Tuple[float, np.ndarray, np.ndarray]:
    N = psi.size
    P0 = pywake_farm_power(wf_model, layout, U, theta_deg, psi)
    grad = np.zeros(N)
    hess_diag = np.zeros(N) if return_hessian else None

    for i in range(N):
        e = np.zeros(N)
        e[i] = eps
        Pp = pywake_farm_power(wf_model, layout, U, theta_deg, psi + e)
        Pm = pywake_farm_power(wf_model, layout, U, theta_deg, psi - e)
        grad[i] = (Pp - Pm) / (2 * eps)
        if return_hessian:
            hess_diag[i] = (Pp - 2 * P0 + Pm) / (eps ** 2)

    if return_hessian:
        return P0, grad, hess_diag
    return P0, grad

# ============================================================================
# Delay computation
# ============================================================================

def compute_delays(farm: Farm, wind: Wind, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute advection ordering and integer delays based on wind direction.

    Wind direction is given in meteorological convention (degrees from which
    the wind originates). We project turbine positions onto the downwind unit
    vector to obtain along-wind coordinates and derive propagation delays.
    """
    theta_prop = (wind.theta + 180.0) % 360.0  # direction the wind travels toward
    th = np.deg2rad(theta_prop)
    downwind_dir = np.array([np.sin(th), np.cos(th)])

    positions = np.vstack([farm.x, farm.y])
    xprime = downwind_dir @ positions
    order = np.argsort(xprime)

    N = len(farm.x)
    tau = np.zeros((N, N), dtype=int)

    for a in range(N):
        i = order[a]
        for b in range(a + 1, N):
            j = order[b]
            dx = xprime[j] - xprime[i]
            if dx > 1e-9:
                t_ij = dx / max(1e-6, wind.U)
                tau[i, j] = int(np.floor(t_ij / dt))

    return order, tau

# ============================================================================
# acados MPC with parameters (FIXED VERSION)
# ============================================================================

def create_acados_model_with_params(N_turbines: int, dt: float) -> AcadosModel:
    """
    Create acados model with PARAMETERS for gradient vector.

    This allows us to update the cost gradient without rebuilding the solver.
    """
    model = AcadosModel()
    model.name = "wind_farm_yaw_param"

    # States: yaw angles [deg]
    x = ca.SX.sym('x', N_turbines)

    # Controls: yaw rates [deg/s]
    u = ca.SX.sym('u', N_turbines)

    # PARAMETERS: gradient vector (per turbine) and quadratic weights
    p = ca.SX.sym('p', 2 * N_turbines)

    # Dynamics: simple integrator
    x_next = x + u * dt
    model.f_expl_expr = x_next
    model.f_impl_expr = x_next - (x + u * dt)

    model.x = x
    model.u = u
    model.xdot = ca.SX.sym('xdot', N_turbines)
    model.p = p  # parameters hold gradient and reference

    return model

def setup_acados_ocp_with_params(farm: Farm, wind: Wind, limits: Limits, cfg: MPCConfig,
                                  N_turbines: int) -> AcadosOcpSolver:
    """
    Set up acados OCP solver ONCE with parameterized cost.

    The gradient will be updated via parameters, not by rebuilding.
    """
    ocp = AcadosOcp()

    # Model with parameters
    model = create_acados_model_with_params(N_turbines, cfg.dt)
    ocp.model = model

    # Dimensions
    ocp.dims.N = cfg.N_h

    # Cost using parameters
    x = model.x
    u = model.u
    p = model.p
    N_t = N_turbines
    grad_P = p[:N_t]
    quad_weights = p[N_t:]
    dt = cfg.dt

    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    # Stage cost: linearized power improvement via yaw-rate command
    reg = 1e-6 * (u.T @ u)
    delta_x = u * dt
    delta_sq = ca.power(delta_x, 2)
    cost_expr = (
        -grad_P.T @ delta_x
        + 0.5 * quad_weights.T @ delta_sq
        + (cfg.lam_move / 2) * (u.T @ u)
        + reg
    )
    ocp.model.cost_expr_ext_cost = cost_expr

    # Terminal cost: only quadratic penalty towards stability
    cost_expr_e = 1e-6 * (x.T @ x)
    ocp.model.cost_expr_ext_cost_e = cost_expr_e

    # State bounds
    ocp.constraints.lbx = np.full(N_turbines, limits.yaw_min)
    ocp.constraints.ubx = np.full(N_turbines, limits.yaw_max)
    ocp.constraints.idxbx = np.arange(N_turbines)

    # Control bounds
    ocp.constraints.lbu = np.full(N_turbines, -limits.yaw_rate_max)
    ocp.constraints.ubu = np.full(N_turbines, limits.yaw_rate_max)
    ocp.constraints.idxbu = np.arange(N_turbines)

    # Terminal state bounds
    ocp.constraints.lbx_e = ocp.constraints.lbx
    ocp.constraints.ubx_e = ocp.constraints.ubx
    ocp.constraints.idxbx_e = np.arange(N_turbines)

    # Initial state (will be updated each timestep)
    ocp.constraints.x0 = np.zeros(N_turbines)

    # Set initial parameter values (will be updated each timestep)
    # This is required by acados to know the parameter dimension
    ocp.parameter_values = np.zeros(2 * N_turbines)

    # Solver options
    ocp.solver_options.qp_solver = cfg.qp_solver
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.tf = cfg.N_h * cfg.dt
    ocp.solver_options.qp_solver_iter_max = 200  # Increase QP iterations
    ocp.solver_options.nlp_solver_max_iter = 50   # More SQP iterations
    ocp.solver_options.tol = 1e-3   # Relax tolerance
    ocp.solver_options.qp_tol = 1e-4  # QP tolerance
    ocp.solver_options.print_level = 1
    ocp.solver_options.qp_solver_cond_N = cfg.N_h  # Full condensing for small problems

    # Build solver
    print("Building acados solver (this happens once)...")
    solver = AcadosOcpSolver(ocp, json_file='acados_ocp_param.json', verbose=False)
    print("Solver built successfully!")

    return solver

# ============================================================================
# MPC controller class (FIXED)
# ============================================================================

class AcadosYawMPC:
    """
    Wind farm yaw control using acados MPC with parameter updates.

    FIXED: Builds solver once, updates gradient via parameters each timestep.
    """

    def __init__(self, farm: Farm, wind: Wind, limits: Limits, cfg: MPCConfig):
        if not ACADOS_AVAILABLE:
            raise ImportError("acados not available")

        self.farm = farm
        self.wind = wind
        self.limits = limits
        self.cfg = cfg
        self.N = len(farm.x)

        # Build PyWake model
        self.wf_model, self.layout = build_pywake_model(farm.x, farm.y, farm.D, ti=wind.TI)

        # Compute delays
        self.order, self.tau = compute_delays(farm, wind, cfg.dt)
        self.max_tau = int(np.max(self.tau))

        print(f"Turbine order (upstream→downstream): {self.order}")
        print(f"Max advection delay: {self.max_tau} steps ({self.max_tau * cfg.dt:.1f}s)")

        # Preferred yaw direction: deflect wakes away from downstream turbines
        theta_prop = np.deg2rad((wind.theta + 180.0) % 360.0)
        downwind_dir = np.array([np.sin(theta_prop), np.cos(theta_prop)])
        cross_dir = np.array([-downwind_dir[1], downwind_dir[0]])
        positions = np.vstack([farm.x, farm.y]).T

        pref_sign = np.ones(self.N)
        for idx in range(len(self.order) - 1):
            i = self.order[idx]
            j = self.order[idx + 1]
            lateral = np.dot(cross_dir, positions[j] - positions[i])
            if abs(lateral) > 1e-6:
                pref_sign[i] = np.sign(lateral)
            else:
                pref_sign[i] = 1.0
        pref_sign[self.order[-1]] = 0.0
        self.pref_sign = pref_sign

        self.yaw_target = None
        if self.cfg.target_weight > 0.0:
            self.yaw_target = self._compute_static_target()

        self.solver = setup_acados_ocp_with_params(farm, wind, limits, cfg, self.N)

        # Initialize state and history
        if self.cfg.initial_bias != 0.0:
            self.psi_current = self.pref_sign * self.cfg.initial_bias
        elif self.yaw_target is not None:
            self.psi_current = 0.1 * self.yaw_target
        else:
            self.psi_current = np.zeros(self.N)
        self.delay_hist = [self.psi_current.copy() for _ in range(self.max_tau + cfg.N_h + 10)]
        self.psi_ref_traj = np.tile(self.psi_current, (self.cfg.N_h + 1, 1))

        # Performance tracking
        self.solve_times = []
        self.power_history = []

    def get_delayed_yaw(self, k: int) -> np.ndarray:
        tau_i = np.max(self.tau, axis=1).astype(int)
        psi_delayed = np.zeros(self.N)

        for i in range(self.N):
            hist_idx = -(k - tau_i[i])
            if hist_idx < 0:
                hist_idx = 0
            elif hist_idx >= len(self.delay_hist):
                hist_idx = len(self.delay_hist) - 1
            psi_delayed[i] = self.delay_hist[hist_idx][i]

        return psi_delayed

    def _update_reference_from_plan(self, psi_plan: np.ndarray, psi_terminal: np.ndarray):
        if psi_plan.size == 0:
            self.psi_ref_traj[:] = self.psi_current
            self.psi_ref_traj[-1] = psi_terminal
            return

        for k in range(self.cfg.N_h):
            idx = min(k, psi_plan.shape[0] - 1)
            self.psi_ref_traj[k] = psi_plan[idx, :]
        self.psi_ref_traj[-1] = psi_terminal

    def set_state(self, psi: np.ndarray):
        psi = np.asarray(psi, dtype=float)
        if psi.shape != (self.N,):
            raise ValueError(f"psi must have shape ({self.N},), got {psi.shape}")
        self.psi_current = psi.copy()
        self.delay_hist[0] = psi.copy()
        for k in range(len(self.psi_ref_traj)):
            self.psi_ref_traj[k] = psi.copy()

    def set_history(self, history: np.ndarray):
        history = np.asarray(history, dtype=float)
        if history.ndim != 2 or history.shape[1] != self.N:
            raise ValueError(f"history must have shape (T, {self.N})")
        required_len = self.max_tau + self.cfg.N_h + 10
        padded = list(history)
        while len(padded) < required_len:
            padded.append(history[-1].copy())
        self.delay_hist = [row.copy() for row in padded[:required_len]]
        self.psi_current = self.delay_hist[0].copy()
        for k in range(len(self.psi_ref_traj)):
            self.psi_ref_traj[k] = self.psi_current.copy()

    def _compute_static_target(self) -> np.ndarray:
        step = max(1.0, float(self.cfg.coarse_yaw_step))
        yaw_vals = np.arange(self.limits.yaw_min, self.limits.yaw_max + 1e-9, step)
        best_power = -np.inf
        best_yaw = None

        for combo in product(yaw_vals, repeat=self.N):
            yaw_array = np.array(combo, dtype=float)
            power = pywake_farm_power(
                self.wf_model, self.layout,
                self.wind.U, self.wind.theta,
                yaw_array
            )
            if power > best_power:
                best_power = power
                best_yaw = yaw_array

        if best_yaw is None:
            return np.zeros(self.N)

        aligned = best_yaw.copy()
        for i in range(self.N):
            sign = self.pref_sign[i]
            if sign == 0.0:
                aligned[i] = 0.0
            else:
                if aligned[i] * sign < 0:
                    aligned[i] = -aligned[i]
                if abs(aligned[i]) < step:
                    aligned[i] = 0.0
        return aligned

    def solve_step(self) -> Tuple[np.ndarray, float, float, np.ndarray]:
        """
        Solve one MPC step.

        Returns:
            psi_plan: optimal yaw trajectory
            solve_time: actual solve time (not including gradient computation)
            grad_time: gradient computation time
            grad_P: computed gradient
        """
        # IMPORTANT: Compute gradient at CURRENT yaws, not delayed yaws!
        # The gradient tells us how changing the current yaw angles affects future power.
        # Using delayed yaws would give us the gradient at the wrong point in space.

        t_grad = time.time()
        P_at_current, grad_P, hess_diag = finite_diff_gradient(
            self.wf_model, self.layout,
            self.wind.U, self.wind.theta,
            self.psi_current,  # Use CURRENT yaws, not delayed!
            eps=1e-2,
            return_hessian=True
        )
        grad_time = time.time() - t_grad

        if self.cfg.direction_bias != 0.0:
            grad_P = grad_P + self.cfg.direction_bias * self.pref_sign

        if self.cfg.target_weight > 0.0 and self.yaw_target is not None:
            grad_P = grad_P - (self.cfg.target_weight *
                               (self.yaw_target - self.psi_current) /
                               max(self.cfg.dt, 1e-6))

        if self.cfg.grad_clip is not None:
            grad_cap = float(abs(self.cfg.grad_clip))
            grad_P = np.clip(grad_P, -grad_cap, grad_cap)

        # For reporting, also compute actual power including delays
        psi_delayed = self.get_delayed_yaw(k=0)
        P_with_delays = pywake_farm_power(
            self.wf_model, self.layout,
            self.wind.U, self.wind.theta,
            psi_delayed
        )

        self.power_history.append(P_with_delays)

        # Derive quadratic weights from local curvature (ensure convexity)
        base_weights = np.maximum(self.cfg.trust_region_weight, -hess_diag)
        if self.cfg.max_quadratic_weight is not None:
            base_weights = np.clip(base_weights, self.cfg.trust_region_weight,
                                   self.cfg.max_quadratic_weight)

        quad_weights = base_weights

        grad_P_scaled = grad_P.copy()

        # Update initial condition
        self.solver.set(0, 'lbx', self.psi_current)
        self.solver.set(0, 'ubx', self.psi_current)

        # Refresh reference trajectory start
        self.psi_ref_traj[0] = self.psi_current.copy()

        # Update gradient parameters for all stages (use scaled gradient)
        for k in range(self.cfg.N_h):
            param_vec = np.concatenate([grad_P_scaled, quad_weights])
            self.solver.set(k, 'p', param_vec)
        # Also for terminal stage
        terminal_param = np.concatenate([grad_P_scaled, quad_weights])
        self.solver.set(self.cfg.N_h, 'p', terminal_param)

        # Warm start with current yaw
        for k in range(self.cfg.N_h + 1):
            self.solver.set(k, 'x', self.psi_current)

        # Solve (THIS is the actual optimization, should be fast)
        t_solve = time.time()
        status = self.solver.solve()
        solve_time = time.time() - t_solve

        self.solve_times.append(solve_time)

        if status != 0:
            print(f"Warning: acados solver returned status {status}")

        # Extract solution
        psi_plan = np.zeros((self.cfg.N_h, self.N))
        for k in range(self.cfg.N_h):
            psi_plan[k, :] = self.solver.get(k, 'x')
        psi_terminal = self.solver.get(self.cfg.N_h, 'x')

        self._update_reference_from_plan(psi_plan, psi_terminal)

        return psi_plan, solve_time, grad_time, grad_P  # Return original grad for logging

    def apply_control(self, psi_plan: np.ndarray):
        # psi_plan[0] is the current state (x[0], fixed by initial condition)
        # psi_plan[1] is the next state after applying first control: x[1] = x[0] + u[0]*dt
        # So we should use psi_plan[1] as our target!
        psi_next = psi_plan[1, :] if psi_plan.shape[0] > 1 else psi_plan[0, :]

        # Enforce rate limit (should already be satisfied, but double-check)
        dpsi = np.clip(
            psi_next - self.psi_current,
            -self.limits.yaw_rate_max * self.cfg.dt,
            self.limits.yaw_rate_max * self.cfg.dt
        )
        psi_applied = self.psi_current + dpsi

        # Update history
        self.delay_hist.insert(0, psi_applied.copy())
        if len(self.delay_hist) > (self.max_tau + self.cfg.N_h + 10):
            self.delay_hist = self.delay_hist[:(self.max_tau + self.cfg.N_h + 10)]

        self.psi_current = psi_applied
        self.psi_ref_traj[0] = self.psi_current.copy()
        return psi_applied

    def step(self) -> Dict:
        psi_plan, solve_time, grad_time, grad_P = self.solve_step()
        qp_status = None
        if hasattr(self.solver, 'get_stats'):
            try:
                qp_status = self.solver.get_stats('qp_residual_norm')
            except Exception:
                qp_status = None
        stat = {'qp_info': qp_status}
        applied = self.apply_control(psi_plan)

        # Compute current power
        psi_delayed = self.get_delayed_yaw(k=0)
        P_current = pywake_farm_power(
            self.wf_model, self.layout,
            self.wind.U, self.wind.theta,
            psi_delayed
        )

        return {
            'psi': self.psi_current.copy(),
            'power': P_current,
            'solve_time': solve_time,
            'grad_time': grad_time,
            'total_time': solve_time + grad_time,
            'psi_plan': psi_plan,
            'grad_P': grad_P,
            'solver_stats': stat,
            'applied_yaw': applied.copy(),
        }

    def run(self, n_steps: int, verbose: bool = True) -> List[Dict]:
        history = []
        if self.delay_hist:
            self.delay_hist[0] = self.psi_current.copy()

        for t in range(n_steps):
            info = self.step()
            info['step'] = t
            history.append(info)

            if verbose:
                print(f"t={t:02d}, ψ={np.round(info['psi'], 1)}, "
                      f"P={info['power']/1e6:.3f} MW, "
                      f"solve={info['solve_time']*1000:.1f}ms, "
                      f"grad={info['grad_time']*1000:.0f}ms, "
                      f"|∇P|={np.linalg.norm(info['grad_P']):.2e}, "
                      f"qp_info={info['solver_stats']['qp_info']}")

        if verbose:
            avg_solve = np.mean([h['solve_time'] for h in history]) * 1000
            avg_grad = np.mean([h['grad_time'] for h in history]) * 1000
            print(f"\nAverage solve time: {avg_solve:.1f}ms (optimization only)")
            print(f"Average gradient time: {avg_grad:.0f}ms (PyWake evaluation)")
            print(f"Total average: {avg_solve + avg_grad:.0f}ms per step")

        return history

# ============================================================================
# Demo
# ============================================================================

def run_demo():
    if not ACADOS_AVAILABLE:
        print("acados not installed")
        return

    print("=" * 70)
    print("Wind Farm Yaw Control with acados MPC (FIXED)")
    print("=" * 70)

    np.random.seed(42)
    D = 178.0

    # 4-turbine row
    x = np.array([0.0, 7*D, 14*D, 21*D])
    y = np.zeros_like(x)
    farm = Farm(x=x, y=y, D=D)

    wind = Wind(U=8.0, theta=0.0)
    limits = Limits(yaw_min=-25, yaw_max=25, yaw_rate_max=0.3)
    cfg = MPCConfig(dt=10.0, N_h=12, lam_move=0.2)

    print(f"\nFarm layout: {len(x)} turbines")
    print(f"Wind: {wind.U} m/s at {wind.theta}°")
    print(f"Horizon: {cfg.N_h} steps = {cfg.N_h * cfg.dt:.0f}s")

    # Create controller
    print("\nInitializing MPC controller...")
    controller = AcadosYawMPC(farm, wind, limits, cfg)

    # Run
    print("\nRunning MPC...\n")
    n_steps = 10
    history = controller.run(n_steps, verbose=True)

    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    powers = [h['power'] for h in history]
    print(f"Initial power: {powers[0]/1e6:.3f} MW")
    print(f"Final power:   {powers[-1]/1e6:.3f} MW")
    print(f"Gain:          {(powers[-1]/powers[0] - 1)*100:.1f}%")
    print(f"Final yaw angles: {np.round(controller.psi_current, 1)}°")

if __name__ == "__main__":
    run_demo()
