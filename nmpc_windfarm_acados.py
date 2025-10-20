"""
Wind Farm Yaw Control using acados MPC

This module implements a linearized MPC approach for wind farm yaw control using acados.
acados is a fast embedded optimization solver particularly suited for real-time MPC.

Key features:
- Linearized power model around current operating point (updated each step)
- QP formulation for fast solve times
- Handles advection delays between turbines
- Integrates with PyWake for accurate wake modeling

Installation:
    pip install casadi py_wake
    # For acados, follow: https://docs.acados.org/installation/
    # Quick install: pip install acados_template

Formulation:
    States (x):     yaw angles [psi_1, ..., psi_N] ∈ R^N
    Controls (u):   yaw rates [dpsi_1, ..., dpsi_N] ∈ R^N
    Dynamics:       psi_{k+1} = psi_k + u_k * dt

    Objective:      min Σ_k [ -P_k + λ/2 ||u_k||² ]
    where P_k is linearized: P_k ≈ P_0 + ∇P^T (psi_k - psi_0)

    Constraints:    psi_min ≤ psi_k ≤ psi_max
                   |u_k| ≤ u_max
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import time

# PyWake imports
from py_wake.site import UniformSite
from py_wake.examples.data.dtu10mw import DTU10MW as wind_turbine
from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.turbulence_models import CrespoHernandez
from py_wake.deflection_models.jimenez import JimenezWakeDeflection

try:
    from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
    ACADOS_AVAILABLE = True
except ImportError:
    ACADOS_AVAILABLE = False
    print("Warning: acados not available. Install from https://docs.acados.org/installation/")

# ============================================================================
# Data structures (same as original)
# ============================================================================

@dataclass
class Farm:
    x: np.ndarray  # turbine x-positions [m]
    y: np.ndarray  # turbine y-positions [m]
    D: float       # rotor diameter [m]

@dataclass
class Wind:
    U: float       # freestream speed [m/s]
    theta: float   # direction [deg], 0=+x, 90=+y
    TI: float = 0.06  # turbulence intensity

@dataclass
class Limits:
    yaw_min: float = -30.0      # [deg]
    yaw_max: float = 30.0       # [deg]
    yaw_rate_max: float = 0.25  # [deg/s]

@dataclass
class MPCConfig:
    dt: float = 10.0           # [s] sampling time
    N_h: int = 20              # horizon steps
    lam_move: float = 0.5      # weight on yaw rate penalty
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"  # acados QP solver
    # Options: PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM

# ============================================================================
# PyWake model setup
# ============================================================================

def build_pywake_model(x: np.ndarray, y: np.ndarray, D: float):
    """Build PyWake wind farm model for power evaluation."""
    site = UniformSite()
    wt = wind_turbine()
    wf_model = Blondel_Cathelain_2020(
        site, wt,
        turbulenceModel=CrespoHernandez(),
        deflectionModel=JimenezWakeDeflection()
    )
    layout = dict(x=x, y=y, D=D)
    return wf_model, layout

def pywake_farm_power(wf_model, layout, U: float, theta_deg: float, psi_deg: np.ndarray) -> float:
    """Compute total farm power using PyWake."""
    x, y, D = layout["x"], layout["y"], layout["D"]
    N = len(x)
    wd = np.array([theta_deg], dtype=float)
    ws = np.array([U], dtype=float)
    yaw_ilk = psi_deg.reshape(N, 1, 1)

    sim_res = wf_model(x=x, y=y, wd=wd, ws=ws, yaw=yaw_ilk, tilt=0)
    P_ilk = sim_res.Power.values  # shape (N, n_wd, n_ws)
    return float(P_ilk.sum())

def finite_diff_gradient(wf_model, layout, U: float, theta_deg: float,
                        psi: np.ndarray, eps: float = 1e-2) -> Tuple[float, np.ndarray]:
    """
    Compute gradient ∇P via central finite differences.

    Returns:
        P0: power at psi
        grad: gradient dP/dpsi shape (N,)
    """
    N = psi.size
    P0 = pywake_farm_power(wf_model, layout, U, theta_deg, psi)
    grad = np.zeros(N)

    for i in range(N):
        e = np.zeros(N)
        e[i] = eps
        Pp = pywake_farm_power(wf_model, layout, U, theta_deg, psi + e)
        Pm = pywake_farm_power(wf_model, layout, U, theta_deg, psi - e)
        grad[i] = (Pp - Pm) / (2 * eps)

    return P0, grad

# ============================================================================
# Delay computation
# ============================================================================

def rotation_matrix(theta_deg: float) -> np.ndarray:
    """2D rotation matrix to wind-aligned frame."""
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, s], [-s, c]])

def compute_delays(farm: Farm, wind: Wind, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute advection delays between turbines.

    Returns:
        order: turbine indices sorted upstream to downstream
        tau: delay matrix (N, N) where tau[i,j] = delay steps from i to j
    """
    R = rotation_matrix(wind.theta)
    XY = np.vstack([farm.x, farm.y])
    XYp = R @ XY
    xprime = XYp[0]
    order = np.argsort(xprime)

    N = len(farm.x)
    tau = np.zeros((N, N), dtype=int)

    for a in range(N):
        i = order[a]
        for b in range(a + 1, N):
            j = order[b]
            dx = xprime[j] - xprime[i]
            if dx > 0:
                t_ij = dx / max(1e-6, wind.U)
                tau[i, j] = int(np.floor(t_ij / dt))

    return order, tau

# ============================================================================
# acados MPC formulation
# ============================================================================

def create_acados_model(N_turbines: int, dt: float) -> AcadosModel:
    """
    Create acados model for wind farm yaw control.

    Simple integrator dynamics: x_{k+1} = x_k + u_k * dt
    where x = yaw angles, u = yaw rates
    """
    from casadi import SX, vertcat

    model = AcadosModel()
    model.name = "wind_farm_yaw"

    # States: yaw angles [deg]
    x = SX.sym('x', N_turbines)

    # Controls: yaw rates [deg/s]
    u = SX.sym('u', N_turbines)

    # Simple integrator dynamics
    x_next = x + u * dt

    # Explicit dynamics
    model.f_expl_expr = x + u * dt
    model.f_impl_expr = x_next - (x + u * dt)

    model.x = x
    model.u = u
    model.xdot = SX.sym('xdot', N_turbines)  # not used for discrete-time
    model.p = []  # no parameters for now

    return model

def setup_acados_ocp(farm: Farm, wind: Wind, limits: Limits, cfg: MPCConfig,
                     psi_current: np.ndarray,
                     grad_P: np.ndarray) -> AcadosOcpSolver:
    """
    Set up and build acados OCP solver for wind farm yaw control.

    The cost is linearized around psi_current:
        P(psi) ≈ P_0 + grad_P^T (psi - psi_current)

    Objective (to minimize):
        Σ_k [ -grad_P^T * x_k + λ/2 ||u_k||² ]

    Args:
        farm: wind farm layout
        wind: wind conditions
        limits: yaw angle and rate limits
        cfg: MPC configuration
        psi_current: current yaw angles for linearization point
        grad_P: gradient ∇P evaluated at psi_current

    Returns:
        acados OCP solver ready to use
    """
    N = len(farm.x)
    ocp = AcadosOcp()

    # Model
    model = create_acados_model(N, cfg.dt)
    ocp.model = model

    # Dimensions
    ocp.dims.N = cfg.N_h

    # Cost: linear term from power + quadratic term from yaw rates
    # Stage cost: -grad_P^T * x + λ/2 * u^T * u
    # Terminal cost: -grad_P^T * x

    from casadi import SX, vertcat

    # Linear cost on states (maximize power → minimize -P)
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    # Stage cost: we want to minimize [ -grad_P^T * x + λ/2 ||u||² ]
    # In acados LINEAR_LS: 0.5 || V_x * x + V_u * u - y_ref ||²_W
    # We can use: Vx = [I; 0], Vu = [0; sqrt(λ)*I], W = [0, I], y_ref = [grad_P/sqrt(λ), 0]
    # Actually, it's easier to use EXTERNAL cost, but LINEAR_LS is faster for QP

    # Let's use a trick: combine linear and quadratic costs
    # Cost = q^T x + 0.5 u^T R u where q = -grad_P, R = λ*I

    # For LINEAR_LS to work, we need to reformulate
    # Actually, acados supports cost_type = 'LINEAR_LS' where:
    # cost = 0.5 || Vx*x + Vu*u - y_ref ||²_W

    # Alternative: use EXTERNAL cost for more flexibility
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    x = model.x
    u = model.u

    # Stage cost: -grad_P^T * x + (lam_move/2) * u^T * u
    q = SX(grad_P)
    cost_expr = -q.T @ x + (cfg.lam_move / 2) * (u.T @ u)
    ocp.model.cost_expr_ext_cost = cost_expr

    # Terminal cost: just -grad_P^T * x_N
    ocp.model.cost_expr_ext_cost_e = -q.T @ x

    # Constraints
    # State bounds: yaw_min ≤ x ≤ yaw_max
    ocp.constraints.lbx = np.full(N, limits.yaw_min)
    ocp.constraints.ubx = np.full(N, limits.yaw_max)
    ocp.constraints.idxbx = np.arange(N)

    # Control bounds: -yaw_rate_max ≤ u ≤ yaw_rate_max
    ocp.constraints.lbu = np.full(N, -limits.yaw_rate_max)
    ocp.constraints.ubu = np.full(N, limits.yaw_rate_max)
    ocp.constraints.idxbu = np.arange(N)

    # Terminal state bounds (same as stage)
    ocp.constraints.lbx_e = ocp.constraints.lbx
    ocp.constraints.ubx_e = ocp.constraints.ubx
    ocp.constraints.idxbx_e = np.arange(N)

    # Initial state constraint
    ocp.constraints.x0 = psi_current

    # Solver options
    ocp.solver_options.qp_solver = cfg.qp_solver
    ocp.solver_options.hessian_approx = 'EXACT'  # we have exact Hessian for QP
    ocp.solver_options.integrator_type = 'ERK'   # explicit RK (not used for discrete)
    ocp.solver_options.nlp_solver_type = 'SQP'   # SQP for QP problems
    ocp.solver_options.tf = cfg.N_h * cfg.dt

    # Tolerances
    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.nlp_solver_max_iter = 20
    ocp.solver_options.tol = 1e-4

    # Build solver
    solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')

    return solver

# ============================================================================
# MPC controller class
# ============================================================================

class AcadosYawMPC:
    """
    Wind farm yaw control using acados MPC.

    Uses successive linearization: each control step, we:
    1. Evaluate current power and gradient using PyWake
    2. Update the linearized cost in acados
    3. Solve QP for optimal yaw trajectory
    4. Apply first control move
    """

    def __init__(self, farm: Farm, wind: Wind, limits: Limits, cfg: MPCConfig):
        if not ACADOS_AVAILABLE:
            raise ImportError("acados not available. Install from https://docs.acados.org/installation/")

        self.farm = farm
        self.wind = wind
        self.limits = limits
        self.cfg = cfg

        self.N = len(farm.x)

        # Build PyWake model
        self.wf_model, self.layout = build_pywake_model(farm.x, farm.y, farm.D)

        # Compute delays
        self.order, self.tau = compute_delays(farm, wind, cfg.dt)
        self.max_tau = int(np.max(self.tau))

        print(f"Turbine order (upstream→downstream): {self.order}")
        print(f"Max advection delay: {self.max_tau} steps ({self.max_tau * cfg.dt:.1f}s)")

        # Initialize solver (will be rebuilt each step with new gradient)
        self.solver = None

        # State history for delays
        self.psi_current = np.zeros(self.N)
        self.delay_hist = [self.psi_current.copy() for _ in range(self.max_tau + cfg.N_h + 10)]

        # Performance tracking
        self.solve_times = []
        self.power_history = []

    def get_delayed_yaw(self, k: int) -> np.ndarray:
        """
        Get effective yaw angles at step k accounting for advection delays.

        Args:
            k: time step relative to current (0 = current, -1 = one step ago, etc.)

        Returns:
            psi_delayed: yaw angles shape (N,) with per-turbine delays applied
        """
        tau_i = np.max(self.tau, axis=1).astype(int)  # max delay affecting each turbine
        psi_delayed = np.zeros(self.N)

        for i in range(self.N):
            hist_idx = -(k - tau_i[i])  # look back by delay
            if hist_idx < 0:
                hist_idx = 0
            elif hist_idx >= len(self.delay_hist):
                hist_idx = len(self.delay_hist) - 1
            psi_delayed[i] = self.delay_hist[hist_idx][i]

        return psi_delayed

    def update_solver(self):
        """
        Rebuild acados solver with updated gradient at current operating point.

        This implements the successive linearization strategy:
        - Evaluate P and ∇P at current delayed yaw angles
        - Rebuild OCP with new linear cost coefficients
        """
        # Get effective yaw considering delays
        psi_delayed = self.get_delayed_yaw(k=0)

        # Compute gradient via PyWake
        P_current, grad_P = finite_diff_gradient(
            self.wf_model, self.layout,
            self.wind.U, self.wind.theta,
            psi_delayed, eps=1e-2
        )

        self.power_history.append(P_current)

        # Build new solver with updated gradient
        # Note: in practice, you might want to just update cost parameters
        # rather than rebuilding the entire solver, but acados makes this tricky
        # for EXTERNAL costs. For production, consider using code generation.
        self.solver = setup_acados_ocp(
            self.farm, self.wind, self.limits, self.cfg,
            self.psi_current, grad_P
        )

        return P_current, grad_P

    def solve_step(self) -> Tuple[np.ndarray, float]:
        """
        Solve one MPC step and return optimal yaw trajectory.

        Returns:
            psi_plan: optimal yaw angles over horizon, shape (N_h, N)
            solve_time: solver time in seconds
        """
        t0 = time.time()

        # Update solver with current gradient
        P_current, grad_P = self.update_solver()

        # Warm start with current yaw
        for k in range(self.cfg.N_h + 1):
            self.solver.set(k, 'x', self.psi_current)

        # Solve
        status = self.solver.solve()
        solve_time = time.time() - t0
        self.solve_times.append(solve_time)

        if status != 0:
            print(f"Warning: acados solver returned status {status}")

        # Extract solution
        psi_plan = np.zeros((self.cfg.N_h, self.N))
        for k in range(self.cfg.N_h):
            psi_plan[k, :] = self.solver.get(k, 'x')

        return psi_plan, solve_time

    def apply_control(self, psi_plan: np.ndarray):
        """
        Apply first control move and update state history.

        Args:
            psi_plan: optimal yaw trajectory from solve_step
        """
        psi_next = psi_plan[0, :]

        # Enforce rate limit (redundant but safe)
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

    def step(self) -> Dict:
        """
        Execute one full MPC step: solve + apply control.

        Returns:
            info: dict with step information
        """
        psi_plan, solve_time = self.solve_step()
        self.apply_control(psi_plan)

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
            'psi_plan': psi_plan
        }

    def run(self, n_steps: int, verbose: bool = True) -> List[Dict]:
        """
        Run MPC for multiple steps.

        Args:
            n_steps: number of control steps
            verbose: whether to print progress

        Returns:
            history: list of info dicts from each step
        """
        history = []

        for t in range(n_steps):
            info = self.step()
            info['step'] = t
            history.append(info)

            if verbose:
                print(f"t={t:02d}, ψ={np.round(info['psi'], 1)}, "
                      f"P={info['power']/1e6:.3f} MW, "
                      f"solve={info['solve_time']*1000:.1f}ms")

        if verbose:
            avg_solve = np.mean(self.solve_times) * 1000
            print(f"\nAverage solve time: {avg_solve:.1f}ms")

        return history

# ============================================================================
# Demo
# ============================================================================

def run_demo():
    """Demonstrate acados MPC for wind farm yaw control."""

    if not ACADOS_AVAILABLE:
        print("acados not installed. Please install from https://docs.acados.org/installation/")
        return

    print("=" * 70)
    print("Wind Farm Yaw Control with acados MPC")
    print("=" * 70)

    # Setup
    np.random.seed(42)
    D = 178.0  # DTU 10MW rotor diameter

    # 4-turbine row layout
    x = np.array([0.0, 7*D, 14*D, 21*D])
    y = np.zeros_like(x)
    farm = Farm(x=x, y=y, D=D)

    wind = Wind(U=8.0, theta=0.0)  # 8 m/s along +x
    limits = Limits(yaw_min=-25, yaw_max=25, yaw_rate_max=0.3)
    cfg = MPCConfig(dt=10.0, N_h=12, lam_move=0.2)

    print(f"\nFarm layout: {len(x)} turbines in a row")
    print(f"Wind: {wind.U} m/s at {wind.theta}°")
    print(f"Horizon: {cfg.N_h} steps = {cfg.N_h * cfg.dt:.0f}s")
    print(f"QP solver: {cfg.qp_solver}")

    # Create controller
    print("\nInitializing MPC controller...")
    controller = AcadosYawMPC(farm, wind, limits, cfg)

    # Run
    print("\nRunning MPC...\n")
    n_steps = 10
    history = controller.run(n_steps, verbose=True)

    print("\n" + "=" * 70)
    print("Results summary:")
    print("=" * 70)
    powers = [h['power'] for h in history]
    print(f"Initial power: {powers[0]/1e6:.3f} MW")
    print(f"Final power:   {powers[-1]/1e6:.3f} MW")
    print(f"Gain:          {(powers[-1]/powers[0] - 1)*100:.1f}%")
    print(f"Final yaw angles: {np.round(controller.psi_current, 1)}°")

if __name__ == "__main__":
    run_demo()
