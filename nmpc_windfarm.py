"""
Minimal NMPC stub for wind-farm yaw control using a steady-state power map + advection delays.

What it is:
- A compact, self-contained Python module built on CasADi's Opti (solved by IPOPT by default).
- Centralized NMPC over a prediction horizon, directly optimizing yaw angles under angle/rate limits.
- Delay handling via integer-step transport delays between upstream→downstream turbines.
- A placeholder steady-state power model you can later swap with PyWake/FLORIS/WindGym.

What to change first:
- Plug your own `power_model()` (see TODO in the function below) or calibrate `c_deficit`, `beta`, and length scale.
- Set your farm layout (x_pos, y_pos, D), sampling time `dt`, horizon length `N_h`, and limits.

How it works (high level):
- Each control step t, we estimate U, theta (kept constant here for simplicity).
- We build/solve an NMPC problem for yaw sequences ψ[0:N_h-1] subject to |Δψ| ≤ r_max*dt and ψ_min ≤ ψ ≤ ψ_max.
- Objective: maximize farm power over horizon minus a yaw-move penalty. We *minimize* negative power in the NLP.
- We include advection delays by looking up upstream yaw angles at time (t - τ_ij) for the influence on turbine j.
- The solver returns the optimal plan; we apply the first move, advance time, shift the delay buffers, and repeat.

Dependencies:
- casadi (pip install casadi)

This is intentionally small and hackable—use it as a scaffold to integrate with WindGym.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import casadi as ca


# pip install py_wake
from py_wake.site import UniformWeibullSite, UniformSite
from py_wake.examples.data.dtu10mw import DTU10MW as wind_turbine
from py_wake import BastankhahGaussian  # or your preferred model
from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.turbulence_models import CrespoHernandez
from py_wake.site import UniformSite
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.superposition_models import SquaredSum
from py_wake.superposition_models import MaxSum



def build_pywake_model(x, y, D):
    site = UniformSite()  # dummy; we’ll override with fixed ws/wd
    wt = wind_turbine()                      # swap to your turbine model if you have one
    wf_model = Blondel_Cathelain_2020(
            site, wt, 
            turbulenceModel=CrespoHernandez(), 
            deflectionModel=JimenezWakeDeflection()
        )
    layout = dict(x=x, y=y, D=D)
    return wf_model, layout

def finite_diff_grad_pywake(wf_model, layout, U, theta_deg, psi, eps=1e-2):
    """Central-diff gradient dP/dpsi at a single step. psi shape (N,) degrees."""
    N = psi.size
    P0 = pywake_farm_power(wf_model, layout, U, theta_deg, psi)
    g = np.zeros(N)
    for i in range(N):
        e = np.zeros(N); e[i] = eps
        Pp = pywake_farm_power(wf_model, layout, U, theta_deg, psi + e)
        Pm = pywake_farm_power(wf_model, layout, U, theta_deg, psi - e)
        g[i] = (Pp - Pm) / (2*eps)
    return P0, g


import numpy as np

def pywake_farm_power(wf_model, layout, U, theta_deg, psi_deg_vec):
    x, y, D = layout["x"], layout["y"], layout["D"]
    N = len(x)
    wd = np.array([theta_deg], dtype=float)
    ws = np.array([U], dtype=float)
    yaw_ilk = psi_deg_vec.reshape(N, 1, 1)

    sim_res = wf_model(x=x, y=y, wd=wd, ws=ws, yaw=yaw_ilk, tilt=0,)
    P_ilk = sim_res.Power.values  # shape (N, n_wd, n_ws)
    return float(P_ilk.sum())

import casadi as ca


class PyWakePowerCallback(ca.Callback):
    def __init__(self, name, wf_model, layout, U, theta_deg, N, eps=1e-3):
        ca.Callback.__init__(self)
        # Store simple Python scalars/objects
        self.N = int(N)
        self.wf_model = wf_model
        self.layout = layout
        self.U = float(U)
        self.theta_deg = float(theta_deg)
        self.eps = float(eps)
        self._last_psi = None
        self._last_P = None
        # Let CasADi do finite-difference derivatives
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):  return 1
    def get_n_out(self): return 1

    # Use N only (don’t touch self.layout here)
    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(self.N, 1)   # column (N,1)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(1, 1)        # scalar

    def eval(self, arg):
        # arg[0] is (N,1); make a flat 1D numpy array
        psi = np.array(arg[0]).reshape(self.N)
        key = psi.tobytes()
        if self._last_psi == key:
            return [self._last_P]
        P = pywake_farm_power(self.wf_model, self.layout, self.U, self.theta_deg, psi)
        out = ca.DM([[P]])
        self._last_psi, self._last_P = key, out
        return [out]


def build_qp_mpc(farm, wind, limits, cfg, psi_prev, delay_hist, tau, wf_model, layout):
    """
    Build an Opti-based QP:
        min 0.5 x^T H x + f^T x
        s.t. lbx <= x <= ubx
             lA <= A x <= uA
    x stacks PSI[k,i] row-wise: [PSI[0,:], PSI[1,:], ..., PSI[Nh-1,:]].
    """
    N  = len(farm.x)
    Nh = cfg.N_h
    nvar = N * Nh

    def idx(k, i): return k * N + i

    # Bounds on angles
    lbx = np.full(nvar, limits.yaw_min, dtype=float)
    ubx = np.full(nvar, limits.yaw_max, dtype=float)

    # Rate constraints: -r <= PSI[k]-PSI[k-1] <= r
    r = limits.yaw_rate_max * cfg.dt
    n_rate = (Nh - 1) * N
    A = ca.DM.zeros(n_rate, nvar)
    lA = ca.DM.full(n_rate, -r)
    uA = ca.DM.full(n_rate,  +r)

    row = 0
    for k in range(1, Nh):
        for i in range(N):
            A[row, idx(k, i)]   =  1.0
            A[row, idx(k-1, i)] = -1.0
            row += 1

    # Quadratic move penalty: lam * sum ||PSI[k]-PSI[k-1]||^2
    lam = cfg.lam_move
    H = ca.DM.zeros(nvar, nvar)
    for k in range(1, Nh):
        for i in range(N):
            ii = idx(k,   i)
            jj = idx(k-1, i)
            H[ii, ii] += lam
            H[jj, jj] += lam
            H[ii, jj] += -lam
            H[jj, ii] += -lam

    # Linear term from PyWake linearization around current delayed ψ
    f = ca.DM.zeros(nvar, 1)
    tau_i = np.max(tau, axis=1).astype(int)

    for k in range(Nh):
        psi_eff = np.zeros(N, dtype=float)
        for i in range(N):
            kk = k - tau_i[i]
            if kk < 0:
                hist_idx = -1 - kk
                if hist_idx >= len(delay_hist):
                    hist_idx = len(delay_hist) - 1
                psi_eff[i] = float(delay_hist[hist_idx][i])
            # else: depends on decision vars; base can stay 0

        _, Sk = finite_diff_grad_pywake(wf_model, layout, wind.U, wind.theta, psi_eff, eps=1e-2)
        for i in range(N):
            kk = k - tau_i[i]
            if kk >= 0:
                f[idx(kk, i)] += -Sk[i]  # minimize -P

    # Build Opti QP
    opti = ca.Opti()
    x = opti.variable(nvar, 1)

    # Cost: 0.5 x^T H x + f^T x
    J = 0.5 * ca.mtimes([x.T, H, x]) + ca.mtimes(f.T, x)
    opti.minimize(J)

    # Bounds
    opti.subject_to(x >= ca.DM(lbx).reshape((nvar, 1)))
    opti.subject_to(x <= ca.DM(ubx).reshape((nvar, 1)))

    # Linear rate constraints
    Ax = ca.mtimes(A, x)
    opti.subject_to(Ax >= lA)
    opti.subject_to(Ax <= uA)

    # Fast, silent SQP (stays in quadratic/linear land)
    p_opts = {"expand": True}
    s_opts = {
        "qpsol": "qrqp",             # pure-Python fallback QP (portable)
        "print_header": False,
        "print_iteration": False,
        "print_time": 0,
    }
    opti.solver("sqpmethod", p_opts, s_opts)

    return opti, x




# ---------------------------
# Utility structures
# ---------------------------
@dataclass
class Farm:
    x: np.ndarray  # [m]
    y: np.ndarray  # [m]
    D: float       # rotor diameter [m]

@dataclass
class Wind:
    U: float       # freestream speed [m/s]
    theta: float   # direction [deg], 0=+x, 90=+y (conventional compass-ish)
    TI: float = 0.06  # turbulence intensity (optional)

@dataclass
class Limits:
    yaw_min: float = -30.0  # [deg]
    yaw_max: float = 30.0   # [deg]
    yaw_rate_max: float = 0.25  # [deg/s]

@dataclass
class NMPCConfig:
    dt: float = 10.0      # [s] sampling time
    N_h: int = 36         # horizon steps (~6 min with dt=10s)
    lam_move: float = 0.2 # weight on yaw moves (deg^2 cost units)
    c_deficit: float = 0.15 # wake deficit coefficient (toy model)
    beta: float = 1.8     # power loss vs. local yaw misalignment: cos(beta*deg2rad(psi))
    L_def: float = 8.0    # wake decay length-scale in D (toy)

# ---------------------------
# Coordinate helpers
# ---------------------------

def rotation_matrix(theta_deg: float) -> np.ndarray:
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, s], [-s, c]])  # rotates (x,y) into wind-aligned frame


def order_and_delays(farm: Farm, wind: Wind, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ordering of turbines along the wind and integer delay steps τ_ij.

    - We rotate coordinates so wind blows along +x'. Upstream → downstream means increasing x'.
    - For each pair i->j with x'_j > x'_i, compute advection delay τ_ij = floor((x'_j-x'_i)/U / dt).

    Returns:
      order: permutation of indices sorted by x' (upstream first)
      xprime: positions in wind frame shape (N,) for info/debug
      tau_ij: integer delays [steps], shape (N,N), tau_ij = steps from i to j (0 if not upstream)
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
        for b in range(a+1, N):
            j = order[b]
            dx = xprime[j] - xprime[i]
            if dx <= 0:
                continue
            t_ij = dx / max(1e-6, wind.U)
            tau[i, j] = int(np.floor(t_ij / dt))
    return order, xprime, tau

# ---------------------------
# Power model (placeholder, replace with PyWake/FLORIS)
# ---------------------------

def power_model(psi_delayed: np.ndarray, farm: Farm, wind: Wind, tau: np.ndarray, cfg: NMPCConfig) -> ca.MX:
    """Toy steady-state farm power with advection-delayed wake influences.

    Arguments:
      psi_delayed: array of shape (N,) of *current* effective yaw angles at each turbine, in degrees,
                   where upstream influences have already been time-shifted via τ_ij when this is called
                   inside the horizon loop (we assemble that outside this function).

    Returns a CasADi expression for total farm power in Watts (relative scale). The model is intentionally simple:
      - Local yaw misalignment reduces power as cos(beta * psi_rad)^3-like (here: cos(beta*psi))^2 times U^3.
      - Upstream wakes reduce local inflow speed via a multiplicative speed factor:
            U_eff_j = U * (1 - sum_i c_deficit * w_ij)
        where w_ij decays ~exp(-dx/(L_def*D)) and only applies if i is upstream of j.
      - The delay logic is handled externally; this function just computes power given the per-turbine *effective* psis.

    NOTE: Replace with a calibrated map from PyWake/FLORIS or your own surrogate for real results.
    """
    N = len(farm.x)
    # Compute upstream weights (geometry-only, precomputable); here we recompute for clarity
    R = rotation_matrix(wind.theta)
    XY = np.vstack([farm.x, farm.y])
    XYp = R @ XY
    xprime = XYp[0]
    yprime = XYp[1]

    W = np.zeros((N, N))  # geometric influence weight
    for i in range(N):
        for j in range(N):
            if xprime[j] <= xprime[i] or i == j:
                continue
            dx = xprime[j] - xprime[i]
            dy = abs(yprime[j] - yprime[i])
            lateral = np.exp(- (dy / (1.5 * farm.D))**2)
            stream = np.exp(- dx / (cfg.L_def * farm.D))
            W[i, j] = lateral * stream

    # Use CasADi-safe degrees→radians conversion
    psi_rad = psi_delayed * (ca.pi / 180)
    # Yaw misalignment factor (toy):
    yaw_fac = ca.power(ca.cos(cfg.beta * psi_rad), 2)

    # Wake speed reduction (linear superposition, clipped): accumulate into an MX scalar
    Psum = 0
    for j in range(N):
        inflow = 1.0
        for i in range(N):
            if W[i, j] > 0:
                inflow = inflow - cfg.c_deficit * W[i, j] * ca.cos(ca.fabs(psi_rad[i]))
        inflow = ca.fmax(0.2, inflow)  # avoid negative speeds
        # Power ~ U^3 * yaw_fac
        Psum = Psum + (wind.U**3) * ca.power(inflow, 3) * yaw_fac[j]
    return Psum

# ---------------------------
# NMPC builder
# ---------------------------

# def build_nmpc(farm: Farm, wind: Wind, limits: Limits, cfg: NMPCConfig,
#                psi_prev: np.ndarray,
#                delay_hist: List[np.ndarray],
#                tau: np.ndarray) -> Tuple[ca.Opti, Dict[str, ca.MX]]:
def build_nmpc(farm: Farm, wind: Wind, limits: Limits, cfg: NMPCConfig,
               psi_prev: np.ndarray,
               delay_hist: List[np.ndarray],
               tau: np.ndarray,
               pw_cb: ca.Callback) -> Tuple[ca.Opti, Dict[str, ca.MX]]:
    """Construct a CasADi Opti problem for yaw NMPC.

    Args:
      psi_prev: last applied yaw angles (deg), shape (N,)
      delay_hist: list of past psi arrays length H_hist, each shape (N,), most recent first.
                  Used to look up ψ_i at t - τ_ij when τ_ij > current horizon index.
      tau: integer delays [steps] between i→j.

    Decision variables:
      PSI[k, i]: yaw angle of turbine i at horizon step k, for k=0..N_h-1.

    Constraints:
      - yaw angle bounds per step
      - rate bounds via |PSI[k]-PSI[k-1]| ≤ r_max * dt

    Objective (minimize):
      sum_k [ - Power( PSI_delayed(k) ) + lam_move * ||PSI[k]-PSI[k-1]||^2 ]

    Returns (opti, vars) for solving; call opti.solve().
    """
    N = len(farm.x)
    opti = ca.Opti()
    PSI = opti.variable(cfg.N_h, N)

    # Ensure psi_prev is a 1xN row for CasADi broadcasting
    psi_prev_row = ca.DM(psi_prev).T  # shape (1 x N)

    # Helper to get Δψ and enforce rate limits
    for k in range(cfg.N_h):
        for i in range(N):
            opti.subject_to(PSI[k, i] >= limits.yaw_min)
            opti.subject_to(PSI[k, i] <= limits.yaw_max)
        if k == 0:
            dpsi = PSI[k, :] - psi_prev_row
        else:
            dpsi = PSI[k, :] - PSI[k-1, :]
        opti.subject_to(ca.fabs(dpsi) <= limits.yaw_rate_max * cfg.dt)

    # Assemble objective with delayed influences
    J = 0
    lam = cfg.lam_move

    # Precompute maximum delay to know how far delay_hist must extend
    max_tau = int(np.max(tau))

    def get_psi_at(i: int, k: int) -> ca.MX:
        """Yaw of turbine i at absolute step k relative to horizon start.
        k>=0: from decision variables; k<0: from delay_hist (most recent first).
        """
        if k >= 0:
            return PSI[k, i]
        hist_idx = -1 - k  # k=-1 → 0 (most recent), k=-2 → 1, ...
        if hist_idx >= len(delay_hist):
            # If history is insufficient, fall back to the oldest available
            hist_idx = len(delay_hist) - 1
        return delay_hist[hist_idx][i]

    for k in range(cfg.N_h):
        # Build per-turbine effective ψ including advection delays from upstream turbines
        psi_eff = []
        for j in range(N):
            # Local turbine's own yaw at time k (no self-delay beyond rate limits)
            psi_j = get_psi_at(j, k)
            psi_eff.append(psi_j)
        psi_eff = ca.hcat(psi_eff).T  # shape (N,)

        # NOTE: upstream influences enter the wake reduction inside power_model via W[i,j] and cos(|psi_i|).
        # The time-delayed yaw of upstream i that affects j at step k is psi_i at (k - tau[i,j]).
        # Our toy power_model uses only psi_delayed vector; we approximate by using the delayed *vector*
        # where each i uses psi_i(k - tau_i*), but since psi_delayed is per turbine not per edge,
        # we take the *max* required delay per i (conservative). You can refine this by expanding the model.
        # Compute a per-i effective delay = max_j tau[i,j]
        tau_i = np.max(tau, axis=1)
        psi_delayed = []
        for i in range(N):
            psi_i_eff = get_psi_at(i, k - int(tau_i[i]))
            psi_delayed.append(psi_i_eff)
        psi_delayed = ca.hcat(psi_delayed).T

        # P = power_model(psi_delayed, farm, wind, tau, cfg)
        # Ensure ψ_delayed is a column (N,1) for the callback
        P = pw_cb(psi_delayed)  # returns a 1x1 MX; CasADi handles it fine in the objective

        if k == 0:
            dpsi_vec = PSI[k, :] - psi_prev_row
        else:
            dpsi_vec = PSI[k, :] - PSI[k-1, :]
        J += (-P) + lam * ca.sumsqr(dpsi_vec)

    opti.minimize(J)

    # Solver options
    # p_opts = {"expand": True}
    # s_opts = {"print_time": 0, "ipopt.print_level": 0}
    # opti.solver("ipopt", p_opts, s_opts)


    opti.solver("ipopt", {"expand": True}, {
        "print_time": 0,
        "ipopt.print_level": 0,
        "max_iter": 50,                 # cap iterations
        "linear_solver": "mumps",       # already default in your build
        "acceptable_tol": 1e-3,
        "tol": 1e-3
    })

    return opti, {"PSI": PSI}

# ---------------------------
# Closed-loop driver (demo)
# ---------------------------

def run_demo():
    np.random.seed(0)
    # --- Layout: simple 4-turbine row ---
    D = 120.0
    x = np.array([0.0, 8*D, 16*D, 24*D])
    y = np.zeros_like(x)
    farm = Farm(x=x, y=y, D=D)

    wind = Wind(U=8.0, theta=0.0)  # blowing along +x

    # wf_model, layout = build_pywake_model(farm.x, farm.y, farm.D)
    # pw_cb = PyWakePowerCallback("pw_power", wf_model, layout, wind.U, wind.theta, eps=1e-2)


    wf_model, layout = build_pywake_model(farm.x, farm.y, farm.D)
    pw_cb = PyWakePowerCallback("pw_power", wf_model, layout, wind.U, wind.theta,
                                N=len(farm.x), eps=1e-2)

    limits = Limits(yaw_min=-25, yaw_max=25, yaw_rate_max=0.3)
    cfg = NMPCConfig(dt=10.0, N_h=8, lam_move=0.1, c_deficit=0.12, beta=1.6, L_def=9.0)

    order, xprime, tau = order_and_delays(farm, wind, cfg.dt)
    print("Order (upstream→downstream):", order)
    print("Max delay (steps):", int(np.max(tau)))

    N = len(x)
    psi_curr = np.zeros(N)  # start aligned

    # History buffer: list of past psi vectors, most recent first. Seed with zeros for max delay coverage.
    max_tau = int(np.max(tau))
    delay_hist: List[np.ndarray] = [psi_curr.copy() for _ in range(max_tau+5)]

    # Closed-loop for a few steps
    n_steps = 5
    for t in range(n_steps):
        # (Optional) update wind estimates here; we keep fixed for demo
        # opti, vars = build_nmpc(farm, wind, limits, cfg, psi_prev=psi_curr, delay_hist=delay_hist, tau=tau)
        # opti, vars = build_nmpc(farm, wind, limits, cfg,
        #                 psi_prev=psi_curr, delay_hist=delay_hist, tau=tau,
        #                 pw_cb=pw_cb)

        
        # # Warm start with current psi as a flat plan
        # guess = np.tile(psi_curr, (cfg.N_h, 1))
        # opti.set_initial(vars["PSI"], guess)

        opti_qp, Xvar = build_qp_mpc(farm, wind, limits, cfg, psi_prev=psi_curr,
                                    delay_hist=delay_hist, tau=tau,
                                    wf_model=wf_model, layout=layout)
        sol = opti_qp.solve()
        X = np.array(sol.value(Xvar)).reshape(cfg.N_h, N)
        psi_plan = X


        # try:
        #     sol = opti.solve()
        # except RuntimeError as e:
        #     print(f"Solver failed at t={t}: {e}. Using previous yaw.")
        #     psi_plan = guess
        # else:
        #     psi_plan = sol.value(vars["PSI"])  # shape (N_h, N)

        # Apply the first control move
        psi_next = psi_plan[0, :]
        # Enforce rate limit (redundant due to constraint, but safer numerically)
        dpsi = np.clip(psi_next - psi_curr, -limits.yaw_rate_max*cfg.dt, limits.yaw_rate_max*cfg.dt)
        psi_applied = psi_curr + dpsi

        # Shift history: insert the *applied* angles at the front
        delay_hist.insert(0, psi_applied.copy())
        # Trim history to a reasonable length
        if len(delay_hist) > (max_tau + cfg.N_h + 10):
            delay_hist = delay_hist[: (max_tau + cfg.N_h + 10)]

        # Log & advance
        psi_curr = psi_applied

        # Compute instantaneous farm power (toy) for logging
        # Using same per-i max delay approximation for consistency
        tau_i = np.max(tau, axis=1)
        psi_delayed_now = np.array([ delay_hist[int(tau_i[i])][i] for i in range(N) ])
        # Cast to CasADi DM to avoid NumPy↔CasADi ufunc issues
        # P_now = float(ca.evalf(power_model(ca.DM(psi_delayed_now), farm, wind, tau, cfg)))
        P_now = float(pw_cb(ca.DM(psi_delayed_now).reshape((-1,1))).full()[0,0])


        print(f"t={t:02d}, ψ={np.round(psi_curr,1)}, P≈{P_now/1e6:.3f} (arb MW)")

    print("Demo finished. Replace power_model() with PyWake/FLORIS for real results.")


if __name__ == "__main__":
    run_demo()
