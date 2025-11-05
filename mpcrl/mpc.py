import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import List, Tuple

from scipy.optimize import dual_annealing
from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.turbulence_models import CrespoHernandez
from py_wake.site import UniformSite
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.superposition_models import SquaredSum
from py_wake.superposition_models import MaxSum
import logging

# --------------------------
# Basis function
# --------------------------
def sat01(x): return np.clip(x, 0.0, 1.0)

def psi(o1, o2, t, t_AH, r_gamma):
    tn = np.asarray(t) / float(t_AH); eps = 1e-9
    mag = abs(o1 - 0.5); denom = 2.0 * max(mag, eps)
    tsn = o2 * (1.0 - 2.0 * mag); rgn = r_gamma * t_AH
    s = (tn - tsn) / denom
    return 2.0 * (o1 - 0.5) * sat01(s) * rgn

def yaw_traj(gamma0, o1, o2, t_AH, r_gamma, dt, T_total):
    t = np.arange(0.0, T_total + 1e-9, dt)
    dgamma = psi(o1, o2, np.minimum(t, t_AH), t_AH, r_gamma)
    return t, gamma0 + dgamma

# --------------------------
# Enhanced LRU Cache with statistics
# --------------------------
class YawCache:
    def __init__(self, maxsize=4096, quant=0.01, wind_quant=0.1):
        self.maxsize = maxsize
        self.quant = quant
        self.wind_quant = wind_quant
        self.store = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def _key(self, yaws: tuple, U_inf: float, TI: float, wd: float):
        """Create cache key including both yaw angles and wind conditions."""
        q = self.quant
        wq = self.wind_quant
        
        yaw_key = tuple(round(y / q) * q for y in yaws)
        U_key = round(U_inf / wq) * wq
        TI_key = round(TI / 0.001) * 0.001
        wd_key = round(wd / wq) * wq
        
        return (yaw_key, U_key, TI_key, wd_key)
    
    def get(self, yaws: tuple, U_inf: float, TI: float, wd: float):
        key = self._key(yaws, U_inf, TI, wd)
        if key in self.store:
            self.hits += 1
            val = self.store.pop(key)
            self.store[key] = val
            return val
        self.misses += 1
        return None
    
    def put(self, yaws: tuple, U_inf: float, TI: float, wd: float, value):
        key = self._key(yaws, U_inf, TI, wd)
        if key in self.store:
            self.store.pop(key)
        elif len(self.store) >= self.maxsize:
            self.store.popitem(last=False)
        self.store[key] = value
    
    def get_stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total > 0 else 0
        return f"Cache: {self.hits} hits, {self.misses} misses ({hit_rate:.1f}% hit rate), {len(self.store)} entries"
    
    def clear(self):
        """Clear all cache entries and reset statistics."""
        self.store.clear()
        self.hits = 0
        self.misses = 0
    
    def memory_size_mb(self):
        """Estimate cache memory usage in MB."""
        import sys
        total_bytes = sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in self.store.items())
        total_bytes += sys.getsizeof(self.store)
        return total_bytes / (1024 * 1024)

# --------------------------
# Wind Farm Model
# --------------------------
class WindFarmModel:
    def __init__(self, x_pos, y_pos, wt, D=80.0, U_inf=8.0, TI=0.06, wd=270.0, 
                 cache_quant=0.25, cache_size=64000, wind_quant=0.2, U_adv=None,
                 apply_yaw_penalty=True):
        self.x_pos_orig = np.asarray(x_pos)
        self.y_pos_orig = np.asarray(y_pos)
        self.D = D
        self.n_turbines = len(x_pos)
        self.wt = wt
        self.cache = YawCache(maxsize=cache_size, quant=cache_quant, wind_quant=wind_quant)
        self.apply_yaw_penalty = apply_yaw_penalty
        self.update_conditions(U_inf, TI, wd, U_adv)

    def update_conditions(self, U_inf, TI, wd, U_adv=None):
        """Update environmental conditions and rebuild necessary structures."""
        # Apply minimum thresholds to prevent numerical issues in PyWake
        # MIN_TI = 0.01  # Minimum turbulence intensity to prevent division by zero
        # MIN_WS = 3.0   # Minimum wind speed for valid physics

        self.U_inf = U_inf
        self.wd = wd
        self.TI = TI

        self.site = UniformSite(p_wd=[1.0], ti=self.TI)
        self.wfm = Blondel_Cathelain_2020(
            self.site, self.wt, 
            superpositionModel=SquaredSum(),
            # superpositionModel=MaxSum(),
            turbulenceModel=CrespoHernandez(), 
            deflectionModel=JimenezWakeDeflection()
        )
        
        # Recalculate sort order based on wind direction
        wd_rad = np.deg2rad(270.0 - self.wd)
        proj = self.x_pos_orig * np.cos(wd_rad) + self.y_pos_orig * np.sin(wd_rad)
        self.sorted_indices = np.argsort(proj)
        self.unsorted_indices = np.argsort(self.sorted_indices)
        
        self.xs = self.x_pos_orig[self.sorted_indices]
        self.ys = self.y_pos_orig[self.sorted_indices]
        self.hs = np.full(self.n_turbines, self.wt.hub_height())
        
        self.U_adv = U_adv if U_adv is not None else self.U_inf
        
        # Calculate delays based on downstream distance along wind direction
        proj_sorted = self.xs * np.cos(wd_rad) + self.ys * np.sin(wd_rad)
        
        self.delays = np.zeros((self.n_turbines, self.n_turbines))
        for i in range(self.n_turbines):
            for j in range(i):
                downstream_dist = proj_sorted[i] - proj_sorted[j]
                self.delays[j, i] = downstream_dist / self.U_adv
        
    def farm_power_sorted(self, yaw_angles_sorted: np.ndarray) -> np.ndarray:
        """
        Calculate power for sorted yaw angles. Returns power in sorted order.
        Cache key includes yaw angles AND wind conditions.
        Applies yaw penalty (Equation 5) if enabled.
        """
        got = self.cache.get(tuple(yaw_angles_sorted), self.U_inf, self.TI, self.wd)
        if got is not None:
            powers = got
        else:
            # try:
            sim_res = self.wfm(
                x=self.xs, y=self.ys, h=self.hs, 
                wd=[self.wd], ws=[self.U_inf], 
                tilt=0, yaw=yaw_angles_sorted
            )

            powers = sim_res.Power.values.flatten()
            powers = np.where(np.isnan(powers) | np.isinf(powers), 0.0, powers)

            self.cache.put(tuple(yaw_angles_sorted), self.U_inf, self.TI, self.wd, powers)
        
        # Apply implicit yaw constraint (Equation 5 from paper) if enabled
        if self.apply_yaw_penalty:
            misalignments = yaw_angles_sorted
            gamma_max = 33.0   # deg
            gamma_min = -33.0  # deg
            
            w_gamma = (0.5 * np.tanh(50 * (-misalignments + gamma_max)) + 0.5) * \
                  (-0.5 * np.tanh(50 * (-misalignments + gamma_min)) + 0.5)
            
            powers = powers * w_gamma
        
        return powers
    
    def farm_power(self, yaw_angles_orig: np.ndarray) -> np.ndarray:
        """
        Public API: accepts yaw angles in original order, returns power in original order.
        """
        sorted_yaws = yaw_angles_orig[self.sorted_indices]
        sorted_powers = self.farm_power_sorted(sorted_yaws)
        return sorted_powers[self.unsorted_indices]

# --------------------------
# Simulation & Optimization Functions
# --------------------------
def run_farm_delay_loop_optimized(
    model: WindFarmModel, 
    yaw_params: List[List[float]], 
    current_yaw_angles_sorted: np.ndarray, 
    r_gamma: float, 
    t_AH: float, 
    dt: float, 
    T: float
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Optimized delay loop with caching and pre-allocation.
    model: WindFarmModel instance
    yaw_params: list of [o1, o2] for each turbine
    current_yaw_angles_sorted: current yaw angles in sorted order
    r_gamma: yaw rate (deg/s)
    t_AH: action horizon (s)
    dt: time step (s)
    T: total simulation time (s)
    """
    n_turbines = model.n_turbines
    t = np.arange(0.0, T + 1e-9, dt)
    n_steps = len(t)
    
    # Pre-calculate all trajectories
    trajectories = [
        yaw_traj(current_yaw_angles_sorted[i], yaw_params[i][0], 
                yaw_params[i][1], t_AH, r_gamma, dt, T)[1] 
        for i in range(n_turbines)
    ]
    
    traj_array = np.array(trajectories)
    delay_k = (model.delays / dt).round().astype(int)
    P = np.zeros((n_turbines, n_steps))
    
    for k in range(n_steps):
        yaws_at_k = traj_array[:, k].copy()
        
        for i in range(n_turbines):
            yaws_delayed = yaws_at_k.copy()
            for j in range(i):
                delay_idx = max(0, k - delay_k[j, i])
                yaws_delayed[j] = traj_array[j, delay_idx]
            
            all_powers_sorted = model.farm_power_sorted(yaws_delayed)
            P[i, k] = all_powers_sorted[i]
    
    return t, trajectories, P

def farm_energy(P_matrix: np.ndarray, t: np.ndarray) -> float:
    """Calculate total energy using trapezoidal integration."""
    return np.trapezoid(np.sum(P_matrix, axis=0), t)

def optimize_farm_back2front(
    model: WindFarmModel, 
    current_yaw_angles_sorted: np.ndarray, 
    r_gamma: float, 
    t_AH: float, 
    dt_opt: float, 
    T_opt: float, 
    seed: int,
    initial_params: np.ndarray = None,
    use_time_shifted: bool = False,
    method: str = "direct",         # <--- NEW: 'direct' | 'shgo' | 'sobol_powell'
    per_turbine_budget: int = 30,    # <--- NEW: tiny eval budget per turbine
    verbose: bool = False
) -> np.ndarray:
    """
    Back-to-front optimization with warm-starting support.

    Parameters:
    -----------
    use_time_shifted : bool, default=False
        If True, uses time-shifted cost function accounting for wake delays.
        If False (recommended), uses standard total energy maximization.

        Note: Empirical testing shows standard cost performs equivalently or
        better for aligned flow scenarios while being computationally simpler.
        Default is False.
    """
    n_turbines = model.n_turbines
    # print("Running the back-to-front optimization...")
    # print("Using method: ", method)
    # [0.5, 0.5] corresponds to no yaw adjustment. So it just holds current yaw.
    if initial_params is None:
        initial_params = [[0.5, 0.5]] + [[0.5, 0.5] for _ in range(n_turbines - 1)]
    
    opt_params = np.array(initial_params, dtype=float)

    for i in range(n_turbines - 1, -1, -1):
        def objective_func(x):
            current_params = np.copy(opt_params)
            current_params[i, :] = x
            t_opt, _, P_opt = run_farm_delay_loop_optimized(
                model, current_params, current_yaw_angles_sorted, 
                r_gamma, t_AH, dt_opt, T_opt
            )
            
            if use_time_shifted:
                # Time-shifted cost function (Section 2.3.1)
                energy = 0.0
                
                # For this turbine i, integrate over action horizon only
                n_steps_action = int(t_AH / dt_opt)
                energy += np.trapezoid(P_opt[i, :n_steps_action], 
                                      t_opt[:n_steps_action])
                
                # For downstream turbines, shift integration window by delay
                for j in range(i + 1, n_turbines):
                    # Calculate delay from turbine i to turbine j
                    delay_steps = int(model.delays[i, j] / dt_opt)
                    
                    # Integrate power of turbine j from [delay, delay+action_horizon]
                    start_idx = delay_steps
                    end_idx = min(delay_steps + n_steps_action, len(t_opt))
                    
                    if start_idx < len(t_opt):
                        energy += np.trapezoid(P_opt[j, start_idx:end_idx], 
                                             t_opt[start_idx:end_idx])
                
                return -energy
            else:
                # Standard energy maximization
                return -farm_energy(P_opt, t_opt)

        x_best, _ = solve_problem(
            objective=lambda z: objective_func(np.asarray(z)),
            seed=seed + i,
            method=method,
            budget=per_turbine_budget,
            verbose=verbose,
        )
        opt_params[i, :] = x_best


    return opt_params


def make_logger(name="solve_problem", level=logging.INFO):
    """Convenience creator that avoids duplicate handlers."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(name)s][%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def solve_problem(objective, seed=0, method="direct", budget=30, verbose=False, logger=None):
    """
    Minimize 'objective(x)' for x in [0,1]^2 under a strict evaluation budget.
    Returns (x_best (2,), f_best).

    Methods:
      "direct","shgo","sobol_powell",
      "dual_annealing","dual_annealing_powell",
      "gp_bo","gp_bo_powell"

    Logging:
      - Set verbose=True to print INFO logs to stdout
      - Or provide a `logger` (logging.Logger). If both provided, `logger` wins.
    """
    # ----------------- logging setup -----------------
    if logger is None:
        logger = make_logger(level=logging.INFO if verbose else logging.WARNING)

    bounds = [(0.0, 1.0), (0.0, 1.0)]
    if budget <= 0:
        raise ValueError("budget must be >= 1")

    # ---- evaluation counter (wrap the user's objective) ----
    eval_count = {"n": 0}
    def f_wrapped(x):
        eval_count["n"] += 1
        return float(objective(np.asarray(x, dtype=float)))

    def log_result(tag, x, f):
        logger.info(f"{tag}: x={np.asarray(x).round(6)}, f={float(f):.6g}, evals={eval_count['n']}")

    # Single-shot fallback used by methods that can't run with tiny budgets
    def single_probe(tag):
        try:
            from scipy.stats.qmc import Sobol
            X = Sobol(d=2, scramble=True, seed=seed).random_base2(1)
            x = X[0]
            logger.info(f"{tag}: using single Sobol probe (budget={budget})")
        except Exception as e:
            logger.warning(f"{tag}: Sobol unavailable ({type(e).__name__}: {e}); using random probe")
            rng = np.random.default_rng(seed)
            x = rng.random(2)
        f = f_wrapped(x)
        log_result(tag, x, f)
        return np.asarray(x, dtype=float), f

    logger.info(f"method={method}, budget={budget}, seed={seed}")

    # ----------------- solver branches -----------------
    if method == "direct":
        try:
            from scipy.optimize import direct
            logger.info("DIRECT available; honoring maxfun strictly.")
            res = direct(f_wrapped, bounds, maxfun=int(budget), locally_biased=True)
            log_result("DIRECT", res.x, res.fun)
            return np.asarray(res.x), float(res.fun)
        except Exception as e:
            logger.warning(f"DIRECT unavailable ({type(e).__name__}: {e}); falling back to sampling.")
            # fallback: exactly `budget` samples, pick best
            m = int(budget)
            try:
                from scipy.stats.qmc import Sobol
                X = Sobol(d=2, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(m))))[:m]
                logger.info(f"Fallback: Sobol sampling m={len(X)}")
            except Exception:
                rng = np.random.default_rng(seed)
                X = rng.random((m, 2))
                logger.info(f"Fallback: random sampling m={len(X)}")
            fX = np.array([f_wrapped(x) for x in X], dtype=float)
            j = int(np.argmin(fX))
            log_result("DIRECT-fallback-best", X[j], fX[j])
            return np.asarray(X[j]), float(fX[j])

    elif method == "shgo":
        if budget < 6:
            logger.info("SHGO: budget too small (<6). Falling back to single probe.")
            return single_probe("SHGO-fallback")
        try:
            from scipy.optimize import shgo
            n = int(budget)
            logger.info(f"SHGO: running with n={n}, iters=1, sampling='sobol'")
            res = shgo(f_wrapped, bounds, n=n, iters=1, sampling_method="sobol")
            log_result("SHGO", res.x, res.fun)
            return np.asarray(res.x), float(res.fun)
        except Exception as e:
            logger.warning(f"SHGO failed ({type(e).__name__}: {e}); falling back to single probe.")
            return single_probe("SHGO-except-fallback")

    elif method == "sobol_powell":
        from scipy.optimize import minimize
        # --- allocate budget ---
        # aim for ~1/3 of the budget on sampling, up to 12 samples
        m = int(min(12, max(1, budget // 3)))
        local_budget = max(0, int(budget) - m)
        logger.info(f"Sobol+Powell: m={m} samples, local_budget={local_budget}")

        # --- sampling phase ---
        try:
            from scipy.stats.qmc import Sobol
            # Sobol.random_base2(k) yields 2**k points; choose k so we have >= m
            k = int(np.ceil(np.log2(max(1, m))))
            X = Sobol(d=2, scramble=True, seed=seed).random_base2(k)[:m]
            logger.info(f"Sobol sampling with m={len(X)} (2**{k} generated, sliced to m)")
        except Exception as e:
            logger.warning(f"Sobol unavailable ({type(e).__name__}: {e}); using random sampling")
            rng = np.random.default_rng(seed)
            X = rng.random((m, 2))

        fX = np.array([f_wrapped(x) for x in X], dtype=float)
        j0 = int(np.argmin(fX))
        best_x, best_f = np.asarray(X[j0]), float(fX[j0])
        logger.info(f"Sampling best: j={j0}, f={best_f:.6g}")

        # --- local polish (Powell, box-bounded) ---
        if local_budget >= 5:
            logger.info(f"Powell local polish with maxfev={local_budget}")
            res = minimize(
                f_wrapped, best_x,
                method="Powell",
                bounds=bounds,                       # SciPy Powell respects bounds (recent versions)
                options={"maxfev": local_budget, "xtol": 1e-3, "ftol": 1e-3}
            )
            if res.fun <= best_f:
                best_x, best_f = np.asarray(res.x), float(res.fun)
                logger.info("Powell improved the solution.")
            else:
                logger.info("Powell did not improve; keeping sampling best.")
        else:
            logger.info("Local budget too small (<5); skipping local polish.")

        log_result("Sobol+Powell", best_x, best_f)
        return best_x, best_f


    elif method == "dual_annealing":
        try:
            from scipy.optimize import dual_annealing
            logger.info(f"Dual Annealing with maxfun={int(budget)}")
            res = dual_annealing(f_wrapped, bounds=bounds, seed=seed, maxfun=int(budget))
            log_result("DualAnnealing", res.x, res.fun)
            return np.asarray(res.x), float(res.fun)
        except Exception as e:
            logger.warning(f"Dual Annealing failed ({type(e).__name__}: {e}); fallback to single probe.")
            return single_probe("DualAnnealing-except-fallback")

    elif method == "dual_annealing_powell":
        from scipy.optimize import minimize
        try:
            from scipy.optimize import dual_annealing
        except Exception as e:
            logger.warning(f"Dual Annealing unavailable ({type(e).__name__}: {e}); fallback to Sobol+Powell logic.")
            # emulate sobol_powell with given budget
            return solve_problem(objective, seed, "sobol_powell", budget, verbose=False, logger=logger)

        if budget == 1:
            logger.info("DualAnnealing+Powell: budget=1 → single probe fallback.")
            return single_probe("DualAnnealing+Powell-fallback")

        budget_da = max(1, int(np.floor(0.7 * budget)))
        budget_local = max(0, int(budget) - budget_da)
        logger.info(f"DualAnnealing+Powell: DA={budget_da}, local={budget_local}")

        da = dual_annealing(f_wrapped, bounds=bounds, seed=seed, maxfun=budget_da)
        x0, f0 = np.asarray(da.x), float(da.fun)
        logger.info(f"DA result: f={f0:.6g}")

        if budget_local >= 5:
            logger.info(f"Powell local polish with maxfev={budget_local}")
            loc = minimize(
                f_wrapped, x0, method="Powell", bounds=bounds,
                options={"maxfev": budget_local, "xtol": 1e-3, "ftol": 1e-3}
            )
            if loc.fun <= f0:
                log_result("DualAnnealing+Powell", loc.x, loc.fun)
                return np.asarray(loc.x), float(loc.fun)
            else:
                logger.info("Powell did not improve; keeping DA result.")
        else:
            logger.info("Local budget too small (<5); skipping local polish.")

        log_result("DualAnnealing+Powell", x0, f0)
        return x0, f0

    elif method in ("gp_bo", "gp_bo_powell"):
        try:
            from skopt import gp_minimize
            from skopt.space import Real
            logger.info("scikit-optimize detected.")
        except Exception as e:
            logger.warning(f"skopt unavailable ({type(e).__name__}: {e}); falling back to single probe.")
            return single_probe("GPBO-fallback")

        if budget < 4:
            m = int(budget)
            logger.info(f"GP-BO: budget too small (<4). Fallback to {m} probe(s).")
            if m <= 1:
                return single_probe("GPBO-tiny-fallback")
            # m≥2: pick best of m samples
            try:
                from scipy.stats.qmc import Sobol
                X = Sobol(d=2, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(m))))[:m]
                logger.info(f"Fallback Sobol sampling m={len(X)}")
            except Exception:
                rng = np.random.default_rng(seed)
                X = rng.random((m, 2))
                logger.info(f"Fallback random sampling m={len(X)}")
            fX = np.array([f_wrapped(x) for x in X], dtype=float)
            j = int(np.argmin(fX))
            log_result("GPBO-fallback-best", X[j], fX[j])
            return np.asarray(X[j]), float(fX[j])

        space = [Real(0.0, 1.0, name="x0"), Real(0.0, 1.0, name="x1")]
        def f_list(x_list):
            return f_wrapped(x_list)  # x_list is list-like of length 2

        n_calls = int(budget)
        n_init = max(2, min(6, n_calls // 3))
        logger.info(f"GP-BO: n_calls={n_calls}, n_init={n_init}, acq=EI")

        res_bo = gp_minimize(
            f_list, space,
            acq_func="EI",
            n_calls=n_calls,
            n_initial_points=n_init,
            noise=1e-10,
            random_state=seed,
            n_restarts_optimizer=3,
            xi=0.01,
        )
        x_bo, f_bo = np.asarray(res_bo.x, dtype=float), float(res_bo.fun)
        log_result("GP-BO", x_bo, f_bo)

        if method == "gp_bo":
            return x_bo, f_bo

        # For strictness we don't spend extra budget on local polish here.
        # If you want a polish, just increase 'budget' and split it yourself.
        logger.info("GP-BO+Powell: strict budget; skipping local polish by default.")
        return x_bo, f_bo

    else:
        raise ValueError(f"Unknown method {method!r}")