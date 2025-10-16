import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import List, Tuple

from scipy.optimize import dual_annealing
from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.turbulence_models import CrespoHernandez
from py_wake.site import UniformSite
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
# from py_wake.examples.data.hornsrev1 import V80

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
        self.U_inf = U_inf
        self.TI = TI
        self.wd = wd
        self.site = UniformSite(p_wd=[1.0], ti=self.TI)
        self.wfm = Blondel_Cathelain_2020(
            self.site, self.wt, 
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
            sim_res = self.wfm(
                x=self.xs, y=self.ys, h=self.hs, 
                wd=[self.wd], ws=[self.U_inf], 
                tilt=0, yaw=yaw_angles_sorted
            )
            
            powers = sim_res.Power.values.flatten()
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
    maxfun: int, 
    seed: int,
    initial_params: np.ndarray = None,
    use_time_shifted: bool = False
) -> np.ndarray:
    """
    Back-to-front optimization with warm-starting support.
    
    Parameters:
    -----------
    use_time_shifted : bool
        If True, uses time-shifted cost function (paper's "CLC shifted")
        If False, uses standard energy maximization
    """
    n_turbines = model.n_turbines
    
    if initial_params is None:
        initial_params = [[0.75, 0.2]] + [[0.5, 0.5] for _ in range(n_turbines - 1)]
    
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

        res = dual_annealing(
            objective_func, 
            bounds=[(0.0, 1.0), (0.0, 1.0)], 
            seed=seed + i, 
            maxfun=maxfun
        )
        opt_params[i, :] = res.x
    
    return opt_params
