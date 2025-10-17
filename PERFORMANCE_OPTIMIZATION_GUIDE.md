# MPC Performance Optimization Guide

## Problem Analysis

The MPC-based SAC training is slow because each environment step requires expensive optimization:

**Current computational cost per step:**
- 3 turbines optimized sequentially (back-to-front)
- Each turbine: 50 dual_annealing function evaluations
- Each evaluation: 50 time steps × 3 turbines = 150 power calculations
- **Total: 3 × 50 × 150 = 22,500 power calculations per step**

With 6 parallel environments and 100k training steps:
- **~375 million power calculations total**
- With 70% cache hit rate: **~110 million PyWake simulations**

---

## Quick Wins (Implemented)

### ✅ Use `MPCenvFast`

I've created `mpcrl/environment_fast.py` with optimized parameters:

```python
from mpcrl import MPCenvFast  # Instead of MPCenv

env = MPCenvFast(
    mpc_maxfun=15,      # Reduced from 50 (3.3x speedup)
    mpc_T_opt=300.0,    # Reduced from 500s (1.67x speedup)
    mpc_dt_opt=20.0,    # Increased from 10s (2x speedup)
    # ... other parameters same as before
)
```

**Expected speedup: 6-8x** (theoretical 11x, practical 6-8x with caching effects)

**Why this works:**
- Warm-starting from previous solution means fewer iterations needed
- Action horizon is 100s, so optimizing 500s ahead has diminishing returns
- Coarser time discretization still captures wake dynamics
- Solution quality should remain high (verify with evaluation episodes)

---

## To Use in Your Training

### Option 1: Quick Test (modify `sac_MPC_local.py`)

Change line 22:
```python
# OLD:
from mpcrl import MPCenv, make_config

# NEW:
from mpcrl import MPCenvFast as MPCenv, make_config
```

This drops in as a replacement with optimized defaults.

### Option 2: Custom Tuning

```python
from mpcrl import MPCenvFast, make_config

def make_env(seed):
    # ... your layout code ...

    env = MPCenvFast(
        turbine=wind_turbine(),
        n_passthrough=args.max_eps,
        x_pos=x_pos, y_pos=y_pos,

        # Standard parameters (same as before)
        ws_scaling_min=6, ws_scaling_max=15,
        wd_scaling_min=250, wd_scaling_max=290,
        ti_scaling_min=0.01, ti_scaling_max=0.15,

        # TUNABLE MPC parameters (defaults shown)
        mpc_maxfun=15,          # Try 10-20
        mpc_T_opt=300.0,        # Try 200-400s
        mpc_dt_opt=20.0,        # Try 15-30s
        mpc_t_AH=100.0,         # Action horizon (probably keep at 100)
        mpc_cache_size=64000,   # Larger = better hit rate but more memory
        mpc_cache_quant=0.25,   # Finer = more accurate but lower hit rate

        # ... rest of parameters
    )
    return env
```

---

## Profiling Your Setup

Run the profiling script to measure actual performance:

```bash
python profile_mpc_performance.py
```

This will:
1. Measure time per step with current configuration
2. Test different parameter combinations
3. Show cache hit rates
4. Estimate total training time
5. Recommend optimal settings

**Expected output:**
```
Current: ~10-15s per step
Optimized: ~1.5-2s per step (6-8x faster)
Training time: 10+ hours → 1.5-2 hours
```

---

## Advanced Optimizations (Future Work)

### 1. **Parallel Turbine Optimization** (Complex, high reward)

Currently turbines are optimized sequentially (back-to-front). Could parallelize with coordination:

```python
# Pseudocode for parallel optimization
from multiprocessing import Pool

def optimize_turbine(i, model, current_yaws, fixed_params):
    # Optimize turbine i with fixed downstream turbines
    return optimized_params_i

# In optimize_farm_back2front():
with Pool(3) as pool:
    results = pool.starmap(optimize_turbine, tasks)
```

**Challenges:**
- Need to coordinate between turbines (back-to-front is physically motivated)
- May hurt solution quality
- Overhead of multiprocessing

**Expected speedup:** 2-3x on top of other optimizations

### 2. **Learned MPC Approximator** (Research project)

Train a neural network to approximate the MPC policy:

```python
class MPCApproximator(nn.Module):
    """
    Input: [current_yaws, estimated_ws, estimated_wd, estimated_TI]
    Output: optimal_yaw_deltas
    """
    def forward(self, state):
        # Fast neural network forward pass (~1ms)
        return optimal_yaws

# Training:
# 1. Run slow MPC to collect (state, optimal_action) pairs
# 2. Train NN via supervised learning
# 3. Use NN during RL training (100-1000x faster)
# 4. Optionally fine-tune with RL
```

**Expected speedup:** 100-1000x (neural network forward pass vs full optimization)

**Challenges:**
- Need to collect training data first
- May not generalize to all conditions
- Could use as warm-start then run few MPC iterations

### 3. **Adaptive Optimization Budget**

Allocate more compute when needed:

```python
def adaptive_maxfun(change_in_conditions):
    # If wind conditions changed a lot, use more iterations
    if change_in_conditions > threshold:
        return 30  # More iterations
    else:
        return 10  # Fewer iterations (trust warm-start)
```

**Expected speedup:** 1.5-2x average

### 4. **Simplified Wake Model During Training**

Use a faster but less accurate wake model for training:

```python
# Training: Use LinearSum (fast but less accurate)
from py_wake.superposition_models import LinearSum
wfm = Blondel_Cathelain_2020(..., superpositionModel=LinearSum())

# Evaluation: Use SquaredSum (slow but accurate)
from py_wake.superposition_models import SquaredSum
wfm = Blondel_Cathelain_2020(..., superpositionModel=SquaredSum())
```

**Expected speedup:** 1.5-2x

**Trade-off:** Less accurate gradients during training, but RL is robust to noise

### 5. **Pre-computed Wake Lookup Tables**

For fixed layouts, pre-compute power for grid of (yaw, ws, wd, TI):

```python
# Offline: Pre-compute and save
lookup_table = precompute_power_table(
    yaw_range=(-45, 45, 0.5),
    ws_range=(6, 15, 0.5),
    wd_range=(250, 290, 1),
    ti_range=(0.01, 0.15, 0.01)
)

# Online: Fast interpolation
power = interpolate_from_table(lookup_table, current_state)
```

**Expected speedup:** 10-100x (interpolation vs simulation)

**Challenges:**
- Large memory requirements (4D table)
- Only works for fixed layouts
- Need good interpolation scheme

---

## Recommended Approach

### Phase 1: Quick Wins (Now) ✅
1. Use `MPCenvFast` with default settings
2. Run profiling to verify speedup
3. Start training, monitor performance

### Phase 2: Tuning (This week)
1. Adjust `mpc_maxfun` (try 10, 15, 20)
2. Adjust `mpc_T_opt` (try 200, 300, 400)
3. Find sweet spot between speed and solution quality

### Phase 3: Advanced (Future)
1. Implement learned MPC approximator
2. Try parallel turbine optimization
3. Experiment with simplified wake models

---

## Verification

After implementing optimizations, verify solution quality:

```python
# During training, log MPC optimization quality
info['mpc_objective_value'] = optimized_objective
info['mpc_optimization_time'] = optimization_time

# Periodically evaluate with slow but accurate MPC
if episode % 10 == 0:
    env_eval = MPCenv(...)  # Original slow version
    eval_reward = evaluate(agent, env_eval)
    print(f"Eval reward (accurate MPC): {eval_reward}")
```

**What to check:**
- Training rewards should be similar to baseline
- Evaluation rewards (with accurate MPC) should be high
- Cache hit rates should be >50% (higher is better)

---

## Expected Results

| Configuration | Time/Step | Training Time (100k) | Speedup |
|--------------|-----------|---------------------|---------|
| **Current** (maxfun=50, T_opt=500, dt=10) | 10-15s | 10-15 hours | 1x |
| **Fast** (maxfun=15, T_opt=300, dt=20) | 1.5-2s | 1.5-2 hours | 6-8x |
| + Parallel turbines | 0.5-1s | 0.5-1 hour | 15-20x |
| + Learned approximator | 0.01-0.05s | 0.05-0.2 hours | 200-500x |

---

## Troubleshooting

### "Performance didn't improve"
- Check cache hit rates (should be >50%)
- Profile with `profile_mpc_performance.py`
- Make sure warm-starting is working (check `initial_params`)

### "Training rewards dropped"
- `mpc_maxfun` might be too low, try increasing to 20-25
- `mpc_T_opt` might be too short, try 400s
- Evaluate with slow MPC to check if agent is actually worse or MPC is just noisier

### "Still too slow"
- Consider adaptive optimization budget
- Try simplified wake model
- Investigate learned MPC approximator

---

## Questions?

Feel free to tune the parameters in `MPCenvFast` constructor. The defaults are conservative and should maintain good solution quality while providing significant speedup.

The key insight: **warm-starting + reduced horizons = fast MPC with good solutions**
