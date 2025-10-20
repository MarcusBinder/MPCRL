# MPC Linearization Limitation - Wind Farm Yaw Control

**Date:** 2025-10-20
**Status:** ⚠️ Fundamental limitation identified

---

## The Problem

The gradient-based MPC is finding **suboptimal** yaw angles:

| Method | Yaw Angles | Power Gain |
|--------|------------|------------|
| **MPC (gradient-based)** | [-2.4°, -3.7°, -3.7°, 0°] | **+0.4%** |
| **Optimal (grid search)** | [-25°, -20°, -20°, 0°] | **+15.1%** |

The MPC achieves less than 3% of the available power gain!

---

## Root Cause: Nonlinearity + Linearization

### The Wake Power Function is Highly Nonlinear

```python
# PyWake power vs yaw angle (turbine 0):
ψ =  0° → P = 4.941 MW (baseline)
ψ =  5° → P = 5.010 MW (+1.4%)
ψ = 10° → P = 5.198 MW (+5.2%)
ψ = 15° → P = 5.348 MW (+8.2%)
ψ = 20° → P = 5.649 MW (+14.3%)
ψ = 25° → P = 5.686 MW (+15.1%) ← OPTIMAL
```

The power function is **concave** with most gains occurring at larger yaw angles.

### Gradient at Zero is Nearly Zero

Due to wake deflection symmetry:
```python
P(ψ=0°) = 4.941 MW
P(ψ=+0.5°) = 4.942 MW  (wake deflected one way)
P(ψ=-0.5°) = 4.942 MW  (wake deflected other way)

dP/dψ|ψ=0 ≈ 0  ← Zero gradient at origin!
```

Even with `direction_bias=5e4` to break symmetry, the gradient is too weak to drive the controller toward optimal angles.

### MPC Uses First-Order Taylor Expansion

The MPC linearizes the cost:
```python
J(ψ) ≈ J(ψ₀) + ∇J|ψ₀ · (ψ - ψ₀) + quadratic_penalty
```

This approximation is only valid **near ψ₀**. For large moves (0° → 25°), the linearization is completely wrong:

```
True function:    Concave, gains accelerate with larger ψ
Linearization:    Flat near zero, doesn't capture nonlinearity
```

---

## Why Longer Horizon Doesn't Help

**Expected:** Longer horizon → see delayed benefits → larger yaw angles

**Reality:** Linearization breaks down:
1. At ψ=0°, gradient ≈ 0 (after direction bias)
2. MPC tries small step (say, -5°)
3. Re-linearizes at ψ=-5°, gradient still weak
4. Gets stuck in local minimum far from optimal

The problem isn't visibility of delayed effects - it's that the **local linear approximation** doesn't capture the global structure of the power landscape.

---

## Experimental Evidence

### Test 1: Optimal Yaw (Grid Search)
```bash
python test_optimal_yaw.py
```

**Result:** Optimal is [-25°, -20°, -20°, 0°] → +15.1% gain

### Test 2: MPC Performance
```bash
python demo_yaw_optimization.py
```

**Result:** MPC finds [-2.4°, -3.7°, -3.7°, 0°] → +0.4% gain

### Why MPC Fails:
- Starts at ψ=0° where dP/dψ ≈ 0
- Takes small steps based on local gradient
- Never discovers the large gains at ψ ~ 20-25°
- Gets stuck in "near-zero" local region

---

## Alternative Approaches

### 1. ✅ Warm-Start with Good Initial Guess
Start MPC near optimal instead of at zero:

```python
# Instead of:
controller.set_state(np.zeros(4))

# Use:
controller.set_state(np.array([-20, -15, -15, 0]))  # Near-optimal guess
```

**Pros:** MPC can refine near-optimal solution
**Cons:** Requires knowing approximate optimal first

### 2. ✅ Sequential Quadratic Programming (SQP) with Trust Regions
The current implementation IS using SQP, but with very weak trust regions. Could:
- Increase `direction_bias` further (1e5 or higher)
- Reduce `trust_region_weight` to allow larger steps
- Use adaptive trust regions

### 3. ✅ Multi-Start Optimization
Run MPC from multiple initial guesses and pick best:

```python
initial_guesses = [
    [0, 0, 0, 0],
    [-10, -10, -10, 0],
    [-20, -20, -20, 0],
    [10, 10, 10, 0],
]

best_result = None
for guess in initial_guesses:
    controller.set_state(guess)
    result = run_mpc()
    if result['power'] > best_power:
        best_result = result
```

### 4. ✅ Sample-Based Methods (MPPI, CEM)
Instead of gradients, use sampling:
- Model Predictive Path Integral (MPPI)
- Cross-Entropy Method (CEM)
- Doesn't require linearization

**Pros:** Can handle nonlinearity
**Cons:** Computationally expensive, needs many PyWake evaluations

### 5. ✅ Hybrid Approach
1. Use coarse grid search or gradient ascent to get near optimal (~20°)
2. Use gradient-based MPC for real-time fine-tuning

---

## Recommendations

### For This Codebase:

**Short-term fix:**
```python
# Increase direction_bias significantly
cfg = MPCConfig(
    direction_bias=2e5,  # Was 5e4, increase 4×
    trust_region_weight=5e3,  # Was 1e4, reduce to allow larger steps
    ...
)

# OR warm-start near optimal
controller.set_state(np.array([-20, -15, -15, 0]))
```

**Better solution:**
Implement a two-stage approach:
1. **Offline**: Grid search or optimization to find optimal yaw policy for different wind conditions
2. **Online**: Use MPC for tracking and disturbance rejection around the optimal policy

###  For Production Systems:

1. **Use lookup tables** of optimal yaw vs wind conditions (pre-computed)
2. **MPC for tracking** - keep yaw near optimal setpoints
3. **Hybrid with learning** - Reinforcement learning to learn optimal policy offline,  MPC for online execution

---

## Conclusion

The gradient-based MPC **works correctly** from a numerical/solver perspective (good convergence, proper scaling), but has a **fundamental limitation**: it cannot escape local minima in highly nonlinear problems when starting from poor initial guesses.

**The MPC is best suited for:**
- ✅ Tracking known optimal setpoints
- ✅ Disturbance rejection
- ✅ Constraint handling

**The MPC is NOT suitable for:**
- ❌ Finding global optimum from arbitrary start (ψ=0)
- ❌ Exploring highly nonlinear design spaces
- ❌ Problems where linearization breaks down over control horizon

For wind farm yaw optimization, a **hybrid approach** combining global search with local MPC refinement is recommended.

---

## References

- Fleming, P., et al. (2019). "Field test of wake steering at an offshore wind farm" _Wind Energy Science_
  - Reports optimal yaw angles of 15-25° for upstream turbines
  - Demonstrates +10-20% power gains in wake steering

- Original diagnostics: `docs/ACADOS_QP_DIAGNOSTICS.md`
- Numerical fixes: `docs/FINAL_FIX_SUMMARY.md`
- Test scripts: `test_optimal_yaw.py`, `test_long_horizon.py`
