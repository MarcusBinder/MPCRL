# Wind Farm MPC Debugging Results

## Problem Summary

The MPC controller exhibited limit cycle oscillations, alternating between two yaw configurations with decreasing power:
```
t=00: ψ=[ 6.7 -4.8 -6.1 -6.9], P=13.961 MW
t=01: ψ=[-0.8  2.7  1.4  0.6], P=14.027 MW
t=02: ψ=[ 6.7 -4.8 -6.1 -6.9], P=13.961 MW  ← Exact repeat!
```

## Root Cause Analysis

### Test 1: Gradient Correctness ✅
**File**: `test_gradient_correctness.py`

**Finding**: Gradient computation is CORRECT
- At ψ=[0,0,0,0]: gradient is exactly zero (stationary point)
- At non-zero yaws: gradient correctly points in power-increasing direction
- Gradient magnitude: ~0.008-0.028 MW/deg (reasonable)

**Conclusion**: The gradient is working properly.

---

### Test 2: Simple Gradient Ascent ⚠️
**File**: `test_simple_gradient_ascent.py`

**Finding**: Even simple gradient ascent oscillates!
- System converges toward ψ≈[0,0,0,0]
- Near zero, gradient becomes very small (~6.7e-4 MW/deg)
- Fixed step size causes oscillations around the zero-gradient point

**Conclusion**: Problem is NOT unique to MPC - it's fundamental to the objective function.

---

### Test 3: Adaptive Gradient Ascent ✅
**File**: `test_adaptive_gradient_ascent.py`

**Finding**: Line search achieves monotonic convergence
- Power increased from 14.017 → 14.027 MW (+10.5 kW, +0.08%)
- Converged to ψ ≈ [0, 0, 0, 0]
- Final gradient: 1.19e-5 MW/deg (essentially zero)

**Conclusion**: ψ=[0,0,0,0] is where gradient descent converges.

---

### Test 4: Search for Global Optimum 🎯
**File**: `test_search_for_optimum.py`

**CRITICAL FINDING**: ψ=[0,0,0,0] IS THE GLOBAL MAXIMUM!

Power ranking:
1. **ψ=[0,0,0,0]: 14.027 MW** ← BEST (baseline)
2. Demo state B: 14.013 MW (-14 kW)
8. **Demo state A: 13.813 MW (-214 kW)** ← MPC oscillating here!

**Key insights**:
- ALL non-zero yaw configurations DECREASE power
- MPC oscillates between rank #2 and rank #8 (both suboptimal)
- System is perfectly symmetric: P(+θ) = P(-θ)
- ψ=0 is a local maximum (all directions decrease power)
- NO configuration achieves > 1% gain over aligned

---

## Why Is ψ=[0,0,0,0] Optimal?

For a **straight-line layout** with turbines perfectly aligned with the wind:
- Wake steering provides NO benefit (no cross-wind separation to exploit)
- Yawing any turbine REDUCES its power capture (cosine loss)
- No downstream benefit compensates for this loss

**This layout cannot benefit from yaw optimization!**

---

## Why Does MPC Oscillate?

The MPC oscillations occur because:

1. **Flat objective**: ψ=[0,0,0,0] is optimal but has zero gradient
2. **Numerical noise**: Small errors in gradient evaluation
3. **Low move penalty**: λ=0.01-0.05 is TOO SMALL
4. **Large control authority**: Rate limit of 7.5°/step allows big movements
5. **Successive linearization**: Linearization is inaccurate far from operating point

The cycle:
```
1. Start near ψ=[0,0,0,0] (optimal)
2. Gradient is nearly zero + numerical noise
3. MPC says "move to improve power" (false signal from noise)
4. Low move penalty → large movement allowed
5. End up at ψ≠0 (WORSE power!)
6. Gradient now points back toward ψ=0
7. MPC says "move back"
8. Overshoot to other side
9. Repeat forever (limit cycle)
```

---

## Solution

### Option 1: Increase Move Penalty (Recommended) ✅

**Increase λ from 0.01-0.05 to 1.0-10.0**

This will:
- Keep system near ψ=[0,0,0,0] where it should be
- Prevent wandering due to gradient noise
- Make small movements only when gradient is large
- Accept that ψ=[0,0,0,0] is optimal

### Option 2: Add Deadband

Only optimize if:
```python
if |current_power - P_at_aligned| > threshold:
    run_MPC()
else:
    # Stay at aligned, don't optimize
    psi_next = np.zeros(N)
```

### Option 3: Don't Use Yaw Control

For this layout (straight line, aligned with wind):
- **Yaw control provides NO benefit**
- **Best strategy: Keep all turbines aligned (ψ=0)**

Accept that this is the wrong layout for yaw optimization!

---

## Recommended Fix

Modify `nmpc_windfarm_acados_fixed.py`:

```python
# OLD:
cfg = MPCConfig(dt=15.0, N_h=20, lam_move=0.01)

# NEW:
cfg = MPCConfig(dt=15.0, N_h=20, lam_move=5.0)  # 500x larger!
```

This will:
- Heavily penalize movement away from current state
- Converge to ψ=[0,0,0,0] and stay there
- Eliminate oscillations
- Accept ~zero power gain (which is correct for this layout!)

---

## Lessons Learned

1. **Test the gradient first**: Always verify gradient correctness before debugging optimization
2. **Understand the objective**: Know what the optimal solution SHOULD be
3. **Check for flat regions**: Flat objectives are hard to optimize
4. **Not all layouts benefit from yaw control**: Straight-line layouts get no benefit
5. **Move penalties matter**: Too small → oscillations, too large → no optimization

---

## For Future Work

If you want yaw optimization to actually provide benefit:

### Better Layout Options:
1. **Staggered grid**: Turbines not perfectly aligned
2. **Wind farm with varying wind directions**: Yaw can redirect wakes
3. **Closely spaced turbines**: Stronger wake interactions

### For This Layout:
- Accept ψ=[0,0,0,0] is optimal
- Use MPC for other purposes (e.g., tracking changing wind)
- Or: Test with different wind angles (not 0°)

---

## Files Created

1. `test_gradient_correctness.py` - Verify gradient computation
2. `test_gradient_nonzero.py` - Test gradient at various points
3. `test_simple_gradient_ascent.py` - Simple optimization without MPC
4. `test_adaptive_gradient_ascent.py` - Gradient ascent with line search
5. `test_search_for_optimum.py` - Systematic search of solution space
6. `DEBUGGING_RESULTS.md` - This document

---

## Next Steps

1. ✅ Identify root cause → **ψ=[0,0,0,0] is optimal, MPC wanders due to low move penalty**
2. ⏳ Implement fix → Increase move penalty
3. ⏳ Validate fix → Re-run demos
4. ⏳ Test with better layout → Staggered or varying wind direction
