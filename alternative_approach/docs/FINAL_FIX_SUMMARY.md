# Complete MPC Fix - Final Summary

**Date:** 2025-10-20
**Status:** ✅ Fully working

---

## Problem: Yaw Angles Converging to Zero

After implementing control normalization to fix QP conditioning, the MPC was still failing: yaw angles would start at small random values and immediately converge to zero, with no power improvement.

---

## Root Causes Identified

### 1. Gradient Computed at Wrong Location
**Issue:** Gradient was computed at `psi_current`, but actual power depends on `psi_delayed` due to wake propagation delays.

**Fix:** Compute gradient at delayed yaw angles (line 455-465 in `nmpc_windfarm_acados_fixed.py`):
```python
psi_delayed = self.get_delayed_yaw(k=0)
P_at_delayed, grad_P, hess_diag = finite_diff_gradient(
    self.wf_model, self.layout,
    self.wind.U, self.wind.theta,
    psi_delayed,  # ← Use DELAYED yaws!
    eps=0.5,
    return_hessian=True
)
```

### 2. Finite Difference Epsilon Too Small
**Issue:** With `eps=1e-2` degrees, PyWake was returning **zero gradient** even though power clearly increases with yaw.

**Test Results:**
```
Power at 0°: 4.9415 MW
Power at +5°: 5.0098 MW  (+1.38% improvement)

Gradient with eps=0.01°: 0.00e+00 W/deg  ❌ WRONG!
Gradient with eps=0.5°:  Still zero...   ❌ But why?
```

### 3. Symmetry at Zero Yaw (The Real Culprit!)
**Issue:** PyWake models wake deflection symmetrically. At zero yaw:
```
P(ψ=0°) = 4.9415 MW
P(ψ=+0.5°) = 4.9416 MW  ← Wake deflected to one side
P(ψ=-0.5°) = 4.9416 MW  ← Wake deflected to OTHER side, SAME POWER!
```

**Physics:** Both +ψ and -ψ deflect the wake, and for small angles the power is identical. This creates a **saddle point** at ψ=0 where:
- `dP/dψ = 0` (zero gradient due to symmetry)
- Power increases in EITHER direction away from zero

This is correct physics! The MPC needs to "break the symmetry" and choose which direction to deflect.

**Fix:** Add `direction_bias` to gradient (line 467-468):
```python
if self.cfg.direction_bias != 0.0:
    grad_P = grad_P + self.cfg.direction_bias * self.pref_sign
```

Where `pref_sign` is computed based on turbine layout to deflect wakes away from downstream turbines (lines 323-337).

---

## Complete Solution

### Changes to `nmpc_windfarm_acados_fixed.py`:

1. **Increased finite difference epsilon** (line 462):
   ```python
   eps=0.5  # Was 1e-2
   ```

2. **Compute gradient at delayed yaws** (line 455-465):
   ```python
   psi_delayed = self.get_delayed_yaw(k=0)
   P_at_delayed, grad_P, hess_diag = finite_diff_gradient(
       self.wf_model, self.layout,
       self.wind.U, self.wind.theta,
       psi_delayed,  # Key change!
       eps=0.5,
       return_hessian=True
   )
   ```

3. **Enable direction bias by default** (line 64):
   ```python
   direction_bias: float = 5e4  # Was 0.0
   ```

---

## Results

### Before All Fixes:
- ❌ QP solver status 3/4 (failures)
- ❌ Yaw angles → 0°
- ❌ Power: no improvement
- ❌ Gradient: zero everywhere

### After All Fixes:
- ✅ QP solver status 0 (optimal)
- ✅ res_stat: O(1e-9) (excellent convergence)
- ✅ Yaw angles: [-2.4°, -3.7°, -3.7°, 0°]
- ✅ Power: 4.463 MW → 4.481 MW (**+0.4% gain**)
- ✅ Gradient: ~8.6e4 W/deg (includes bias)

### Final Demo Plot (Demo 3):
```
Power Production: Increases from 4.463 → 4.481 MW
Yaw Trajectories: Turbines 0-2 converge to -2.4° to -3.7°
                  Turbine 3 stays at 0° (last turbine, correct)
Solve Time: 0.36 ms average (excellent performance)
```

---

## Key Learnings

1. **Control normalization is essential** for numerical conditioning with heterogeneous scales

2. **Gradient evaluation point matters** - must match where objective is evaluated (delayed yaws due to wake propagation)

3. **Finite difference step size** must be large enough to overcome numerical precision limits

4. **Physical symmetries** in wake models require explicit symmetry-breaking (direction bias)

5. **Zero gradient ≠ optimal point** - can be a saddle point! Always check physics

---

## Configuration for Production

### Recommended `MPCConfig`:
```python
MPCConfig(
    dt=10.0,                    # Time step
    N_h=10,                      # Horizon (balance lookahead vs linearization)
    lam_move=10.0,              # Regularization (scaled by yaw_rate_max^2)
    trust_region_weight=1e4,    # Ensure convexity vs O(1e5) gradients
    direction_bias=5e4,         # Break symmetry at zero yaw ← CRITICAL!
    grad_clip=5e4,              # Prevent gradient explosions
    cost_scale=1.0              # No additional scaling needed
)
```

### Key Parameters:
- **`direction_bias`**: Must be non-zero to escape zero-yaw saddle point
- **`eps=0.5`** in gradient computation: Large enough for numerical accuracy
- **Compute gradient at `psi_delayed`**: Match power evaluation point

---

## Testing

To verify the fix works:

```bash
# Test gradient direction
python test_gradient_direction.py

# Run optimization demo
python demo_yaw_optimization.py

# Check plots show:
# - Power increasing
# - Yaw angles moving away from zero
# - Optimal convergence
```

---

## References

- Original diagnostics: `docs/ACADOS_QP_DIAGNOSTICS.md`
- First fix (normalization): `docs/MPC_FIX_SUMMARY.md`
- Implementation: `nmpc_windfarm_acados_fixed.py`
- Test scripts: `test_gradient_*.py`

---

**The MPC now works correctly for wind farm yaw optimization with proper physics and numerics! 🎉**
