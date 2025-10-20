# MPC Control Normalization Fix - Summary

**Date:** 2025-10-20
**Status:** ✅ Successfully implemented and tested

## Problem Statement

The original acados MPC implementation suffered from severe numerical conditioning issues:
- QP solver returned status 4 (max iterations) and status 3 (infeasibility)
- Stationarity residual (`res_stat`) reached O(1e7), indicating poor convergence
- Controls immediately saturated at bounds, preventing optimization
- Step norms dropped to zero, causing the controller to "freeze"

**Root Cause:** Scaling mismatch between O(1e5) power gradients and O(0.3) deg/s control bounds created an ill-conditioned QP problem.

---

## Solution Implemented: Control Normalization

### Key Changes

#### 1. Normalized Control Variables
- **Before:** `u ∈ [-yaw_rate_max, yaw_rate_max]` deg/s (bounds ≈ ±0.25-1.0)
- **After:** `u_norm ∈ [-1, 1]` (dimensionless)
- **Dynamics:** `x_next = x + u_norm * yaw_rate_max * dt`

**Location:** `nmpc_windfarm_acados_fixed.py:162-186`

#### 2. Cost Function Rescaling
All cost terms were adjusted to work with normalized controls:

**Gradient scaling:**
```python
scale_factor = yaw_rate_max * dt
grad_P_scaled = grad_P * scale_factor  # Applied in solve_step()
```

**Quadratic weight scaling:**
```python
quad_weights_scaled = quad_weights * (scale_factor ** 2)
```

**Regularization scaling:**
```python
lam_move_scaled = lam_move * (yaw_rate_max ** 2)  # Applied in setup
```

**Location:** `nmpc_windfarm_acados_fixed.py:490-506, 222-225`

#### 3. Updated Default Parameters
```python
MPCConfig(
    N_h=10,                      # Shorter horizon for linearization validity
    lam_move=10.0,              # Moderate regularization
    trust_region_weight=1e4,    # Balanced with O(1e5-1e6) gradients
    max_quadratic_weight=1e6,   # Allow large weights for strong convexity
    cost_scale=1.0              # No additional scaling needed
)
```

---

## Results

### ✅ Solver Performance
- **Before normalization:**
  - res_stat: O(1e6 - 1e7)
  - QP status: 3, 4 (failures)
  - QP iterations: 0 or 200 (hitting limits)

- **After normalization:**
  - res_stat: O(1e-4 - 1e0) ✅
  - QP status: 0 (optimal)
  - Convergence: "Optimal solution found!"

### Solver Diagnostics Comparison
```
# BEFORE (from diagnostics):
res_stat ≈ 1e7, QP status=4, step_norm=0.00e+00

# AFTER (from tests):
res_stat ≈ 5e-4, QP status=0, "Optimal solution found!"
```

---

## Remaining Challenges

While control normalization significantly improved solver conditioning, the MPC still faces **fundamental nonlinearity issues**:

### Wake Effect Nonlinearity
- Power gradients are computed via finite differences at current yaw angles
- Linearization assumes gradient validity over entire horizon
- Wake effects are highly nonlinear → linearization breaks down as yaw deviates
- **Result:** Solver may fail after 5-10 successful steps when yaw angles move significantly

### Symptoms of Nonlinearity Issues
- First 5-10 optimization steps succeed
- Later steps return QP status 4 or produce zero step norms
- Controller "freezes" at current state instead of progressing

### Mitigation Strategies
1. **Shorter horizons** (N_h=5-10): Keeps linearization valid over smaller prediction windows
2. **Conservative trust regions**: Large quadratic weights prevent aggressive steps
3. **Frequent relinearization**: 10s timesteps allow gradient updates before linearization degrades

---

## Implementation Details

### Modified Files
1. **`nmpc_windfarm_acados_fixed.py`**
   - `create_acados_model_with_params()`: Added `yaw_rate_max` parameter, updated dynamics
   - `setup_acados_ocp_with_params()`: Changed bounds to [-1, 1], scaled regularization
   - `AcadosYawMPC.solve_step()`: Added gradient and weight scaling
   - Updated `MPCConfig` defaults

2. **`scripts/acados_vs_reference.py`**
   - Updated test parameters for validation

### Testing Performed
- ✅ Control normalization implementation
- ✅ Cost scaling validation
- ✅ Solver convergence on demo scenarios
- ✅ Reference trajectory comparison (first 5-10 steps)

---

## Recommendations

### For Production Use
1. **Use short horizons** (N_h=5-10) to maintain linearization validity
2. **Monitor QP status** and implement fallback logic for failures
3. **Consider hybrid approach**: Use MPC when close to equilibrium, switch to simple gradient descent for large moves
4. **Tune per application**:
   - `trust_region_weight`: Balance between conservatism and progress
   - `lam_move`: Control aggressiveness
   - `N_h`: Trade off lookahead vs linearization accuracy

### Future Improvements
1. **Adaptive trust regions**: Increase weight when QP fails, decrease when succeeding
2. **Multi-step gradient**: Compute gradients at multiple points along trajectory
3. **Model Predictive Path Integral (MPPI)**: Sample-based approach that doesn't require linearization
4. **Successive Convexification**: Better handling of nonconvex objectives

---

## Key Takeaways

1. **Control normalization is essential** for MPC with heterogeneous scales (large gradients, small bounds)
2. **Balanced cost terms** prevent solver from becoming too conservative or aggressive
3. **Nonlinear wake dynamics** remain a fundamental challenge for gradient-based MPC
4. **Shorter horizons** trade long-term optimality for numerical reliability

The implementation now provides a solid foundation for wind farm yaw control, with proper numerical conditioning and clear guidance on parameter tuning.

---

## References
- Original diagnostics: `docs/ACADOS_QP_DIAGNOSTICS.md`
- Implementation: `nmpc_windfarm_acados_fixed.py`
- Test script: `scripts/acados_vs_reference.py`
