# Wind Farm Yaw Control: MPC Implementation Documentation

**Date:** 2025-10-20
**Status:** Active Development - acados-based MPC Implementation

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Two Approaches](#two-approaches)
3. [Current Work: acados Implementation](#current-work-acados-implementation)
4. [Key Findings](#key-findings)
5. [Documentation Guide](#documentation-guide)
6. [Quick Start](#quick-start)

---

## Project Overview

### The Goal

Optimize wind farm power output through **coordinated yaw control** (wake steering). When upstream turbines yaw (turn) away from the wind, they deflect their wake, allowing downstream turbines to capture more wind energy.

**Potential gains:** 10-15% total power increase

### The Challenge

- **Nonlinear dynamics**: Power response is highly nonlinear with yaw angles
- **Long delays**: Wake propagates at ~11 m/s → 330s delay for turbines 3.5km apart
- **Constraints**: Yaw angles (±25°), yaw rates (0.5°/s max)
- **Real-time**: Need fast optimization for online control

---

## Two Approaches

### Approach 1: Model-Free MPC + RL (Previous Work) ✅

**Implementation:** `mpcrl/` package (Python)

**Method:**
- Model Predictive Control with numerical optimization (scipy)
- Reinforcement Learning for learning-enhanced corrections
- Direct PyWake simulation calls (no linearization)

**Advantages:**
- Works with black-box simulator
- No model approximation needed
- Proven results: +11.8% gain from MPC, +2.6% from RL

**Limitations:**
- Slow: Multiple PyWake evaluations per optimization
- Not suitable for <1s control loops
- Optimization time: seconds to minutes

**Status:** ✅ Complete and working

---

### Approach 2: Model-Based MPC with acados (Current Work) ⚙️

**Implementation:** `nmpc_windfarm_acados_fixed.py`

**Method:**
- Model Predictive Control with acados (C/C++ backend)
- Linearized cost function (first-order Taylor approximation)
- SQP optimization with QP subproblems
- Gradient-based optimization

**Advantages:**
- **Fast:** <1ms solve times
- Handles constraints natively
- Suitable for real-time control (5-10s loops)
- Proven in automotive/robotics

**Challenges:**
- Requires differentiable cost function (linearization)
- Local optimization (can get stuck in local minima)
- Struggles with delayed causality (wake delay >> horizon)

**Status:** ⚙️ **In active development** (this is what we're currently working on)

**Why we're doing this:**
- Explore fast online MPC for real-time wind farm control
- Compare performance with model-free approach
- Understand limitations of gradient-based MPC for this problem
- Potentially use as "tactical layer" in hybrid architecture

---

## Current Work: acados Implementation

### What We've Achieved ✅

#### 1. Numerical Stability - **SOLVED**

**Problem:** QP solver failures (status 3/4), residuals O(1e7)

**Root Cause:** Scaling mismatch
- Power gradients: O(1e5) [Watts/degree]
- Control bounds: O(0.3) [deg/s]
- Ratio: 333,000× difference → ill-conditioned QP

**Solution:** Control normalization
```python
# Before: Physical controls
u [deg/s] with bounds [−0.5, +0.5]  # Bad: O(0.3)

# After: Normalized controls
u [−1, +1] with dynamics: Δψ = u × yaw_rate_max × dt  # Good: O(1)
```

**Result:** ✅ res_stat dropped from O(1e7) to O(1e-9), solver converges

**Documentation:** [FINAL_FIX_SUMMARY.md](FINAL_FIX_SUMMARY.md)

---

#### 2. Gradient Computation - **SOLVED**

**Problem:** Gradient computed at wrong location

**Root Cause:** Power depends on yaw angles from 330s ago (wake delay), not current yaw

**Solution:** Compute gradient at delayed angles
```python
psi_delayed = self.get_delayed_yaw(k=0)  # 330s ago
grad_P = finite_diff_gradient(pywake_model, psi_delayed)
```

**Documentation:** [FINAL_FIX_SUMMARY.md](FINAL_FIX_SUMMARY.md)

---

#### 3. Symmetry Breaking - **SOLVED**

**Problem:** dP/dψ|ψ=0 ≈ 0 (wake can deflect left or right with same initial effect)

**Solution:** Add direction bias
```python
grad_P = grad_P + direction_bias × preferred_sign
```

---

### What We're Still Working On ⚙️

#### Fundamental Challenge: Delayed Causality

**The Core Problem:**

Gradient-based MPC **cannot find optimal yaw from cold start** due to time-asymmetry:

| Action | Timing | Effect |
|--------|--------|--------|
| Yaw away from wind | Immediate (t=0) | ❌ Power loss (cosine misalignment) |
| Wake deflects | Delayed (t=330s) | ✅ Power gain (downstream turbines) |

**MPC horizon: 100s << Wake delay: 330s**

**Result:**
- Optimal yaw: [-25°, -20°, -20°, 0°] → **+15.1% gain**
- MPC finds: [-5°, -5°, -5°, 0°] → **-0.2% gain** ❌

MPC can't see the benefit within its horizon, so it resists yawing!

**Documentation:** [LINEARIZATION_LIMITATION.md](LINEARIZATION_LIMITATION.md)

---

#### Current Solution: Hybrid Architecture

We're implementing a **two-layer approach** (industry standard for this problem):

```
┌─────────────────────────────────────────────────┐
│  STRATEGIC LAYER (Slow - every 1-10 minutes)   │
│  ────────────────────────────────────────────   │
│  • Method: Global optimizer (DE, PSO, Grid)    │
│  • Task: Find optimal yaw for current wind     │
│  • Runtime: 10-60s (runs in background)        │
│  • Output: yaw_target                          │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  TACTICAL LAYER (Fast - every 5-10 seconds)    │
│  ────────────────────────────────────────────   │
│  • Method: Gradient MPC (acados)               │
│  • Task: Track yaw_target with constraints     │
│  • Handles: Rate limits, bounds, smoothness    │
│  • Runtime: <1ms                               │
└─────────────────────────────────────────────────┘
```

**Status:** Implementation in progress (tracking convergence issues being resolved)

**Documentation:** [FINAL_RECOMMENDATIONS.md](FINAL_RECOMMENDATIONS.md)

---

### Performance Comparison

| Approach | Final Yaw | Power | Gain | Status |
|----------|-----------|-------|------|--------|
| **Baseline (0°)** | [0, 0, 0, 0] | 4.94 MW | +0.0% | Reference |
| **Grid Search Optimal** | [-25, -20, -20, 0] | 5.69 MW | **+15.1%** | ✅ Ground truth |
| **acados MPC (cold)** | [-5, -5, -5, 0] | 4.94 MW | -0.2% | ❌ Stuck in local min |
| **acados MPC (warm)** | [-18, -15, -15, 0] | 4.93 MW | -0.2% | ❌ Still suboptimal |
| **Hybrid (in progress)** | TBD | TBD | TBD | ⚙️ Working on it |

---

## Key Findings

### 1. Gradient-Based MPC Has Fundamental Limitation for This Problem

**What we learned:**

Gradient-based MPC is the **wrong tool** for:
- ❌ Finding optimal yaw from scratch (planning)
- ❌ Problems with delayed causality (cost now, benefit much later)
- ❌ Highly nonlinear costs with weak gradients at origin

Gradient-based MPC is the **right tool** for:
- ✅ Tracking pre-computed optimal setpoints (control)
- ✅ Constraint handling (rate limits, bounds)
- ✅ Disturbance rejection
- ✅ Fast online execution

**Implication:** Don't use acados MPC for strategic planning. Use it for tactical control.

---

### 2. Control Normalization is Critical for Numerical Stability

**Before:** Gradient O(1e5), Controls O(0.3) → 333,000× mismatch → QP fails

**After:** Normalize controls to [-1, 1], scale cost → O(1) everywhere → QP solves

**Lesson:** Always check condition number when mixing different units in optimization

---

### 3. Wake Delay Must Be Handled Carefully

**Wrong:** Compute gradient at current yaw → incorrect direction
**Right:** Compute gradient at delayed yaw (330s ago) → correct direction

**Reason:** The power we're trying to maximize depends on yaw angles from 330s ago, not now

---

### 4. Hybrid Architecture is Industry Standard

For problems with:
- Slow-changing setpoints (wind conditions)
- Fast disturbances (turbulence, gusts)
- Complex global optimization (nonlinear, multimodal)
- Strict timing constraints (real-time control)

**Use:** Two-layer architecture
- **Strategic** (slow): Find optimal operating point
- **Tactical** (fast): Track it with MPC

This is how it's done in automotive (path planning + tracking), robotics (motion planning + control), and process industries.

---

## Documentation Guide

### Start Here

1. **[README.md](README.md)** - This document (overview)
2. **[INDEX.md](INDEX.md)** - Complete documentation index

### acados Implementation Journey

Read these in order to understand the acados implementation:

3. **[ACADOS_QP_DIAGNOSTICS.md](ACADOS_QP_DIAGNOSTICS.md)** - Original problem (QP failures)
4. **[FINAL_FIX_SUMMARY.md](FINAL_FIX_SUMMARY.md)** - How we fixed numerical issues
5. **[LINEARIZATION_LIMITATION.md](LINEARIZATION_LIMITATION.md)** - Why gradient MPC fails for this problem
6. **[MPC_ALTERNATIVES.md](MPC_ALTERNATIVES.md)** - Analysis of 6 different MPC approaches
7. **[FINAL_RECOMMENDATIONS.md](FINAL_RECOMMENDATIONS.md)** - Production-ready recommendations

### Previous Work (MPC+RL)

If you want to understand the original model-free approach:

8. **[archive/PAPER_OUTLINE_V2.md](archive/PAPER_OUTLINE_V2.md)** - MPC+RL paper outline
9. **[archive/WAKE_DELAY_FIX_SUMMARY.md](archive/WAKE_DELAY_FIX_SUMMARY.md)** - Wake delay handling in model-free MPC
10. **[archive/PERFORMANCE_OPTIMIZATION_GUIDE.md](archive/PERFORMANCE_OPTIMIZATION_GUIDE.md)** - Optimization guide for scipy-based MPC

---

## Quick Start

### Installation

```bash
# Install acados (follow official docs)
# https://docs.acados.org/installation/

# Install Python dependencies
pip install -r requirements.txt
```

### Running Examples

```bash
# 1. Test optimal yaw via grid search (ground truth)
python tests/test_optimal_yaw.py

# Output: [-25°, -20°, -20°, 0°] → +15.1% gain

# 2. Run basic acados MPC demo (shows limitation)
python examples/demo_yaw_optimization.py

# Output: [-5°, -5°, -5°, 0°] → -0.2% gain (stuck!)

# 3. Run warm start comparison
python examples/example_warm_start.py

# Shows that MPC works better when initialized near optimal

# 4. Run hybrid architecture (in progress)
python examples/hybrid_mpc_example.py

# Two-layer approach: global optimizer + MPC tracking
```

### Understanding the Code

**Main implementation:** `nmpc_windfarm_acados_fixed.py`

**Key classes:**
- `AcadosYawMPC` - Main MPC controller
- `MPCConfig` - Configuration parameters
- `Farm`, `Wind`, `Limits` - Problem setup

**Key methods:**
- `solve_step()` - Compute gradient and solve one MPC step
- `get_delayed_yaw()` - Handle wake propagation delay
- `finite_diff_gradient()` - Numerical gradient via PyWake

**Key parameters:**
```python
cfg = MPCConfig(
    dt=10.0,              # Control timestep [s]
    N_h=10,               # MPC horizon [steps]
    lam_move=10.0,        # Control regularization
    trust_region_weight=1e4,    # QP convexity weight
    direction_bias=5e4,   # Symmetry breaking
    target_weight=1e7,    # For tracking mode (>1e6 = pure tracking)
)
```

---

## Repository Structure

```
mpcrl/
├── nmpc_windfarm_acados_fixed.py    # Current acados implementation ⭐
│
├── mpcrl/                            # Original MPC+RL package
│   ├── archive/                      # Old implementations
│   └── ...
│
├── examples/                         # Working examples
│   ├── hybrid_mpc_example.py        # Two-layer architecture
│   ├── demo_yaw_optimization.py     # Basic MPC demo
│   ├── example_warm_start.py        # Warm start comparison
│   └── archive/                      # Experimental demos
│
├── tests/                            # Test scripts
│   ├── test_optimal_yaw.py          # Ground truth via grid search
│   ├── test_gradient_*.py           # Gradient debugging
│   └── test_long_horizon.py         # Horizon experiments
│
├── docs/                             # Documentation ⭐
│   ├── README.md                    # This file
│   ├── INDEX.md                     # Documentation guide
│   ├── FINAL_RECOMMENDATIONS.md     # Solutions & recommendations
│   ├── LINEARIZATION_LIMITATION.md  # Why gradient MPC fails
│   ├── MPC_ALTERNATIVES.md          # Alternative approaches
│   └── archive/                      # Old documentation
│
├── results/                          # Experiment results
│   └── plots/                        # Figures and visualizations
│
├── scripts/                          # Utility scripts
│
└── data/                             # Input data
```

---

## Next Steps

### Immediate (This Week)
1. ✅ Fix numerical stability → **DONE**
2. ✅ Fix gradient computation → **DONE**
3. ⚙️ **Fix pure tracking mode** → IN PROGRESS
4. 📊 Validate hybrid architecture achieves +15% gain

### Short Term (1-2 Weeks)
1. Build lookup table for various wind conditions
2. Benchmark hybrid architecture vs model-free MPC
3. Document final performance comparison

### Medium Term (1-2 Months)
1. **Option A:** Implement sample-based MPC (MPPI) - no gradients needed
2. **Option B:** Train neural network surrogate for nonlinear MPC
3. Write comparison paper: Model-free vs Model-based MPC
4. Field validation (if real data available)

---

## References

### Academic Papers
- Fleming et al. (2019). "Field test of wake steering" _Wind Energy Science_
- Mesbah (2016). "Stochastic Model Predictive Control" _Automatica_

### Software
- **acados:** https://docs.acados.org/
- **PyWake:** https://topfarm.pages.windenergy.dtu.dk/PyWake/
- **CasADi:** https://web.casadi.org/

---

## Summary

**What we had:** Model-free MPC+RL working well but slow

**What we're building:** Fast model-based MPC with acados for real-time control

**What we learned:** Gradient-based MPC can't find optimal from scratch (fundamental limitation)

**What we're doing:** Hybrid architecture - global optimizer + MPC tracking

**Current status:** Numerical stability solved, tracking convergence in progress

**Expected outcome:** <1ms solve times with +15% power gain (matching optimal)

---

**Last Updated:** 2025-10-20
**Implementation:** `nmpc_windfarm_acados_fixed.py`
**Status:** Hybrid architecture in active development
**Contact:** See main README.md for contribution guidelines
