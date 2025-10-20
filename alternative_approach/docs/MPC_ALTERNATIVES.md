# MPC-Based Approaches for Nonlinear Wind Farm Control

**Date:** 2025-10-20

## Problem Summary

Current gradient-based MPC fails because:
- ❌ Linearization at ψ=0° misses nonlinear power gains at ψ=20-25°
- ❌ Local gradients too weak to drive exploration
- ❌ Can't escape local minima

**Question:** What MPC approaches can handle this?

---

## Option 1: Nonlinear MPC (Full NMPC) ⭐⭐⭐⭐⭐

### Description
Solve the **nonlinear** optimization problem directly without linearization:

```
minimize:  -Σ P(ψ_k) + λ·Σ u_k²
subject to:  ψ_{k+1} = ψ_k + u_k·dt
             |u_k| ≤ u_max
             ψ_min ≤ ψ_k ≤ ψ_max
```

Where `P(ψ)` is the actual PyWake power (not linearized).

### acados Implementation

acados already supports this! We just need to:

1. **Define nonlinear external cost function** (instead of linearizing)
2. **Use Gauss-Newton or exact Hessian** for better curvature handling
3. **Multiple shooting** to handle the nonlinearity

```python
# Instead of linearizing cost, use CasADi to define nonlinear cost
import casadi as ca

# Create CasADi function for power (would need to approximate PyWake)
def power_casadi(psi):
    # Option 1: Polynomial approximation of PyWake
    # Option 2: Neural network approximation
    # Option 3: Spline interpolation from lookup table
    return power_expression(psi)

ocp.model.cost_expr_ext_cost = -power_casadi(x)  # Nonlinear!
```

### Pros
- ✅ Can find global optimum if SQP converges
- ✅ No linearization error
- ✅ Still fast (~1-10ms if model is simple)

### Cons
- ❌ Needs differentiable power model (PyWake is too slow for this)
- ❌ Still local optimization (depends on initial guess)
- ⚠️ May need surrogate model

### Feasibility: **HIGH** (with surrogate model)

**Implementation approach:**
1. Pre-compute power on a grid: `P[ψ0, ψ1, ψ2, ψ3]`
2. Fit neural network or polynomial: `P_approx(ψ) ≈ P_true(ψ)`
3. Use `P_approx` in acados with CasADi
4. Validate against true PyWake periodically

---

## Option 2: Sample-Based MPC (MPPI/CEM) ⭐⭐⭐⭐

### Description
Instead of gradients, use **sampling**:

```python
# Model Predictive Path Integral (MPPI)
for iteration in range(n_iterations):
    # 1. Sample K trajectories around current plan
    trajectories = sample_around(current_plan, noise_std)

    # 2. Evaluate cost (call PyWake for each)
    costs = [evaluate_trajectory(traj) for traj in trajectories]

    # 3. Weight samples by cost
    weights = softmax(-costs / temperature)

    # 4. Update plan as weighted average
    current_plan = weighted_average(trajectories, weights)
```

### Characteristics
- **No gradients needed!** Just cost evaluations
- Works with black-box simulators (PyWake directly)
- Handles nonlinearity and non-convexity naturally

### MPPI Parameters
```python
K = 100  # Number of samples per iteration
horizon = 20  # Planning horizon
iterations = 5  # Refinement iterations
noise_std = 5.0  # Exploration (degrees)
temperature = 10.0  # Selection sharpness
```

### Pros
- ✅ No linearization
- ✅ Can use PyWake directly (no surrogate needed)
- ✅ Handles multimodality
- ✅ Simple to implement

### Cons
- ❌ Computationally expensive (K × PyWake evaluations)
- ❌ Slower than gradient-based (~1-5 seconds per step)
- ⚠️ Needs parallelization for real-time control

### Feasibility: **MEDIUM-HIGH**

**Computational cost:**
- K=100 samples × 20 horizon × 4 turbines = 8,000 PyWake calls
- With parallelization: ~0.5-2 seconds per MPC step
- Acceptable for 10s control timesteps

---

## Option 3: Hybrid Global-Local MPC ⭐⭐⭐⭐⭐

### Description
Combine **coarse global search** with **fine local MPC**:

```python
# STAGE 1: Global optimization (offline or slow online)
optimal_yaw = global_optimizer.find_optimum(wind_conditions)
# Uses: Genetic Algorithm, PSO, Grid Search, etc.
# Runtime: 10-60 seconds (acceptable for slowly-changing wind)

# STAGE 2: Local MPC tracking (fast online)
controller.set_target(optimal_yaw)
controller.track_with_disturbance_rejection()
# Runtime: <1ms per step
```

### Implementation

**Global layer** (updated every 1-10 minutes):
```python
from scipy.optimize import differential_evolution

def objective(yaw):
    return -pywake_farm_power(wf_model, layout, U, theta, yaw)

# Run global search when wind changes
result = differential_evolution(
    objective,
    bounds=[(-25, 25)] * 3 + [(0, 0)],  # Last turbine fixed at 0
    maxiter=50,
    workers=4  # Parallel
)
optimal_yaw = result.x
```

**Local MPC** (updated every 5-10 seconds):
```python
# Track optimal setpoint with strong weighting
cfg = MPCConfig(
    target_weight=1e5,  # Strong tracking
    lam_move=10.0,      # Smooth control
)

controller.set_target(optimal_yaw)
# MPC handles: constraints, rate limits, disturbances
```

### Pros
- ✅ ✅ ✅ Best of both worlds
- ✅ Global layer finds optimal (no local minima)
- ✅ Local MPC is fast and handles constraints
- ✅ Minimal changes to existing code

### Cons
- ⚠️ Two-layer complexity
- ⚠️ Global layer adds latency (but wind changes slowly)

### Feasibility: **VERY HIGH** ⭐ **RECOMMENDED**

---

## Option 4: Receding Horizon with Multiple Starts ⭐⭐⭐

### Description
Run gradient-based MPC from **multiple initial guesses**, pick best:

```python
initial_guesses = [
    np.array([0, 0, 0, 0]),      # Baseline
    np.array([-10, -10, -10, 0]), # Moderate
    np.array([-20, -20, -20, 0]), # Aggressive
    np.array([10, 10, 10, 0]),   # Opposite direction
    np.array([-25, -20, -15, 0]), # From lookup table
]

results = []
for guess in initial_guesses:
    controller.set_state(guess)
    result = controller.step()
    results.append(result)

# Pick best power
best_result = max(results, key=lambda r: r['power'])
controller.apply_control(best_result['psi_plan'])
```

### Pros
- ✅ Simple modification to existing code
- ✅ Gradient-based speed for each start
- ✅ Better exploration than single start

### Cons
- ❌ N× computational cost (N starts)
- ⚠️ No guarantee to find global optimum
- ⚠️ Still may miss optimal if not in start set

### Feasibility: **HIGH** (quick fix)

**Runtime:** 5 starts × 0.5ms = 2.5ms (acceptable)

---

## Option 5: Learning-Enhanced MPC ⭐⭐⭐⭐

### Description
Use machine learning to improve MPC:

### Approach A: Learn Power Surrogate
```python
# Train neural network: P_approx = NN(ψ)
# Dataset: Run PyWake offline on grid
# Use in acados as differentiable cost

model = train_power_model(pywake_dataset)
# Now use in nonlinear MPC (Option 1)
```

### Approach B: Learn Optimal Policy
```python
# Use Reinforcement Learning (e.g., SAC, PPO) to learn:
# π(ψ_target | wind_conditions)

# Then MPC tracks learned setpoint:
target = learned_policy(wind_speed, wind_direction)
controller.track_target(target)
```

### Approach C: Learn Cost-to-Go
```python
# Train: V(ψ, wind) = expected future power
# Use as terminal cost in MPC:

ocp.model.cost_expr_ext_cost_e = -learned_value_function(x)
```

### Pros
- ✅ Fast once trained
- ✅ Can capture nonlinear structure
- ✅ Learns from data

### Cons
- ❌ Requires training data
- ❌ Model accuracy critical
- ⚠️ Generalization to new conditions

### Feasibility: **MEDIUM** (research direction)

---

## Option 6: Trajectory Optimization (Collocation) ⭐⭐⭐

### Description
Solve entire trajectory as one large NLP:

```python
from pyomo import ConcreteModel, Var, Objective, Constraint

model = ConcreteModel()
model.t = range(N_horizon)
model.psi = Var(model.t, bounds=(-25, 25))
model.u = Var(model.t, bounds=(-0.5, 0.5))

# Nonlinear objective (use surrogate)
model.obj = Objective(
    expr=sum(-power_surrogate(model.psi[t]) for t in model.t)
)

# Dynamics constraints
model.dynamics = Constraint(...)

# Solve with IPOPT (nonlinear solver)
solver = SolverFactory('ipopt')
solver.solve(model)
```

### Pros
- ✅ Handles nonlinearity
- ✅ Simultaneous optimization (no sequential)
- ✅ mature tools (IPOPT, CasADi)

### Cons
- ❌ Slower than acados (~10-100ms)
- ❌ Still needs surrogate model
- ⚠️ Warm-starting critical

### Feasibility: **MEDIUM-HIGH**

---

## Computational Cost Comparison

| Approach | Per-Step Time | Setup Time | Optimality |
|----------|---------------|------------|------------|
| **Current (linearized)** | 0.3 ms | 0 | ❌ Poor |
| **Nonlinear MPC** | 1-5 ms | Hours (train surrogate) | ✅ Good |
| **MPPI/CEM** | 0.5-2 s | 0 | ✅✅ Best |
| **Hybrid Global-Local** | 0.3 ms (local) | 10-60 s per update | ✅✅ Best |
| **Multiple Starts** | 2-5 ms | 0 | ⚠️ Medium |
| **Learning-enhanced** | 0.5-2 ms | Days (training) | ✅ Good |
| **Trajectory Opt** | 10-100 ms | Hours (surrogate) | ✅ Good |

---

## Recommendations

### 🥇 **Best for Production: Hybrid Global-Local** (Option 3)

**Why:**
- Global layer finds true optimal
- Local MPC handles constraints, smoothness, disturbances
- Minimal risk, proven approach
- Fast enough for real-time (wind changes slowly)

**Implementation Priority:**
1. Add global optimizer (scipy.optimize.differential_evolution)
2. Add target_tracking mode to MPC
3. Update global layer every 1-10 minutes
4. Use MPC for second-by-second control

### 🥈 **Best for Research: Sample-Based MPC** (Option 2)

**Why:**
- No surrogate model needed
- Handles full nonlinearity
- Parallelizable (important for scaling)

**Use if:**
- Have GPU/multi-core compute
- Want to explore different power models
- Research setting (not production-critical)

### 🥉 **Quick Fix: Multiple Starts** (Option 4)

**Why:**
- Minimal code changes
- 10× better than current single start
- Fast to implement and test

**Use if:**
- Need improvement NOW
- Don't want to redesign architecture
- Acceptable to not reach true optimal

---

## Implementation Roadmap

### Phase 1: Quick Wins (1 week)
1. ✅ Implement multiple-starts MPC
2. ✅ Add target tracking capability
3. ✅ Test with known optimal setpoints

### Phase 2: Hybrid System (2-3 weeks)
1. Integrate global optimizer (scipy)
2. Two-layer architecture
3. Adaptive update frequency based on wind changes

### Phase 3: Advanced (1-2 months)
1. Train neural network power surrogate
2. Implement nonlinear MPC with surrogate
3. OR implement MPPI with parallelization

---

## Code Examples Available

I can provide implementation examples for:
- ✅ Multiple-starts MPC (modify existing)
- ✅ Hybrid global-local architecture
- ✅ Basic MPPI implementation
- ⚠️ Neural network surrogate (requires training data)

**Which approach interests you most?**
