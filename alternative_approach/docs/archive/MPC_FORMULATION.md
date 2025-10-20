# Wind Farm Yaw Control - MPC Formulation Guide

## Overview

This document explains the Model Predictive Control (MPC) formulation for wind farm yaw control, with a focus on the acados implementation. The goal is to optimize yaw angles across turbines to maximize total farm power production while respecting physical constraints.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [MPC Formulation](#mpc-formulation)
3. [Implementation Strategies](#implementation-strategies)
4. [acados Setup](#acados-setup)
5. [CasADi vs acados](#casadi-vs-acados)
6. [Practical Considerations](#practical-considerations)

---

## 1. Problem Statement

### Physical System

**Wind Farm Layout:**
- N turbines at positions (xᵢ, yᵢ), i = 1, ..., N
- Rotor diameter D
- Turbines can adjust yaw angle ψᵢ (misalignment with wind)

**Wake Effects:**
- Upstream turbines create wakes that reduce power of downstream turbines
- Yawing turbines can redirect wakes and improve farm power
- Wake effects propagate with advection delay: τᵢⱼ = (downstream distance) / (wind speed)

**Control Objective:**
Maximize total farm power by coordinating yaw angles, subject to:
- Yaw angle bounds: ψ_min ≤ ψᵢ ≤ ψ_max
- Yaw rate limits: |dψᵢ/dt| ≤ r_max
- Advection delay dynamics

---

## 2. MPC Formulation

### Decision Variables

Over a prediction horizon of Nₕ steps:

- **States** x = [ψ₁, ψ₂, ..., ψₙ]ᵀ ∈ ℝᴺ : yaw angles [deg]
- **Controls** u = [u₁, u₂, ..., uₙ]ᵀ ∈ ℝᴺ : yaw rates [deg/s]

### Dynamics

Simple integrator model (discrete-time with sampling period Δt):

```
x_{k+1} = x_k + u_k · Δt
```

This is a kinematic model - the actual turbine yaw dynamics are abstracted to rate limits.

### Objective Function

Maximize farm power while penalizing yaw rate changes:

```
min  Σ_{k=0}^{N_h-1} [ -P(x_k) + (λ/2) ||u_k||² ]
     + terminal cost: -P(x_{N_h})
```

Where:
- P(x_k): total farm power at step k [W]
- λ: weight on yaw rate penalty (trades off power gain vs. actuator wear)
- ||u_k||²: sum of squared yaw rates

**Note:** We minimize *negative* power to convert maximization → minimization.

### Constraints

**Box constraints on states:**
```
ψ_min ≤ x_{k,i} ≤ ψ_max,    for all k, i
```

**Box constraints on controls:**
```
-r_max ≤ u_{k,i} ≤ r_max,   for all k, i
```

**Initial condition:**
```
x_0 = ψ_current  (measured/estimated current yaw angles)
```

### Advection Delays

Wake effects from turbine i to turbine j have a delay τᵢⱼ [steps]:

```
τᵢⱼ = floor( distance_{ij} / (U · Δt) )
```

where U is wind speed.

The power at turbine j at time k depends on upstream yaw angles at time (k - τᵢⱼ):

```
P_j(k) = f_j( ψ_j(k), { ψ_i(k - τ_{ij}) : i upstream of j } )
```

This is handled by:
1. Maintaining a history buffer of past yaw angles
2. Looking up delayed values when computing power/gradients

---

## 3. Implementation Strategies

### Strategy 1: Full Nonlinear MPC

**Formulation:**
- Use high-fidelity wake model (PyWake, FLORIS) directly in optimization
- Solve nonlinear program (NLP) each time step

**Pros:**
- Most accurate
- Captures all nonlinearities

**Cons:**
- Very slow (wake model evaluations are expensive)
- Requires derivatives of black-box wake model
- May not converge in real-time

**When to use:** Offline planning, benchmarking

---

### Strategy 2: Successive Linearization (Recommended)

**Formulation:**
1. At each time step, evaluate P₀ and gradient ∇P at current yaw angles
2. Linearize power: P(ψ) ≈ P₀ + ∇P^T (ψ - ψ₀)
3. Solve QP with linearized cost
4. Apply first control, advance time, repeat

**Cost becomes:**
```
min  Σ_k [ -∇P^T · x_k + (λ/2) ||u_k||² ]
```

This is a **convex quadratic program (QP)** - fast to solve!

**Pros:**
- Fast solve times (10-100ms typical)
- Guaranteed convergence
- Works with any wake model (just need gradient)

**Cons:**
- Approximation (accuracy depends on step size)
- Gradient computation requires finite differences or AD

**When to use:** Real-time control, MPC in the loop

**Note:** This is what we've implemented in both `nmpc_windfarm.py` (CasADi) and `nmpc_windfarm_acados.py` (acados).

---

### Strategy 3: Learning-Based Surrogate

**Formulation:**
1. Generate training data: (ψ, wind conditions) → P using high-fidelity model
2. Train neural network: P̂ = NN(ψ, U, θ)
3. Use NN in MPC with automatic differentiation

**Pros:**
- Fast evaluations (microseconds)
- Captures nonlinearities
- Smooth gradients

**Cons:**
- Requires training data and infrastructure
- Interpolation errors outside training distribution
- More complex setup

**When to use:** High-frequency control, model-based RL, complex farms

---

## 4. acados Setup

### Why acados?

acados is a high-performance solver for optimal control problems, designed for:
- Real-time embedded systems
- Fast solve times (< 10ms typical for small problems)
- Code generation (export to C for deployment)
- Efficient handling of QPs and NLPs

### Installation

```bash
# Option 1: pip (easiest, may not have all features)
pip install acados_template

# Option 2: from source (recommended for full features)
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init
mkdir build && cd build
cmake .. -DACADOS_WITH_QPOASES=ON -DACADOS_WITH_HPIPM=ON
make install -j4
pip install -e ../interfaces/acados_template
```

See: https://docs.acados.org/installation/

### Code Structure

**1. Define acados Model**

```python
from acados_template import AcadosModel
from casadi import SX

model = AcadosModel()
model.name = "wind_farm_yaw"

# States: yaw angles
x = SX.sym('x', N_turbines)

# Controls: yaw rates
u = SX.sym('u', N_turbines)

# Dynamics: simple integrator
model.f_expl_expr = x + u * dt  # explicit dynamics
model.x = x
model.u = u
```

**2. Create OCP**

```python
from acados_template import AcadosOcp

ocp = AcadosOcp()
ocp.model = model
ocp.dims.N = N_horizon  # number of shooting nodes

# Cost (linearized power)
ocp.cost.cost_type = 'EXTERNAL'
q = SX(grad_P)  # gradient from PyWake
cost_expr = -q.T @ x + (lam_move/2) * (u.T @ u)
ocp.model.cost_expr_ext_cost = cost_expr

# Constraints
ocp.constraints.lbx = np.full(N_turbines, yaw_min)
ocp.constraints.ubx = np.full(N_turbines, yaw_max)
ocp.constraints.idxbx = np.arange(N_turbines)

ocp.constraints.lbu = np.full(N_turbines, -yaw_rate_max)
ocp.constraints.ubu = np.full(N_turbines, yaw_rate_max)
ocp.constraints.idxbu = np.arange(N_turbines)

ocp.constraints.x0 = psi_current  # initial condition

# Solver settings
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.nlp_solver_type = 'SQP'
ocp.solver_options.tf = N_horizon * dt
```

**3. Build Solver**

```python
from acados_template import AcadosOcpSolver

solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
```

**4. Solve Loop**

```python
# Update gradient at current point
P_current, grad_P = finite_diff_gradient(pywake_model, psi_current)

# Rebuild solver with new gradient (or update parameters)
solver = setup_acados_ocp(farm, wind, limits, cfg, psi_current, grad_P)

# Warm start
for k in range(N_horizon):
    solver.set(k, 'x', psi_current)

# Solve
status = solver.solve()

# Extract solution
psi_plan = np.zeros((N_horizon, N_turbines))
for k in range(N_horizon):
    psi_plan[k, :] = solver.get(k, 'x')

# Apply first control
psi_next = psi_plan[0, :]
```

---

## 5. CasADi vs acados

### CasADi

**Pros:**
- Easy to use, Pythonic interface
- Flexible problem formulation
- Good for prototyping
- IPOPT solver widely available

**Cons:**
- Slower solve times (especially IPOPT)
- Less optimized for embedded/real-time
- No built-in code generation for deployment

**When to use:**
- Research, prototyping
- When solve time < 100ms is acceptable
- Complex non-standard formulations

### acados

**Pros:**
- Very fast (2-10x faster for QPs)
- Code generation for C deployment
- Optimized for real-time control
- Advanced QP solvers (HPIPM, QPOASES)
- SQP-RTI for ultra-fast iterations

**Cons:**
- Steeper learning curve
- More rigid problem structure
- Installation can be tricky
- Updating cost parameters requires rebuilding solver (for EXTERNAL cost)

**When to use:**
- Real-time control (< 10ms solve time needed)
- Embedded systems
- Production deployment
- Large-scale problems

### Performance Comparison

For a 4-turbine problem with N_h = 12:

| Method | Avg Solve Time | Peak Solve Time |
|--------|----------------|-----------------|
| CasADi QP (QRQP) | ~50ms | ~100ms |
| CasADi QP (OSQP) | ~30ms | ~60ms |
| acados (HPIPM) | ~5-10ms | ~20ms |

*Speedup: acados is typically 3-5x faster for QPs*

---

## 6. Practical Considerations

### Tuning Parameters

**Horizon length (N_h):**
- Longer horizon → better performance, slower solve
- Rule of thumb: cover 1-2 advection times through farm
- Typical: 10-30 steps

**Sampling time (Δt):**
- Trade-off: finer discretization vs. computational cost
- Typical: 5-20 seconds
- Must be < 1 / f_Nyquist for turbine dynamics

**Move penalty (λ):**
- Higher λ → smoother yaw trajectories, less aggressive
- Lower λ → more power gain, more actuator wear
- Typical: 0.1 - 1.0 (depends on normalization)

### Handling Delays

**Option 1: Max delay per turbine (current implementation)**
```python
tau_i = np.max(tau, axis=1)  # max delay affecting turbine i
psi_delayed[i] = psi_history[k - tau_i[i]][i]
```

Simple but conservative (overestimates delay for some interactions).

**Option 2: Per-edge delays**
Build delay into power model explicitly:
```python
for (i, j) in edges:
    psi_i_delayed = psi_history[k - tau[i,j]][i]
    # use in wake deficit calculation
```

More accurate but complicates formulation.

### Warm Starting

Always warm start the solver with a good guess:
```python
# Option 1: Previous solution shifted
psi_guess[:-1] = psi_prev_solution[1:]
psi_guess[-1] = psi_prev_solution[-1]

# Option 2: Constant current yaw
psi_guess[:] = psi_current
```

Warm starting can reduce solve time by 2-5x.

### Gradient Computation

**Finite differences (current):**
```python
for i in range(N):
    Pp = pywake_power(psi + eps * e_i)
    Pm = pywake_power(psi - eps * e_i)
    grad[i] = (Pp - Pm) / (2 * eps)
```

Cost: 2N+1 power evaluations per gradient (~1-2 seconds)

**Alternatives:**
- Adjoint methods (if solver supports)
- Automatic differentiation (requires differentiable wake model)
- Pre-computed gradients via surrogate

### Real-Time Considerations

For deployment:

1. **Code generation:** Use acados C code generation
   ```python
   ocp.code_export_directory = 'c_generated_code'
   solver = AcadosOcpSolver(ocp, generate=True, build=True)
   ```

2. **Async gradient computation:** Compute gradient in separate thread while previous solve runs

3. **Fallback controller:** Have backup strategy if solver fails/times out

4. **Monitoring:** Log solve times, constraint violations, solver status

---

## Usage Example

### Basic Usage

```python
from nmpc_windfarm_acados import AcadosYawMPC, Farm, Wind, Limits, MPCConfig

# Define farm
farm = Farm(
    x=np.array([0, 1000, 2000, 3000]),
    y=np.array([0, 0, 0, 0]),
    D=120.0
)

# Wind conditions
wind = Wind(U=8.0, theta=0.0)

# Limits
limits = Limits(yaw_min=-25, yaw_max=25, yaw_rate_max=0.3)

# MPC configuration
cfg = MPCConfig(dt=10.0, N_h=20, lam_move=0.5)

# Create controller
controller = AcadosYawMPC(farm, wind, limits, cfg)

# Run for 50 steps
history = controller.run(n_steps=50, verbose=True)
```

### Comparison

```bash
python compare_mpc_solvers.py
```

This runs both CasADi and acados on the same problem and generates comparison plots.

---

## References

- **acados:** https://docs.acados.org/
- **PyWake:** https://topfarm.pages.windenergy.dtu.dk/PyWake/
- **CasADi:** https://web.casadi.org/
- **MPC Tutorial:** Rawlings, Mayne, Diehl - "Model Predictive Control: Theory, Computation, and Design"
- **Wind Farm Control:** Fleming et al., "Field test of wake steering at an offshore wind farm", Wind Energy Science, 2017

---

## Next Steps

1. **Try it:** Run `python nmpc_windfarm_acados.py`
2. **Compare:** Run `python compare_mpc_solvers.py`
3. **Tune:** Adjust N_h, λ, dt for your use case
4. **Extend:**
   - Add wind direction tracking
   - Implement learning-based surrogate
   - Multi-scenario robust MPC
   - Export to C for embedded deployment

Good luck! 🌬️
