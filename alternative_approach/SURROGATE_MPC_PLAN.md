# Surrogate-Based MPC Implementation Plan

**Created:** 2025-11-06
**Goal:** Implement pure MPC solution using PyTorch neural network + l4casadi + acados
**Expected Performance:** ~15% power gain (matching optimal)

---

## Overview

We will build a **learned model-based MPC** that:
1. Uses a PyTorch neural network to approximate PyWake power model
2. Converts the network to CasADi via l4casadi
3. Uses it as the objective function in acados MPC
4. Achieves near-optimal performance without needing PyWake at runtime

**Key Advantage:** Pure MPC solution, no hybrid architecture needed, fast (<10ms per solve)

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  OFFLINE: Training Phase                            │
│  ────────────────────────────────────────────────   │
│  1. Generate dataset (PyWake simulations)           │
│  2. Train PyTorch neural network                    │
│  3. Validate model accuracy                         │
│  4. Export model                                    │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  INTEGRATION: l4casadi                              │
│  ────────────────────────────────────────────────   │
│  1. Load PyTorch model                              │
│  2. Convert to CasADi using l4casadi                │
│  3. Test gradients and performance                  │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  ONLINE: MPC Control                                │
│  ────────────────────────────────────────────────   │
│  1. acados uses NN as nonlinear cost function       │
│  2. SQP solver finds optimal control                │
│  3. Fast (<10ms) + Optimal (~15% gain)              │
└─────────────────────────────────────────────────────┘
```

---

## Phase 1: Dataset Generation

### 1.1 Dataset Requirements

**Goal:** Generate diverse training data covering the full operating space

**Inputs (features):**
- `yaw_0, yaw_1, yaw_2, yaw_3` - Yaw angles for 4 turbines (degrees)
- `wind_speed` - Wind speed (m/s)
- `wind_direction` - Wind direction (degrees)

**Output (target):**
- `total_power` - Total farm power output (MW)

**Dataset size:** 100,000 samples
- Training: 80,000 (80%)
- Validation: 10,000 (10%)
- Test: 10,000 (10%)

**Sampling strategy:**
- **Yaw angles:** Latin hypercube sampling in [-30°, 30°] for turbines 0-2, [0°, 0°] for turbine 3
- **Wind speed:** Uniform sampling in [6, 12] m/s
- **Wind direction:** Focus on 270° (main), with some variation ±10°

**Why this works:**
- Covers full yaw space systematically
- Includes boundary regions (important for constraints)
- Balanced representation of operating conditions

### 1.2 Implementation

**File:** `scripts/generate_dataset_v2.py`

**Features:**
- Parallel PyWake evaluations (multiprocessing)
- Progress tracking and ETA
- Automatic train/val/test split
- Data validation and statistics
- Save to HDF5 for efficient loading

**Runtime:** ~2-4 hours on 8 cores

---

## Phase 2: Neural Network Training

### 2.1 Network Architecture

**Model type:** Fully connected feedforward network

**Architecture:**
```python
Input: [yaw_0, yaw_1, yaw_2, yaw_3, wind_speed, wind_direction]  # 6 features
   ↓
Dense(64) + Tanh
   ↓
Dense(64) + Tanh
   ↓
Dense(32) + Tanh
   ↓
Dense(1)  # Output: total_power
```

**Why this architecture:**
- **Tanh activation**: Smooth, differentiable (good for gradients in MPC)
- **3 hidden layers**: Enough capacity for nonlinearity, not too deep
- **64-64-32 neurons**: Standard size for this problem
- **No BatchNorm/Dropout**: Keep it simple for CasADi conversion

### 2.2 Training Strategy

**Loss function:** MSE (Mean Squared Error)
```python
loss = mean((predicted_power - actual_power)^2)
```

**Optimizer:** Adam with learning rate scheduling
- Initial LR: 1e-3
- Decay on plateau
- Max epochs: 1000 (with early stopping)

**Normalization:** Critical for training stability
```python
# Input normalization (Z-score)
yaw_normalized = (yaw - mean_yaw) / std_yaw
wind_speed_normalized = (ws - mean_ws) / std_ws

# Output normalization
power_normalized = (power - mean_power) / std_power
```

**Target accuracy:**
- **MAE < 1%** of mean power (< 50 kW for 5 MW farm)
- **Gradient agreement > 95%** with PyWake

### 2.3 Implementation

**File:** `scripts/train_surrogate_v2.py`

**Features:**
- PyTorch Lightning for clean training loop
- Automatic checkpointing (best model)
- TensorBoard logging
- Validation metrics tracking
- Gradient validation against PyWake

**Runtime:** ~1-2 hours on GPU (or 4-6 hours on CPU)

---

## Phase 3: l4casadi Integration

### 3.1 What is l4casadi?

**l4casadi** = Learning 4 CasADi

It's a library that automatically converts PyTorch models to CasADi operations, enabling:
- Use PyTorch models in optimization (acados, casadi)
- Automatic differentiation (Jacobian, Hessian)
- Efficient evaluation

**GitHub:** https://github.com/Tim-Salzmann/l4casadi

### 3.2 Conversion Process

**Steps:**
1. Load trained PyTorch model
2. Wrap it with l4casadi
3. Generate CasADi function
4. Test evaluation and gradients
5. Export for acados

**File:** `scripts/export_l4casadi_model.py`

**Example:**
```python
import l4casadi as l4c
import casadi as ca

# Load PyTorch model
model = torch.load('models/power_surrogate.pth')

# Wrap with l4casadi
l4c_model = l4c.L4CasADi(model, model_expects_batch_dim=False)

# Create CasADi function
x = ca.SX.sym('x', 6)  # 6 inputs
power_casadi = l4c_model(x)

# Test gradients
grad = ca.gradient(power_casadi, x)
```

### 3.3 Implementation

**Features:**
- Model loading and validation
- CasADi function generation
- Gradient correctness checks
- Performance benchmarking
- Export to acados-compatible format

**Runtime:** < 1 minute

---

## Phase 4: Nonlinear MPC with Surrogate

### 4.1 acados OCP Formulation

**State:** `x = [yaw_0, yaw_1, yaw_2, yaw_3]` (4 turbines)

**Control:** `u = [dyaw_0, dyaw_1, dyaw_2, dyaw_3]` (yaw rate)

**Dynamics:** Simple integrator
```
x[k+1] = x[k] + u[k] * dt
```

**Objective:** Maximize power (minimize negative power)
```python
# Stage cost (at each time step k)
J_k = -power_surrogate(x[k], wind_speed, wind_direction) + λ * ||u[k]||²

# Total cost
J = sum(J_k for k in 0..N-1) + J_terminal
```

**Constraints:**
- State bounds: `-30° ≤ yaw ≤ 30°`
- Control bounds: `-0.3°/s ≤ dyaw ≤ 0.3°/s`

**Solver:** SQP (Sequential Quadratic Programming)
- Better for nonlinear problems than RTI
- Can handle the nonlinear surrogate cost

### 4.2 Key Difference from Current Implementation

**Current (linearized):**
```python
# Compute gradient at current state
grad = finite_diff_gradient(current_yaw)

# Linearize cost around current point
J ≈ grad^T * (yaw - current_yaw) + regularization
```

**New (nonlinear with surrogate):**
```python
# Use surrogate directly as cost
J = -power_surrogate(yaw, wind_speed, wind_direction) + regularization

# acados uses automatic differentiation via CasADi
# No linearization! Full nonlinear optimization
```

### 4.3 Implementation

**File:** `alternative_approach/nmpc_surrogate.py`

**Key components:**
1. Load l4casadi model
2. Set up acados OCP with nonlinear cost
3. Configure SQP solver
4. Warm starting strategy
5. Interface compatible with current code

**API:**
```python
from nmpc_surrogate import SurrogateMPC

# Initialize
controller = SurrogateMPC(
    model_path='models/power_surrogate_casadi.pkl',
    farm=farm,
    wind=wind,
    config=MPCConfig(...)
)

# Use (same interface as current)
result = controller.step(current_yaw)
optimal_yaw = result['psi_plan'][0]
```

---

## Phase 5: Validation

### 5.1 Validation Tests

**Test 1: Model Accuracy**
- Compare surrogate predictions vs PyWake
- On test set (10,000 samples)
- Metrics: MAE, RMSE, R²
- **Target:** MAE < 1%, R² > 0.99

**Test 2: Gradient Correctness**
- Compare surrogate gradients vs PyWake finite differences
- At 100 random points
- **Target:** > 95% agreement (cosine similarity)

**Test 3: MPC Performance**
- Run MPC with surrogate on standard test case
- Compare with grid search optimal (15.1% gain)
- **Target:** Achieve > 13% gain (> 85% of optimal)

**Test 4: Solve Time**
- Measure acados solve time with surrogate cost
- **Target:** < 10ms per solve

**Test 5: Multi-Condition**
- Test on various wind conditions
- Wind speed: 6, 8, 10, 12 m/s
- Wind direction: 270°, 0°, 90°, 180°
- **Target:** Consistent performance across conditions

### 5.2 Implementation

**Files:**
- `tests/test_surrogate_accuracy.py`
- `tests/test_surrogate_gradients.py`
- `tests/test_surrogate_mpc_performance.py`
- `tests/test_surrogate_mpc_speed.py`

---

## Phase 6: Documentation & Examples

### 6.1 Documentation

**Create/Update:**
1. `SURROGATE_MPC_USAGE.md` - How to use the surrogate MPC
2. `TRAINING_GUIDE.md` - How to retrain the model
3. Update `README.md` - Add surrogate approach
4. Update `docs/INDEX.md` - Add surrogate docs

### 6.2 Examples

**Create:**
1. `examples/demo_surrogate_mpc.py` - Basic usage
2. `examples/compare_surrogate_vs_hybrid.py` - Performance comparison
3. `examples/visualize_surrogate.py` - Visualize learned model

---

## Timeline

| Phase | Task | Duration | Dependencies |
|-------|------|----------|--------------|
| 1 | Dataset generation | 1-2 days | PyWake installed |
| 2 | Model training | 1-2 days | Phase 1 complete |
| 3 | l4casadi integration | 1 day | Phase 2 complete |
| 4 | Nonlinear MPC | 2-3 days | Phase 3 complete |
| 5 | Validation | 1-2 days | Phase 4 complete |
| 6 | Documentation | 1 day | Phase 5 complete |
| **Total** | **End-to-end** | **1-2 weeks** | |

---

## Dependencies

### Python Packages

**Already have:**
- `numpy`
- `torch` (PyTorch)
- `py_wake`
- `casadi`
- `acados_template` (acados Python interface)

**Need to install:**
```bash
pip install l4casadi
pip install pytorch-lightning
pip install h5py
pip install tensorboard
pip install scikit-learn  # for data utilities
```

### System Requirements

**For training:**
- CPU: 8+ cores recommended (for parallel dataset generation)
- RAM: 16 GB
- GPU: Optional but recommended (10x faster training)
- Disk: ~2 GB for dataset + models

**For inference (MPC):**
- Same as current (acados requirements)
- No GPU needed for inference

---

## File Structure

```
alternative_approach/
├── SURROGATE_MPC_PLAN.md          ⭐ This file
│
├── data/
│   ├── surrogate_dataset.h5       # Generated dataset (100k samples)
│   └── dataset_stats.json         # Dataset statistics
│
├── models/
│   ├── power_surrogate.pth        # Trained PyTorch model
│   ├── power_surrogate_casadi.pkl # l4casadi converted model
│   └── training_config.json       # Training hyperparameters
│
├── scripts/
│   ├── generate_dataset_v2.py     ⭐ Phase 1
│   ├── train_surrogate_v2.py      ⭐ Phase 2
│   ├── export_l4casadi_model.py   ⭐ Phase 3
│   └── validate_surrogate.py      ⭐ Phase 5
│
├── surrogate_module/
│   ├── dataset.py                 # Dataset utilities
│   ├── model.py                   # PyTorch model definition
│   ├── training.py                # Training utilities
│   └── l4casadi_wrapper.py        # l4casadi integration
│
├── nmpc_surrogate.py              ⭐ Phase 4 - Main MPC controller
│
├── examples/
│   ├── demo_surrogate_mpc.py      # Basic demo
│   └── compare_approaches.py      # Compare surrogate vs hybrid
│
└── tests/
    ├── test_surrogate_accuracy.py
    ├── test_surrogate_mpc.py
    └── test_integration.py
```

---

## Expected Results

### Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Surrogate MAE** | < 50 kW | < 1% of 5 MW farm |
| **Gradient agreement** | > 95% | Ensures correct optimization direction |
| **MPC power gain** | > 13% | > 85% of optimal (15.1%) |
| **MPC solve time** | < 10ms | Real-time capable |
| **Convergence rate** | > 95% | Robust solver |

### Why This Will Work

1. **No linearization error** - NN approximates full nonlinear landscape
2. **Good initialization** - Can warm start from previous solution
3. **Smooth function** - NN with tanh is smooth and differentiable
4. **Fast evaluation** - NN forward pass ~0.1ms
5. **Proven approach** - Used in robotics, autonomous driving, etc.

---

## Risk Mitigation

### Risk 1: Surrogate Not Accurate Enough
**Likelihood:** Low
**Mitigation:**
- Use large dataset (100k samples)
- Validate thoroughly before MPC integration
- Can increase network size if needed

### Risk 2: acados Solver Issues with Nonlinear Cost
**Likelihood:** Medium
**Mitigation:**
- Use SQP instead of RTI
- Tune solver tolerance and iterations
- Warm start from previous solution
- Fall back to hybrid approach if needed

### Risk 3: l4casadi Integration Issues
**Likelihood:** Low
**Mitigation:**
- l4casadi is well-tested with acados
- Extensive gradient validation
- Can manually implement CasADi conversion if needed

### Risk 4: Training Time Too Long
**Likelihood:** Low
**Mitigation:**
- Use GPU if available
- Start with smaller dataset (50k) for prototyping
- Can use pre-trained model from literature

---

## Success Criteria

**Phase 1 Complete:**
- [ ] 100k samples generated
- [ ] Dataset validated (no NaNs, good coverage)
- [ ] Train/val/test split created

**Phase 2 Complete:**
- [ ] Model trained (loss converged)
- [ ] Test MAE < 1%
- [ ] Gradients agree with PyWake > 95%

**Phase 3 Complete:**
- [ ] Model converted to CasADi
- [ ] Gradients match PyTorch
- [ ] Evaluation time < 1ms

**Phase 4 Complete:**
- [ ] acados MPC with surrogate working
- [ ] Solver converges reliably
- [ ] Solve time < 10ms

**Phase 5 Complete:**
- [ ] All tests passing
- [ ] Power gain > 13%
- [ ] Multi-condition validation done

**Phase 6 Complete:**
- [ ] Documentation complete
- [ ] Examples working
- [ ] Ready for production use

---

## Next Steps

1. **Immediate:** Set up dependencies
   ```bash
   pip install l4casadi pytorch-lightning h5py tensorboard scikit-learn
   ```

2. **Phase 1:** Generate dataset
   ```bash
   python scripts/generate_dataset_v2.py --n_samples 100000
   ```

3. **Phase 2:** Train model
   ```bash
   python scripts/train_surrogate_v2.py --dataset data/surrogate_dataset.h5
   ```

4. **Phase 3-6:** Follow plan sequentially

---

## Questions to Resolve

- [ ] Do we have GPU available for training?
- [ ] What's the acceptable training time?
- [ ] Should we include turbulence intensity as input?
- [ ] Do we want to model individual turbine powers or just total?
- [ ] Should we handle multiple wind directions or just 270°?

---

**Status:** ✅ Plan complete, ready to implement
**Confidence:** High - proven approach with existing infrastructure
**Risk:** Low-Medium - main risks identified and mitigated
