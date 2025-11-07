# Future Work: WindGym Integration

**Date:** 2025-11-06
**Status:** Notes for future implementation

---

## Current State vs Future Goal

### Current Implementation (Phase 1 - Testing)
- **Dataset generation:** Using PyWake directly
- **Purpose:** Proof of concept, validate surrogate MPC approach
- **Pros:** Fast to implement, well-understood
- **Cons:** Static model, not a real environment

### Future Goal (Phase 2 - Production)
- **Dataset generation:** Use WindGym environment
- **Purpose:** MPC controlling real WindGym environment
- **End goal:** Surrogate-based MPC as controller for WindGym

---

## Why This Makes Sense

### Current Approach (PyWake)
```
PyWake → Generate Dataset → Train Surrogate → MPC uses Surrogate
```
- Good for: Testing the surrogate MPC concept
- Limitation: No closed-loop control, just optimization

### Future Approach (WindGym)
```
WindGym → Generate Dataset → Train Surrogate → MPC controls WindGym
```
- Better: Closed-loop control in realistic environment
- Realistic: WindGym has dynamics, turbulence, etc.
- Testbed: Can evaluate MPC performance properly

---

## Implementation Plan

### Phase 1: Current (PyWake-based) ✅
**Goal:** Validate that surrogate MPC works
**Steps:**
1. Generate dataset from PyWake (simple, static)
2. Train surrogate model
3. Build MPC with surrogate cost
4. Verify it finds optimal yaw angles

**Status:** In progress (testing with small dataset)

### Phase 2: WindGym Integration (Future)
**Goal:** Use MPC to control WindGym environment
**Steps:**
1. Modify dataset generation to use WindGym
2. Retrain surrogate on WindGym data
3. Create WindGym-MPC interface
4. Run closed-loop control experiments

---

## WindGym Dataset Generation

### Changes Needed to `generate_dataset_v2.py`

**Current (PyWake):**
```python
def evaluate_sample(self, args):
    # Build PyWake model
    wf_model, layout = build_pywake_model(x, y, D)

    # Compute power (static)
    power = pywake_farm_power(wf_model, layout, U, theta, yaw)
    return power
```

**Future (WindGym):**
```python
def evaluate_sample(self, args):
    # Create WindGym environment
    env = WindGym(...)

    # Reset with specific conditions
    env.reset(wind_speed=U, wind_direction=theta)

    # Apply yaw angles and step
    obs, reward, done, info = env.step(yaw)

    # Get power
    power = info['total_power']
    return power
```

### Key Differences

| Aspect | PyWake | WindGym |
|--------|--------|---------|
| **Type** | Static model | Dynamic environment |
| **Evaluation** | Single function call | Step-based simulation |
| **Dynamics** | Instantaneous | Turbulence, delays, transients |
| **Data** | Perfect | Realistic with noise |
| **Speed** | Fast | Slower (more realistic) |

### Advantages of WindGym Data

1. **More realistic:** Captures turbulence, dynamics
2. **Better surrogate:** Learns realistic power response
3. **Better MPC:** Trained on data closer to real operation
4. **Testbed:** Can evaluate MPC in simulation before field

---

## MPC-WindGym Control Loop

### Architecture

```
┌─────────────────────────────────────────────┐
│  Offline: Train Surrogate on WindGym Data  │
│  ─────────────────────────────────────────  │
│  1. Run WindGym with various yaw angles    │
│  2. Record (yaw, wind, power) samples      │
│  3. Train neural network surrogate         │
│  4. Export to l4casadi                     │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Online: MPC Controls WindGym               │
│  ─────────────────────────────────────────  │
│  Loop:                                      │
│    1. MPC solves optimal yaw (using NN)    │
│    2. Apply yaw to WindGym                 │
│    3. WindGym steps (realistic dynamics)   │
│    4. Observe new state and power          │
│    5. Repeat                               │
└─────────────────────────────────────────────┘
```

### Implementation

**File:** `nmpc_windgym_control.py` (future)

```python
class WindGymMPCController:
    def __init__(self, surrogate_model, windgym_env):
        self.mpc = SurrogateMPC(surrogate_model)
        self.env = windgym_env

    def run_episode(self, n_steps=1000):
        obs = self.env.reset()

        for t in range(n_steps):
            # Get current state
            current_yaw = obs['yaw_angles']
            wind = obs['wind']

            # MPC computes optimal control
            result = self.mpc.step(current_yaw, wind)
            optimal_yaw = result['psi_plan'][0]

            # Apply to WindGym
            obs, reward, done, info = self.env.step(optimal_yaw)

            # Log performance
            power = info['total_power']

        return results
```

---

## Benefits of This Approach

### 1. Separates Concerns
- **Surrogate learning:** Offline, learns power model
- **MPC optimization:** Online, uses learned model
- **Environment:** WindGym provides realistic testbed

### 2. Enables Future RL Integration
Once MPC is working:
- Add RL layer on top to handle model mismatch
- RL learns corrections to MPC
- Best of both worlds: MPC structure + RL adaptation

### 3. Realistic Evaluation
- Test MPC in simulation before field deployment
- Measure performance under realistic conditions
- Identify issues (delays, turbulence, constraints)

---

## Timeline

### Phase 1: PyWake (Current) - 1 week
- ✅ Dataset generation script
- ✅ Training pipeline
- ✅ l4casadi export
- ✅ MPC implementation
- ⏳ Testing with small dataset
- ⏳ Full dataset (100k samples)
- ⏳ Validation

### Phase 2: WindGym Integration - 2-3 weeks
1. Modify dataset generation for WindGym
2. Generate WindGym dataset (100k samples)
3. Retrain surrogate
4. Implement MPC-WindGym control loop
5. Run closed-loop experiments
6. Performance evaluation

### Phase 3: RL Enhancement - 2-4 weeks (optional)
1. Add RL layer for model mismatch
2. Train RL policy
3. Combine MPC + RL
4. Evaluate hybrid controller

---

## Open Questions

### Dataset Generation
- **Q:** How long to run each WindGym episode?
- **Q:** Should we include transient dynamics or just steady-state?
- **Q:** Sample from random initial conditions or reset each time?

### Surrogate Model - Handling Dynamics ⭐
- **Q:** Should surrogate predict power or power + dynamics?
- **Q:** Include state history (LSTM/Transformer) or just current state?
- **Q:** Multi-step prediction or single-step?

**ANSWERED - Recommendation for handling temporal dynamics:**

**Option 1: Stacked Observations (RECOMMENDED START)**
- Concatenate recent history: `[yaw(t), yaw(t-1), ..., yaw(t-k), wind(t), wind(t-1), ...]`
- Use feedforward NN (current architecture)
- Window size: 5-10 timesteps (50-100 seconds)
- **Pros:** Simple, compatible with l4casadi, captures short-medium term dynamics
- **Cons:** Fixed window, input dimension grows

**Option 2: Explicit Delay States (ADD IF NEEDED)**
- Include `yaw(t-33)` explicitly (330s wake delay)
- Physically motivated
- Combine with stacked observations
- **Pros:** Captures long-term wake effects specifically
- **Cons:** Requires tracking long history buffer

**Option 3: Recurrent Model - LSTM/GRU (IF NEEDED)**
- Use sequence model to process temporal data
- Learns what history matters
- **Pros:** Flexible, captures long-term dependencies
- **Cons:** More complex, l4casadi compatibility uncertain, harder to train

**Implementation approach:**
1. Start with stacked observations (10 steps = 100s history)
2. If insufficient, add explicit delay term (t-33)
3. If still needed, explore LSTM (check l4casadi support)

**MPC State with History:**
```python
# State includes history
x = [yaw(t), yaw(t-1), yaw(t-2), ..., yaw(t-k)]

# Dynamics: shift history
x_next = [yaw(t) + u(t)*dt, yaw(t), yaw(t-1), ..., yaw(t-k+1)]

# Surrogate input: stacked state + wind
surrogate_input = [x, wind(t), wind(t-1), ...]
```

**Input dimension with history:**
- Without history: 6 inputs (4 yaw + 2 wind)
- With 10-step history: 60 inputs (4×10 yaw + 2×10 wind)
- With history + delay: 64 inputs (60 + 4 delayed yaw)

See detailed analysis in commit notes.

### MPC-WindGym Interface
- **Q:** What MPC update frequency? (1s, 5s, 10s?)
- **Q:** How to handle WindGym's internal delays?
- **Q:** Should MPC predict future wind or use current?

---

## Next Steps

### Immediate (This Week)
1. ✅ Complete Phase 1 testing with PyWake
2. Validate surrogate MPC finds optimal yaw
3. Generate full 100k PyWake dataset
4. Achieve target accuracy (MAE < 1%)

### Short Term (Next 2 Weeks)
1. Design WindGym dataset generation
2. Answer open questions above
3. Implement `generate_dataset_windgym.py`
4. Test with small WindGym dataset

### Medium Term (1 Month)
1. Full WindGym dataset generation
2. Retrain surrogate
3. Implement MPC-WindGym control
4. Closed-loop experiments
5. Performance comparison vs baselines

---

## Files to Create (Future)

```
alternative_approach/
├── scripts/
│   ├── generate_dataset_windgym.py    # New: WindGym data
│   └── generate_dataset_v2.py         # Current: PyWake data
│
├── nmpc_windgym_control.py            # New: MPC-WindGym loop
├── nmpc_surrogate.py                  # Current: MPC only
│
├── examples/
│   ├── demo_windgym_control.py        # New: Closed-loop demo
│   └── demo_surrogate_mpc.py          # Current: MPC only
│
└── docs/
    └── WINDGYM_INTEGRATION.md         # New: Integration guide
```

---

## Resources

- **WindGym:** [Link to WindGym repo/docs]
- **MPC+RL literature:** Combine model-based and model-free
- **Surrogate models:** Neural networks for dynamics

---

**Status:** 📝 Notes captured for future work
**Priority:** After Phase 1 validation completes
**Owner:** TBD
