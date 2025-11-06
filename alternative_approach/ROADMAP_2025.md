# Alternative Approach - 2025 Roadmap

**Created:** 2025-11-06
**Owner:** MPC/RL Team
**Status:** Planning Phase

---

## Vision

Build a **production-ready wind farm yaw controller** that combines:
- Fast MPC for constraint handling and tracking (<1ms)
- Global optimization for finding optimal setpoints
- Ability to achieve ~15% power gain through wake steering

---

## Strategic Goals

1. **Validate the hybrid architecture** achieves near-optimal performance
2. **Build production-ready implementation** with robustness features
3. **Explore surrogate models** for future pure-MPC solution
4. **Compare with main approach** (SAC+MPC) for insights

---

## Phase 1: Validation & Quick Wins (Week 1-2)

### Goal
Confirm that the hybrid architecture works and achieves the expected ~15% gain

### Tasks

#### Task 1.1: Test Existing Hybrid Implementation
**File:** `examples/hybrid_mpc_example.py`

**Steps:**
```bash
cd alternative_approach/
python examples/hybrid_mpc_example.py
```

**Validation:**
- [ ] Code runs without errors
- [ ] MPC converges (<1ms per solve)
- [ ] Power gain is measured
- [ ] Results match expectations

**Expected Outcome:** Understand baseline performance

#### Task 1.2: Run Ground Truth Comparison
**File:** `tests/test_optimal_yaw.py`

**Steps:**
```bash
python tests/test_optimal_yaw.py
```

**Validation:**
- [ ] Grid search finds optimal yaw
- [ ] Optimal power gain ~15%
- [ ] Results are reproducible

**Expected Outcome:** Establish performance target

#### Task 1.3: Multi-Condition Testing

**Steps:**
1. Test hybrid approach on multiple wind conditions:
   - Wind speed: 6, 8, 10, 12 m/s
   - Wind direction: 270° (main), 0°, 90°, 180°
   - Spacing: 5D, 7D, 10D

2. For each condition:
   - Run grid search for optimal
   - Run hybrid MPC
   - Compare performance

**Validation:**
- [ ] Hybrid achieves >90% of optimal gain
- [ ] Performance is consistent across conditions
- [ ] No solver failures

**Expected Outcome:** Confidence in hybrid approach

#### Task 1.4: Document Findings

**Create:** `alternative_approach/results/validation_report.md`

**Contents:**
- Performance comparison table
- Plot: Hybrid vs Optimal across conditions
- Failure modes (if any)
- Recommendations

**Expected Outcome:** Clear documentation of validation results

### Deliverables
- ✅ Validation report
- ✅ Performance benchmarks
- ✅ Decision: Go/No-Go on hybrid approach

### Timeline: 1-2 weeks

---

## Phase 2: Production Implementation (Week 3-6)

### Goal
Build a robust, production-ready hybrid controller

### Tasks

#### Task 2.1: Lookup Table Implementation

**Create:** `alternative_approach/lookup_table.py`

**Features:**
1. Offline grid search for multiple conditions
2. Efficient storage (HDF5 or pickle)
3. Fast interpolation for online queries
4. Handling of edge cases

**API Design:**
```python
class OptimalYawLookup:
    def __init__(self, lookup_file):
        self.table = self.load_table(lookup_file)

    def get_optimal_yaw(self, wind_speed, wind_direction, spacing):
        """Return optimal yaw angles for given conditions"""
        return self.interpolate(wind_speed, wind_direction, spacing)

    @staticmethod
    def build_table(wind_speeds, wind_directions, spacings, output_file):
        """Build lookup table via grid search"""
        # Run PyWake grid search for each condition
        # Store results
```

**Validation:**
- [ ] Table covers expected operating range
- [ ] Interpolation is smooth
- [ ] Lookup time <1ms

#### Task 2.2: Two-Layer Controller

**Create:** `alternative_approach/hybrid_controller.py`

**Architecture:**
```python
class HybridWindFarmController:
    def __init__(self, lookup_table, mpc_controller):
        self.lookup = lookup_table
        self.mpc = mpc_controller
        self.last_update_time = 0
        self.update_interval = 60.0  # seconds

    def step(self, current_state, wind_conditions):
        # Strategic layer: Update target if wind changed
        if self.should_update_target(wind_conditions):
            target = self.lookup.get_optimal_yaw(
                wind_conditions.U,
                wind_conditions.theta,
                self.spacing
            )
            self.mpc.set_target(target)

        # Tactical layer: Track target with constraints
        control = self.mpc.step(current_state)
        return control

    def should_update_target(self, wind_conditions):
        # Check if wind changed significantly
        # Check if enough time passed
        return True/False
```

**Validation:**
- [ ] Smooth transitions between targets
- [ ] No chattering or instability
- [ ] Performance matches standalone tests

#### Task 2.3: Robustness Features

**Add:**
1. **Wind change detection**
   - Track wind speed/direction history
   - Trigger re-optimization when wind changes >5%

2. **Fallback mode**
   - If MPC fails, use simple tracking
   - If lookup fails, use safe default (0° yaw)

3. **Performance monitoring**
   - Log power output
   - Track MPC solve time
   - Alert on degradation

4. **Parameter adaptation**
   - Adjust MPC weights based on conditions
   - Tune update frequency based on wind variability

**Validation:**
- [ ] Handles edge cases gracefully
- [ ] No crashes or hangs
- [ ] Degrades gracefully under failures

#### Task 2.4: Integration Tests

**Create:** `tests/test_hybrid_controller_integration.py`

**Test Cases:**
1. Steady wind conditions
2. Slowly varying wind
3. Rapidly changing wind
4. Edge cases (very low/high wind speed)
5. Solver failures (inject faults)

**Validation:**
- [ ] All tests pass
- [ ] Performance meets targets
- [ ] No memory leaks
- [ ] Real-time capable

#### Task 2.5: Documentation

**Update:**
1. `README.md` - Add hybrid controller usage
2. Create `DEPLOYMENT_GUIDE.md`
3. Add API documentation
4. Create example scripts

**Contents:**
- Installation instructions
- Usage examples
- Configuration options
- Troubleshooting guide

### Deliverables
- ✅ Production-ready hybrid controller
- ✅ Comprehensive test suite
- ✅ Deployment documentation
- ✅ Performance benchmarks

### Timeline: 3-4 weeks

---

## Phase 3: Surrogate Model Development (Week 7-14)

### Goal
Build neural network surrogate for nonlinear MPC (alternative to hybrid approach)

### Tasks

#### Task 3.1: Dataset Generation

**Use:** `scripts/generate_surrogate_dataset.py`

**Steps:**
1. Define sampling strategy
   - Yaw ranges: [-30°, 30°] for upstream turbines
   - Wind speeds: [6, 8, 10, 12] m/s
   - Wind directions: [0°, 90°, 180°, 270°]
   - Sample count: 50,000-100,000 points

2. Generate dataset
   ```bash
   python scripts/generate_surrogate_dataset.py \
       --n_samples 100000 \
       --output data/surrogate_dataset.npz
   ```

3. Validate dataset
   - Check coverage
   - Visualize distribution
   - Verify no NaNs/outliers

**Validation:**
- [ ] Dataset is large enough (>50k samples)
- [ ] Good coverage of yaw space
- [ ] Power values are reasonable

#### Task 3.2: Model Training

**Use:** `scripts/train_surrogate.py`

**Steps:**
1. Define architecture
   - Input: yaw angles (4), wind speed (1), wind direction (1)
   - Hidden layers: [64, 64, 32]
   - Output: total farm power (1)
   - Activation: tanh (smooth for gradients)

2. Train model
   ```bash
   python scripts/train_surrogate.py \
       --dataset data/surrogate_dataset.npz \
       --output models/power_surrogate.pth \
       --epochs 1000
   ```

3. Evaluate model
   - Test set MAE < 1% of mean power
   - Gradient direction agreement > 95%
   - Smooth predictions (no discontinuities)

**Validation:**
- [ ] Training converges
- [ ] Test accuracy meets target
- [ ] Gradients match PyWake

#### Task 3.3: CasADi Integration

**Use:** `surrogate_module/casadi_graph.py`

**Steps:**
1. Convert trained model to CasADi graph
2. Export for acados
3. Validate against PyTorch model

**Validation:**
- [ ] CasADi graph produces same outputs
- [ ] Gradients match PyTorch
- [ ] Compatible with acados

#### Task 3.4: Nonlinear MPC Implementation

**Create:** `alternative_approach/nmpc_surrogate.py`

**Changes from current implementation:**
1. Replace linearized cost with nonlinear surrogate cost
2. Use surrogate in acados external cost function
3. Adjust solver settings for nonlinearity

**Code:**
```python
# In acados setup
import casadi as ca

# Load surrogate
surrogate_graph = load_casadi_surrogate('models/power_surrogate.casadi')

# Define nonlinear cost
psi = ca.SX.sym('psi', 4)
power = surrogate_graph(psi, wind_speed, wind_direction)

ocp.model.cost_expr_ext_cost = -power  # Maximize power
```

**Validation:**
- [ ] Solver converges
- [ ] Solve time <10ms
- [ ] Finds near-optimal solutions

#### Task 3.5: End-to-End Validation

**Test:**
1. Compare surrogate MPC vs pure gradient MPC
2. Compare surrogate MPC vs hybrid approach
3. Test on multiple conditions
4. Measure computational cost

**Expected Results:**
- Surrogate MPC >> pure gradient MPC
- Surrogate MPC ≈ hybrid approach (performance)
- Surrogate MPC faster than hybrid (no global search)

### Deliverables
- ✅ Trained surrogate model
- ✅ Nonlinear MPC implementation
- ✅ Performance comparison
- ✅ Documentation

### Timeline: 6-8 weeks

---

## Phase 4: Comparison & Integration (Week 15-18)

### Goal
Compare alternative approach with main approach (SAC+MPC), identify synergies

### Tasks

#### Task 4.1: Performance Comparison

**Compare:**
1. **Hybrid MPC** (alternative approach)
2. **Model-free MPC + SAC** (main approach)

**Metrics:**
- Power gain (%)
- Computational cost (ms per step)
- Data efficiency (samples needed)
- Robustness (failure modes)
- Adaptability (new conditions)

#### Task 4.2: Identify Synergies

**Explore:**
1. Use RL to learn strategic layer (instead of lookup table)
2. Use MPC for both tactical control (both approaches)
3. Share learned policies / lookup tables
4. Hybrid: RL for long-term, MPC for short-term

#### Task 4.3: Unified Framework

**Design:**
A framework that supports both approaches

```python
class UnifiedWindFarmController:
    def __init__(self, strategic_method, tactical_method):
        self.strategic = strategic_method  # "lookup" or "rl" or "global_opt"
        self.tactical = tactical_method    # "gradient_mpc" or "model_free_mpc"

    def step(self, state, wind):
        target = self.strategic.get_target(wind)
        control = self.tactical.track(state, target)
        return control
```

**Benefits:**
- Easy comparison
- Can mix and match
- Clear separation of concerns

### Deliverables
- ✅ Comparison report
- ✅ Unified framework
- ✅ Recommendations

### Timeline: 3-4 weeks

---

## Success Metrics

### Phase 1 (Validation)
- [ ] Hybrid approach achieves >90% of optimal gain
- [ ] No solver failures on test conditions
- [ ] Documentation complete

### Phase 2 (Production)
- [ ] Controller runs in real-time (<1ms MPC, <10ms total)
- [ ] Robustness: handles edge cases gracefully
- [ ] Deployment-ready with docs

### Phase 3 (Surrogate)
- [ ] Surrogate accuracy: MAE < 1% of mean power
- [ ] Nonlinear MPC performance ≈ hybrid
- [ ] Solve time <10ms

### Phase 4 (Integration)
- [ ] Comprehensive comparison complete
- [ ] Unified framework implemented
- [ ] Clear recommendations for future work

---

## Risk Mitigation

### Risk 1: Hybrid Approach Underperforms
**Likelihood:** Low
**Impact:** High
**Mitigation:**
- Phase 1 validates early
- Multiple fallback options (surrogate, MPPI)
- Can tune lookup table resolution

### Risk 2: Surrogate Model Insufficient Accuracy
**Likelihood:** Medium
**Impact:** Medium
**Mitigation:**
- Start with simple architecture, increase complexity if needed
- Use active learning to focus on important regions
- Can fall back to hybrid approach

### Risk 3: Real-Time Performance Issues
**Likelihood:** Low
**Impact:** Medium
**Mitigation:**
- Current acados solver already <1ms
- Profile and optimize hot paths
- Can reduce horizon length if needed

### Risk 4: Integration Complexity
**Likelihood:** Medium
**Impact:** Low
**Mitigation:**
- Well-defined interfaces
- Incremental integration
- Comprehensive testing

---

## Decision Points

### After Phase 1
**Decision:** Continue with hybrid approach?
- If YES → Proceed to Phase 2
- If NO → Re-evaluate, consider MPPI or other approaches

### After Phase 2
**Decision:** Is production implementation sufficient?
- If YES → Deploy, monitor, consider Phase 3 as future work
- If NO → Investigate issues, iterate

### After Phase 3
**Decision:** Is surrogate approach better than hybrid?
- If YES → Replace hybrid with surrogate in production
- If NO → Keep hybrid, use surrogate for research

### After Phase 4
**Decision:** Unified framework vs separate implementations?
- Choose based on maintenance burden, use cases, team preferences

---

## Resource Requirements

### Compute
- **Phase 1:** Laptop sufficient
- **Phase 2:** Laptop sufficient
- **Phase 3:** GPU recommended for training (optional)
- **Phase 4:** Laptop sufficient

### Data
- **Phase 1:** Use existing test cases
- **Phase 2:** Build lookup table (few hours of PyWake runs)
- **Phase 3:** Generate 50k-100k samples (1-2 days of PyWake)

### Personnel
- **Phase 1:** 1 person, 1-2 weeks
- **Phase 2:** 1-2 people, 3-4 weeks
- **Phase 3:** 1 person with ML expertise, 6-8 weeks
- **Phase 4:** 2 people, 3-4 weeks

---

## Communication Plan

### Weekly Updates
- Progress on current phase
- Blockers / issues
- Preliminary results

### Phase Reviews
- End of each phase: presentation of results
- Decision points reviewed with team
- Adjust roadmap if needed

### Documentation
- Keep docs updated as work progresses
- Use `docs/` folder for technical docs
- Use `ROADMAP_2025.md` (this file) for project management

---

## Future Work (Beyond 2025)

1. **Field Deployment**
   - Test on real wind farm
   - Adapt to hardware constraints
   - Safety certification

2. **Advanced Features**
   - Wake model learning from data
   - Multi-objective optimization (power + loads)
   - Coordination with pitch control

3. **Research Directions**
   - MPPI implementation
   - Distributed MPC (turbine-level)
   - Integration with SCADA systems

---

## Conclusion

This roadmap provides a clear path forward for the alternative approach. The work is well-positioned to succeed:

- ✅ Strong foundation: comprehensive analysis, working code
- ✅ Clear goal: production-ready hybrid controller
- ✅ Incremental approach: validate before investing
- ✅ Multiple options: hybrid, surrogate, future MPPI
- ✅ Risk mitigation: fallbacks at each stage

**Recommended Start:** Phase 1 - Validation (1-2 weeks)

After Phase 1, we'll have clear data to decide whether to proceed with production implementation (Phase 2) or pivot to other approaches.

---

**Status:** ✅ Ready to begin
**Next Action:** Start Phase 1, Task 1.1
**Owner:** TBD
**Target Start Date:** TBD
