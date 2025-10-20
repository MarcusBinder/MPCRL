# Wind Farm MPC Debugging Plan

## Observed Problems

From log file `mpc_demo_log_20251020_124416.txt`:

### Demo 1: Perfect Oscillation
```
t=00: ψ=[ 6.7 -4.8 -6.1 -6.9], P=13.961 MW
t=01: ψ=[-0.8  2.7  1.4  0.6], P=14.027 MW
t=02: ψ=[ 6.7 -4.8 -6.1 -6.9], P=13.961 MW  ← Same as t=00!
t=03: ψ=[-0.8  2.7  1.4  0.6], P=14.027 MW  ← Same as t=01!
```

**This is a LIMIT CYCLE** - oscillating between exactly two configurations.

### Key Issues

1. **Oscillation**: System alternates between two states instead of converging
2. **Power Loss**: Final power < initial power (getting worse, not better!)
3. **All turbines move**: Large simultaneous changes in all yaws
4. **Non-convergent**: No steady-state, just keeps oscillating

---

## Root Cause Hypotheses

### Hypothesis 1: Gradient Sign Error ❌
**Symptoms**: Power would consistently decrease
**What we see**: Power oscillates (sometimes better, sometimes worse)
**Verdict**: Unlikely to be the only issue

### Hypothesis 2: Delay Model Mismatch ⚠️
**Problem**: We compute gradient without accounting for delays in the optimization
- Current: Compute ∇P(ψ_current), optimize as if delays don't exist
- Reality: Actual power depends on ψ(t-τ), not ψ(t)

**Why this causes oscillation**:
- MPC says "go to state A" based on gradient without delays
- We go there, power is different than predicted (due to delays)
- New gradient says "go back to state B"
- Repeat forever

### Hypothesis 3: Overstepping in Successive Linearization ⚠️
**Problem**: Taking full optimal steps of linearized problem
- Linearization: P(ψ) ≈ P₀ + ∇P·(ψ - ψ₀)
- Only valid near ψ₀
- We're taking steps of 7.5° (rate limit) which might be too large

**Why this causes oscillation**:
- Linearization says "optimal is 10° away"
- We jump there (violates linearization validity)
- New linearization says "optimal is back where we started"
- Limit cycle forms

### Hypothesis 4: Cost Function Issue ⚠️
**Problem**: The cost we're minimizing doesn't match our goal

Current cost: `-∇P·ψ + λ/2·||u||²`

Issues:
- This is a LINEAR approximation of power
- No accounting for delays
- No receding horizon benefit

---

## Debugging Strategy

### Phase 1: Verify Basic Components ✓

**Test 1**: Does optimizer work at all?
- [ ] Create simple test without wind farm
- [ ] Manually set gradient = [1, 1, 1, 1]
- [ ] Check if optimizer moves in that direction
- **File**: `test_optimizer_basic.py`

**Test 2**: Is gradient correct?
- [ ] Compute gradient at ψ=[0,0,0,0]
- [ ] Manually verify by computing P(ψ + ε·e_i) - P(ψ - ε·e_i)
- [ ] Check sign and magnitude
- **File**: `test_gradient_correctness.py`

**Test 3**: Single step without MPC
- [ ] Start at ψ=[0,0,0,0]
- [ ] Compute gradient
- [ ] Take small step: ψ_new = ψ + α·∇P (α small, like 0.1)
- [ ] Verify power increases
- **File**: `test_gradient_descent.py`

### Phase 2: Understand The Delay Problem

**The Core Issue**:

In a system with delays, if turbine 0 changes yaw at time t, turbine 3 doesn't feel the effect until time t+τ.

Current approach treats it as:
```
P(t) = f(ψ₀(t), ψ₁(t), ψ₂(t), ψ₃(t))  ← WRONG!
```

Reality:
```
P(t) = f(ψ₀(t-τ₀), ψ₁(t-τ₁), ψ₂(t-τ₂), ψ₃(t-τ₃))  ← RIGHT!
```

**What this means for MPC**:

If we change ψ₀(t) today, the power benefit shows up at t+τ₀, not at t!

So the cost at time step k in the horizon should be:
```
J_k = -P(ψ₀(k-τ₀), ψ₁(k-τ₁), ...) + λ||u_k||²
```

But ψ(k-τ) might be:
- In the past (k < τ): Use historical data (fixed)
- In the future (k ≥ τ): Decision variable in the optimization

**Test 4**: Verify delay understanding
- [ ] Create a 2-turbine case with known delay
- [ ] Change upstream yaw, measure when downstream power changes
- [ ] Confirm delay matches theory
- **File**: `test_delay_propagation.py`

### Phase 3: Fix The Formulation

**Option A: Ignore Delays (Simplest)**
- Just optimize current power without future prediction
- Won't be optimal but should at least not oscillate
- Good for debugging

**Option B: Proper Delay-Aware MPC**
- For each horizon step k, compute which past yaws affect power
- Build cost correctly accounting for this
- More complex but correct

**Option C: Receding Horizon with Look-Ahead**
- Use a simpler model that captures delay effects
- Don't try to be perfect, just directionally correct

**Recommendation**: Start with Option A to verify basics, then try Option B.

### Phase 4: Fix Successive Linearization

Current approach:
```python
1. Compute ∇P at ψ_current
2. Solve: min -∇P·ψ + λ||u||²
3. Apply full optimal step
```

Better approach:
```python
1. Compute ∇P at ψ_current
2. Solve: min -∇P·ψ + λ||u||²
3. Apply αψ_optimal + (1-α)·ψ_current  # Partial step (α=0.1-0.5)
```

Or use trust region:
```python
1. Compute ∇P at ψ_current
2. Solve: min -∇P·ψ + λ||u||²
         s.t. ||ψ - ψ_current|| ≤ Δ  # Trust region
3. Apply full optimal (but constrained to stay close)
```

**Test 5**: Add step size control
- [ ] Implement line search or trust region
- [ ] Verify convergence to local optimum
- **File**: `test_step_size_control.py`

---

## Implementation Plan

### Step 1: Create Diagnostic Tests (Today)
1. `test_optimizer_basic.py` - Does acados work?
2. `test_gradient_correctness.py` - Is ∇P right?
3. `test_gradient_descent.py` - Does steepest descent work?
4. `test_delay_propagation.py` - Do delays work as expected?

### Step 2: Fix The Obvious Bugs (Today)
Based on test results, fix:
- Sign errors in gradient
- Wrong indexing in state extraction
- Incorrect delay calculation

### Step 3: Simplify To Minimum Working Example (Tomorrow)
Strip down to simplest possible case:
- 2 turbines only
- No delays initially
- Simple gradient descent (not even MPC)
- Get THIS working first

### Step 4: Add Complexity Incrementally (Tomorrow)
Once basic case works:
1. Add MPC (still no delays)
2. Add delays but with proper formulation
3. Add more turbines
4. Add realistic constraints

### Step 5: Tune and Validate (Next Day)
- Tune move penalties
- Tune horizon length
- Validate against known results

---

## Decision Points

### Should we use MPC at all?

**Pros of MPC**:
- Handles constraints naturally
- Predicts future
- Principled framework

**Cons of MPC**:
- Complex with delays
- Requires accurate model
- Can be hard to debug

**Alternative**: Simple gradient ascent
```python
while not converged:
    grad = compute_gradient(psi_current)
    psi_next = psi_current + alpha * grad
    psi_next = clip(psi_next, bounds)
    psi_current = psi_next
```

This would at least monotonically increase power (if gradient is right).

**Recommendation**: Get gradient ascent working FIRST, then add MPC complexity.

### How to handle delays?

**Option 1**: Ignore them in optimization, accept sub-optimality
- Pro: Simple, debuggable
- Con: Not optimal

**Option 2**: Model them properly in cost
- Pro: Correct, optimal
- Con: Complex, hard to implement

**Option 3**: Use a "virtual delay" heuristic
- Delay the gradient update by τ steps
- Pro: Simple approximation
- Con: Ad-hoc, no guarantees

**Recommendation**: Option 1 for now (get it working), then Option 2 (get it right).

---

## Success Criteria

### Minimum (must have):
1. ✓ No oscillation - yaws converge to steady state
2. ✓ Power increases or stays same - never decreases
3. ✓ Gradients point in sensible direction
4. ✓ Optimizer finds feasible solution

### Target (should have):
1. ✓ Power increases by 1-5% from aligned case
2. ✓ Convergence in < 50 steps
3. ✓ Smooth yaw trajectories (no chattering)
4. ✓ Solve times < 100ms per step

### Stretch (nice to have):
1. ✓ Handles changing wind conditions
2. ✓ Robust to initial conditions
3. ✓ Delay-aware optimization
4. ✓ Provably optimal (at least locally)

---

## Next Steps

**IMMEDIATE** (next 30 min):
1. Create `test_gradient_correctness.py`
2. Verify gradient at [0,0,0,0] makes sense
3. Document findings

**TODAY**:
1. Create all 4 diagnostic tests
2. Run them and document results
3. Make a go/no-go decision on current approach

**TOMORROW**:
1. If tests pass → proceed with fixing delays
2. If tests fail → simplify to gradient descent
3. Build from working foundation

---

## Questions to Answer

1. **Is the gradient correct?** (Test 2)
2. **Does the optimizer work?** (Test 1)
3. **Are delays the problem?** (Test 4)
4. **Is linearization the problem?** (Test 3, 5)
5. **Should we use MPC at all?** (Decide after tests)

Let's answer these systematically rather than guessing!
