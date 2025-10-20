# Wind Farm Yaw Control MPC - Project Status

**Date**: 2025-10-20
**Status**: Debugging Phase - Critical Issue Discovered

---

## Project Overview

### Goal
Implement Model Predictive Control (MPC) using acados to optimize wind turbine yaw angles for maximizing total wind farm power production through wake steering.

### Approach
- **Method**: Successive Linearization MPC
  - Linearize power function P(ψ) around current operating point
  - Solve quadratic program (QP) at each timestep
  - Use acados for fast embedded optimization
- **Wake Model**: PyWake with Blondel_Cathelain_2020 + Jimenez deflection
- **Layout**: 4 turbines in straight line, 5-7D spacing
- **Control**: Yaw angles (ψ) with rate limits

### Key Components
- `nmpc_windfarm_acados_fixed.py`: Main MPC implementation
- `demo_yaw_optimization.py`: Demo with 3 test scenarios
- acados parameters for fast gradient updates (no rebuilding)

---

## Problem Statement

User observed **perfect limit cycle oscillations** in MPC demos:
```
t=00: ψ=[ 6.7 -4.8 -6.1 -6.9], P=13.961 MW
t=01: ψ=[-0.8  2.7  1.4  0.6], P=14.027 MW
t=02: ψ=[ 6.7 -4.8 -6.1 -6.9], P=13.961 MW  ← Exact repeat!
t=03: ψ=[-0.8  2.7  1.4  0.6], P=14.027 MW  ← Exact repeat!
```

System oscillates between two states instead of converging, with power often decreasing.

---

## Debugging Journey

### Phase 1: Initial Hypotheses (REJECTED)

**Hypothesis 1**: Gradient computation error
**Result**: ✅ Gradient is CORRECT (`test_gradient_correctness.py`)
- Verified with manual finite differences
- Matches at all test points
- Sign and magnitude are correct

**Hypothesis 2**: Move penalty too low
**Result**: ❌ Not the issue
- Tested λ = 0.01, 5.0, 50.0, 500.0
- ALL values still produced oscillations
- Gradient term dominates even with huge penalties

**Hypothesis 3**: Rate limit too large (linearization invalid)
**Result**: ❌ Making it worse
- Reduced rate limit to 0.05 deg/s
- Caused QP solver failures (status 4)
- Problem becomes infeasible

### Phase 2: Searching for the Optimum

**Test**: `test_search_for_optimum.py`
**Discovery**: ψ=[0,0,0,0] is the GLOBAL MAXIMUM for 7D spacing

| Configuration | Power (MW) | vs Aligned |
|---------------|-----------|------------|
| ψ=[0,0,0,0] | 14.027 | 0 kW (BEST) |
| Demo State B | 14.013 | -14 kW |
| Demo State A | 13.813 | -214 kW |
| [20,20,20,0] | 12.361 | -1666 kW |

**Initial Conclusion** (WRONG): Straight-line layouts don't benefit from yaw control.

**User Pushback**: "I dont believe that solution. Did you test it with the normal pywake solution?"
- User was RIGHT to question this!
- Known that wake steering should work at 5D spacing
- Expected optimum around [20, 20, 20, 0]

### Phase 3: Testing PyWake Directly (BREAKTHROUGH!)

**Test**: `test_wake_basic.py`
**Critical Discovery**: PyWake is NOT computing wakes AT ALL!

```
Two turbines at 5D spacing:
  T0: 3.507 MW  (upstream)
  T1: 3.507 MW  (downstream) ← Should be ~2.0-2.5 MW!
  Wake loss: 0.0%  ← Should be 30-50%!
```

**Verification**: Wind direction test shows:
- Wind from 0°: T0=3.507 MW, T1=3.507 MW (both same!)
- Wind from 90°: T0=0.638 MW, T1=3.507 MW

This confirms T1 is NOT experiencing wake losses when it should be.

**Test**: `test_different_wake_models.py`
**Result**: NO wake model computes wakes correctly
- Tried: NOJ (Jensen), BastankhahGaussian, IEA37SimpleBastankhahGaussian
- ALL show 0% wake loss
- All turbines produce identical power regardless of position

---

## Current Understanding

### Root Cause Chain

```
PyWake NOT computing wakes
         ↓
No wake interactions between turbines
         ↓
No benefit from wake steering (only cosine losses)
         ↓
ψ=[0,0,0,0] appears "optimal" (incorrectly)
         ↓
MPC tries to optimize flat/noisy objective
         ↓
Oscillations and instability
```

### Why This Explains Everything

1. **No wake losses**: T1 has same power as T0 regardless of yaw
2. **Only cosine losses**: Yawing reduces power via cos³(ψ) effect
3. **No steering benefit**: Downstream turbines unchanged by upstream yaw
4. **Flat objective**: All configurations near ψ=0 have similar power
5. **MPC struggles**: Small gradients + numerical noise → large control moves
6. **Oscillations**: Linearization errors dominate in flat region

### What Should Happen (vs Reality)

**Expected behavior** (5D spacing):
- T1 in wake: 2.0-2.5 MW (30-40% loss)
- T0 yawed 25°: T1 gains ~0.3-0.5 MW (wake deflection)
- Net benefit: Possible with right yaw angles

**Actual behavior**:
- T1 in wake: 3.507 MW (0% loss) ❌
- T0 yawed 25°: T1 still 3.507 MW (0 gain) ❌
- Net benefit: Impossible (only losses)

---

## Possible Causes of PyWake Issue

### 1. Wind Direction Convention
- Maybe `wd=0` doesn't mean "from west"?
- PyWake might use different convention (to/from, met/math)
- Need to check PyWake documentation

### 2. Turbine Placement
- Maybe x-axis doesn't align with wind direction?
- Coordinate system confusion?
- All turbines at y=0 - could this be causing issues?

### 3. Wake Model Configuration
- Missing required parameters?
- Models need explicit superposition settings?
- Turbulence intensity not set correctly?

### 4. Power Curve Issue
- DTU10MW turbine data not loading correctly?
- Operating at constant power (rated)?
- Wind speed 8 m/s might be special point?

### 5. Code Bug in PyWake Call
- Yaw array shape wrong (but seems correct: [N, 1, 1])?
- Missing required arguments?
- Some parameter disabling wake calculations?

---

## Immediate Next Steps

### 1. Verify Wind Direction Convention
```python
# Test all 4 cardinal directions
# See which one puts T1 in T0's wake
for wd in [0, 90, 180, 270]:
    sim = model(x=x, y=y, wd=wd, ws=8.0)
    # Check which gives T1 < T0
```

### 2. Check PyWake Documentation
- Look up wind direction convention
- Find working example with wake losses
- Verify correct way to call models

### 3. Try Simple PyWake Example
- Copy exact code from PyWake tutorials
- Verify wakes work in isolation
- Then adapt to our use case

### 4. Test Different Configurations
```python
# Maybe turbines need to be at different y?
x = [0, 5*D, 10*D]
y = [0, 0.1, 0]  # Tiny offset?

# Or different wind speed?
ws = 10.0  # Instead of 8.0

# Or explicit turbulence?
site = UniformSite(ti=0.06)
```

### 5. Check PyWake Version
- Verify compatible version installed
- Look for known issues in GitHub
- Try updating/downgrading if needed

---

## Key Files Created

### Diagnostic Tests
1. `test_gradient_correctness.py` - Gradient verification ✅
2. `test_gradient_nonzero.py` - Gradient at various points ✅
3. `test_simple_gradient_ascent.py` - Non-MPC optimization
4. `test_adaptive_gradient_ascent.py` - With line search ✅
5. `test_search_for_optimum.py` - Global optimum search ✅
6. `test_wake_deflection.py` - Wake steering analysis
7. `test_pywake_standard.py` - Direct PyWake testing
8. `test_wake_basic.py` - Fundamental wake test ✅ **Found the issue!**
9. `test_different_wake_models.py` - Model comparison ✅

### Fix Attempts (Pre-Discovery)
10. `demo_fixed_high_penalty.py` - λ=5.0 (didn't work)
11. `demo_very_high_penalty.py` - λ up to 500 (didn't work)
12. `demo_small_steps.py` - Small rate limits (solver failed)

### Documentation
13. `DEBUG_PLAN.md` - Original systematic debugging plan
14. `DEBUGGING_RESULTS.md` - Detailed test findings
15. `FINAL_CONCLUSIONS.md` - Conclusions (now outdated!)
16. `PROJECT_STATUS.md` - This file

---

## What We Know For Sure

### ✅ Confirmed Working
- acados solver builds and runs
- Gradient computation is mathematically correct
- QP solver finds solutions (when problem is well-posed)
- Parameter updates work (fast, no rebuilding)
- MPC formulation is reasonable

### ❌ Confirmed Broken
- PyWake wake calculations (0% loss when should be 30-50%)
- Wake steering benefit (0 MW gain when should be positive)
- Overall power improvement (losses only, no gains)

### ❓ Unknown / To Investigate
- Why PyWake doesn't compute wakes
- Correct wind direction convention
- Proper way to set up wake models
- Whether this is a PyWake version issue
- If DTU10MW turbine data is loading correctly

---

## Revised Theory

### Original Theory (WRONG)
"Straight-line layouts don't benefit from yaw control due to geometry."

### Current Theory (LIKELY CORRECT)
"PyWake is misconfigured or has a bug, preventing wake calculations. This makes yaw optimization appear useless when it shouldn't be."

### Evidence Supporting Current Theory
1. ✅ User expectation of [20,20,20,0] being optimal
2. ✅ Literature shows wake steering works at 5D
3. ✅ Explicit test shows 0% wake loss (impossible)
4. ✅ Wind direction test shows T0 affected but not T1
5. ✅ ALL wake models show same issue (suggests common root cause)

---

## Questions to Answer

### High Priority
1. **What is PyWake's wind direction convention?**
   - Is `wd=0` from north, east, or custom?
   - Does it use "from" or "to" convention?

2. **How to correctly set up wake models?**
   - Required parameters beyond basic x, y, wd, ws?
   - Superposition settings needed?

3. **Why do ALL models show 0% wake loss?**
   - Common bug in our code?
   - PyWake version incompatibility?
   - Fundamental misunderstanding?

### Medium Priority
4. Can we find ANY PyWake configuration that shows wake losses?
5. Is DTU10MW turbine loaded with correct power curve?
6. Should we try a different wake model library entirely?

### Low Priority
7. Once wakes work, will MPC converge properly?
8. What will optimal yaw angles actually be?
9. How much power gain is realistic for this layout?

---

## Success Criteria (Revised)

### Milestone 1: Get PyWake Working ← **WE ARE HERE**
- [ ] T1 shows 30-50% power loss when in T0's wake
- [ ] Wake loss depends on wind direction
- [ ] Yawing T0 changes T1 power (wake steering)

### Milestone 2: Verify Optimum
- [ ] Find global optimum yaw configuration
- [ ] Confirm it's NOT ψ=[0,0,0,0]
- [ ] Measure realistic power gain (1-5% expected)

### Milestone 3: Fix MPC
- [ ] MPC converges to correct optimum
- [ ] No oscillations
- [ ] Stable and repeatable

### Milestone 4: Validation
- [ ] Works across different wind conditions
- [ ] Reasonable solve times (<100ms)
- [ ] Realistic power gains achieved

---

## Notes and Observations

### User Feedback
- User correctly suspected wake steering should work
- User was right to question "straight-line" explanation
- User's intuition about [20,20,20,0] is likely correct
- Good instinct to verify with "normal PyWake"

### Debugging Approach
- Systematic testing worked well
- Found issue by going back to first principles
- User skepticism was crucial to finding real problem
- Should have tested PyWake basics earlier

### What Went Wrong
- Assumed PyWake was working without verification
- Built elaborate theory on false premise
- Didn't question why ALL yaw angles decreased power
- Should have been red flag when [20,20,20,0] was terrible

### Lessons Learned
1. Test assumptions at lowest level first
2. If results violate domain knowledge, dig deeper
3. Listen when user says "I don't believe that"
4. Physics sanity checks are crucial
5. Don't build castles on unverified foundations

---

## State of the Code

### Working Components
- `nmpc_windfarm_acados_fixed.py`: MPC implementation (solid)
- acados solver setup and parameter updates (good)
- Gradient computation (verified correct)
- Demo infrastructure and logging (works)

### Broken Components
- `build_pywake_model()`: Returns model that doesn't compute wakes
- `pywake_farm_power()`: Gets wrong results from PyWake
- Any optimization based on current PyWake output (garbage in, garbage out)

### Next Code Changes Needed
1. Fix PyWake setup in `build_pywake_model()`
2. Verify correct wind direction handling
3. Re-run all tests once wakes work
4. May need to adjust MPC parameters once real gradients available

---

## Environment Info

- Python: 3.11
- PyWake: Version TBD (need to check)
- acados: Working
- Location: `/home/marcus/Documents/mpcrl/`
- Turbine: DTU10MW (10 MW reference turbine)
- Rotor diameter: D = 178 m

---

## Communication with User

User has been:
- Patient through extensive debugging
- Correctly skeptical of wrong conclusions
- Helpful in identifying real issue
- Asking good clarifying questions

Next update should:
- Acknowledge user was right
- Show wake calculation working
- Demonstrate realistic power gains
- Then revisit MPC convergence

---

## Mental Model Update

### Before
```
MPC → Oscillations → Must be MPC bug
                   → Or layout doesn't benefit from yaw
                   → Or move penalty wrong
```

### After
```
PyWake → No wakes → No gradients → Flat objective
                                  → MPC has nothing to optimize
                                  → Oscillations are symptom, not cause
```

**The oscillations were a RED HERRING.** The real problem is upstream (PyWake), not downstream (MPC).

---

## Action Items

**IMMEDIATE** (next 30 minutes):
- [ ] Check PyWake documentation for wind direction convention
- [ ] Find working PyWake example with verified wake losses
- [ ] Test different wd values (0, 90, 180, 270)

**TODAY**:
- [ ] Get PyWake computing wakes correctly
- [ ] Verify wake steering provides benefit
- [ ] Find actual optimal yaw configuration

**TOMORROW**:
- [ ] Re-run MPC with correct wake model
- [ ] Tune MPC now that gradients are meaningful
- [ ] Validate convergence and power gains

---

## Confidence Levels

| Statement | Confidence |
|-----------|------------|
| PyWake not computing wakes | 99% ✅ |
| This is causing MPC issues | 95% ✅ |
| Issue is in our setup (not PyWake bug) | 70% |
| Wake steering should work at 5D | 90% |
| [20,20,20,0] is near optimal | 60% |
| MPC will converge once PyWake fixed | 75% |
| Can achieve 1-5% power gain | 60% |

---

## Open Questions for User

1. Do you have any working PyWake examples we can reference?
2. What version of PyWake are you using?
3. Have you verified wake calculations work elsewhere in your code?
4. Any known issues with PyWake + DTU10MW turbine?
5. Expected power gain percentage for this layout?

---

**Status**: Blocked on PyWake wake calculation issue
**Priority**: Fix PyWake setup (P0 - Critical)
**Next Step**: Investigate wind direction convention and model configuration
**ETA**: TBD - depends on finding PyWake issue

Last Updated: 2025-10-20
