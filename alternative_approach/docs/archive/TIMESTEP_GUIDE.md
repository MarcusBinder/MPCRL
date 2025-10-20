# Time Step Selection Guide for Wind Farm MPC

## TL;DR

For wind farm yaw control: **Use dt = 15-30 seconds**

This is much larger than typical MPC applications, but wind farms are slow!

## Understanding Time Scales

### Wind Farm Dynamics

**Yaw Actuators (slowest):**
- Max yaw rate: ~0.3 deg/s
- Time to yaw 20°: ~67 seconds (over 1 minute!)
- These are SLOW mechanical systems

**Wake Advection (slow):**
- Turbine spacing: 4-7 rotor diameters (D)
- D ≈ 120-180m for modern turbines
- Spacing ≈ 500-1200m between turbines
- At U = 8 m/s wind speed:
  - 4D (712m): ~89 seconds
  - 5D (890m): ~111 seconds
  - 7D (1246m): ~156 seconds

**Wind Speed/Direction Changes (slow):**
- Turbulence integral time scale: 10-100 seconds
- Meandering wakes: 30-300 seconds
- Wind direction shifts: minutes

### Comparison with Other MPC Applications

| Application | Typical dt | Why |
|-------------|-----------|-----|
| **Wind Farm Yaw** | **15-30s** | Slow actuators, slow wake dynamics |
| Chemical reactor | 1-10s | Reaction kinetics |
| Robot manipulator | 0.01-0.1s | Fast mechanical dynamics |
| Autonomous vehicle | 0.05-0.2s | High-speed motion, safety critical |
| Building HVAC | 60-300s | Very slow thermal dynamics |

Wind farms are closer to HVAC than robots!

## Choosing dt: The Rule

**Rule of thumb**: Your horizon should cover at least 1-2 turbine-to-turbine advection times.

```
dt × N_h ≥ (1 to 2) × (turbine spacing / wind speed)
```

### Example Calculations

**Scenario 1: 7D spacing, U=8 m/s**
- Advection time: 1246m / 8 m/s = 156s
- Target horizon coverage: 1.5 × 156s = 234s
- With N_h=20: dt ≥ 234s/20 = **12s minimum**
- Recommended: dt=15-20s

**Scenario 2: 5D spacing, U=8 m/s**
- Advection time: 890m / 8 m/s = 111s
- Target horizon coverage: 1.5 × 111s = 167s
- With N_h=20: dt ≥ 167s/20 = **8s minimum**
- Recommended: dt=10-15s

**Scenario 3: 4D spacing, U=8 m/s**
- Advection time: 712m / 8 m/s = 89s
- Target horizon coverage: 1.5 × 89s = 134s
- With N_h=20: dt ≥ 134s/20 = **7s minimum**
- Recommended: dt=10-15s

## What Happens if dt is Too Small?

### Problem 1: Horizon Can't See Wake Effects

```python
# BAD: dt too small
dt = 5.0
N_h = 12
horizon = 5 × 12 = 60s  # Only 60 seconds!

# Wake takes 156s to propagate
# Optimizer can't see benefit of yawing → zero gradient!
```

### Problem 2: Unnecessary Computational Cost

- More control updates than needed
- More gradient computations (expensive with PyWake!)
- No benefit since actuators are slow anyway

### Problem 3: Numerical Issues

- Discretization errors accumulate over many steps
- Harder to maintain feasibility with tight rate constraints

## What Happens if dt is Too Large?

### Problem 1: Coarse Control

```python
# TOO LARGE
dt = 60.0  # 1 minute

# At 0.3 deg/s max rate:
# Max change per step = 60s × 0.3 deg/s = 18°
# This is close to the ±25° bounds!
# Control becomes "jerky"
```

### Problem 2: Rate Constraint Violations

With large dt, the rate constraint becomes:
```
|ψ[k+1] - ψ[k]| ≤ yaw_rate_max × dt
```

If dt is too large, this bound becomes loose and you lose smoothness.

### Problem 3: Discretization Error

- Linear dynamics approximation: x[k+1] = x[k] + u[k] × dt
- Error scales with dt²
- For dt > 30s, this becomes noticeable

## Recommended Settings

### For 7D Spacing (standard wind farms)

```python
cfg = MPCConfig(
    dt=20.0,      # 20 second sampling
    N_h=25-30,    # 500-600s horizon
    lam_move=0.1  # Moderate move penalty
)
```

Covers: 2-3 turbine delays, allows smooth yaw trajectories

### For 5D Spacing (closer spacing)

```python
cfg = MPCConfig(
    dt=15.0,      # 15 second sampling
    N_h=20-25,    # 300-375s horizon
    lam_move=0.1
)
```

Covers: 2-3 turbine delays

### For 4D Spacing (very close, offshore)

```python
cfg = MPCConfig(
    dt=10.0,      # 10 second sampling
    N_h=20-25,    # 200-250s horizon
    lam_move=0.15  # Slightly higher to keep smooth
)
```

Covers: 2-3 turbine delays

## Tuning Process

1. **Calculate advection time**: `T_adv = spacing / wind_speed`

2. **Choose target horizon**: `T_horizon = (1.5 to 2) × T_adv`

3. **Pick N_h** based on:
   - acados limits: N_h ≤ 30 is safe for HPIPM
   - Memory/speed: smaller is faster
   - Typical: N_h = 20-30

4. **Calculate dt**: `dt = T_horizon / N_h`

5. **Check rate constraint**:
   ```python
   max_change_per_step = yaw_rate_max × dt
   # Should be << yaw_max (e.g., 3-6° is good)
   ```

6. **Test and adjust**:
   - If gradients are zero → increase dt or N_h
   - If control is jerky → decrease dt or increase lam_move
   - If too slow → decrease dt (but may lose horizon coverage)

## Example: Debugging Zero Gradients

```python
# Your current setup
D = 178.0
spacing = 7*D = 1246m
U = 8.0
dt = 10.0
N_h = 12

# Check horizon coverage
T_adv = 1246 / 8.0 = 156s
T_horizon = dt × N_h = 10 × 12 = 120s
Coverage = T_horizon / T_adv = 120 / 156 = 0.77

# Problem: Coverage < 1.0 means horizon doesn't even reach next turbine!
# Solution: Increase dt to 15s
T_horizon_new = 15 × 12 = 180s
Coverage_new = 180 / 156 = 1.15  # Good!

# Or increase N_h to 20
T_horizon_new = 10 × 20 = 200s
Coverage_new = 200 / 156 = 1.28  # Even better!

# Or both: dt=15, N_h=20
T_horizon_new = 15 × 20 = 300s
Coverage_new = 300 / 156 = 1.92  # Excellent! Covers 2 delays
```

## Summary Table

| Turbine Spacing | Advection Time | Recommended dt | Recommended N_h | Horizon Coverage |
|----------------|----------------|----------------|-----------------|------------------|
| 4D (~712m) | ~89s | 10-15s | 20-25 | 2.2-2.9 delays |
| 5D (~890m) | ~111s | 10-15s | 20-25 | 1.8-2.8 delays |
| 6D (~1068m) | ~134s | 15-20s | 20-25 | 2.2-3.0 delays |
| 7D (~1246m) | ~156s | 15-20s | 25-30 | 2.4-3.2 delays |
| 8D (~1424m) | ~178s | 20-25s | 25-30 | 2.8-3.8 delays |

All assuming U = 8 m/s, D = 178m (DTU 10MW)

## Key Takeaways

1. **Wind farms are SLOW** → Use large dt (15-30s)

2. **Horizon must cover wake propagation** → Check dt × N_h ≥ 1.5 × advection_time

3. **Larger dt ≠ worse performance** → Actually helps see wake effects!

4. **Balance factors**:
   - Large dt: Better horizon coverage, fewer updates
   - Small dt: Smoother control, less discretization error
   - Sweet spot: dt = 15-20s for most wind farms

5. **If gradients are zero**: Increase dt or N_h to extend horizon

6. **acados limits**: Keep N_h ≤ 30 to avoid segfaults with HPIPM
