# Critical Discovery: Random Seed Bias in Optimization Tests

## The Problem

When running `tests/test_optimization_quality.py`, we observed that configurations with **lower maxfun** appeared to outperform configurations with **higher maxfun**, which seemed counterintuitive.

### Original Results (with seed=42 for all configs):
```
"Best" configuration: dt_opt=30, T_opt=200, maxfun=10
  Power: 1,150,302 W (100.91% of reference!)

Reference: dt_opt=10, T_opt=500, maxfun=100
  Power: 1,139,930 W (baseline)
```

The "best" config appeared to **beat the reference** by ~10,000W!

## The Root Cause

**The original test used `seed=42` for ALL 100 configurations.**

This meant:
- Every configuration got the same random number sequence from dual annealing
- Some configurations got "lucky" with seed=42
- Other configurations got "unlucky" with seed=42
- Results did NOT represent expected performance

## The Investigation

We ran `tests/test_maxfun_investigation.py` with multiple random seeds:

### Reference Config (dt_opt=10, T_opt=500, maxfun=100)
Across 5 different seeds:
```
Seed 0: 1,138,341 W
Seed 1: 1,139,113 W
Seed 2: 1,122,799 W
Seed 3: 1,120,513 W
Seed 4: 1,139,971 W

Mean:   1,132,147 W
Std:    8,612 W (1.72% variance)
```

### "Best" Config (dt_opt=30, T_opt=200, maxfun=10)
Across 5 different seeds:
```
Seed 0: 1,134,367 W
Seed 1: 1,127,509 W
Seed 2: 1,121,485 W
Seed 3: 1,118,111 W
Seed 4: 1,075,776 W  ← Terrible!

Mean:   1,115,449 W
Std:    20,595 W (5.25% variance!)
```

## The Truth

**The reference is actually ~17,000W BETTER on average!**

With seed=42 specifically:
- Low maxfun got lucky: 1,150,302 W
- High maxfun got unlucky: 1,139,930 W

But across multiple seeds:
- High maxfun is both better on average AND more consistent
- Low maxfun has high variance (can be great or terrible depending on seed)

## Why Low maxfun Has Higher Variance

1. **Fewer iterations = more seed-dependent**: With only 10 function evaluations, the optimizer doesn't have time to converge from different starting points. Results heavily depend on where random initialization places you.

2. **More iterations = convergence**: With 100 function evaluations, the optimizer has time to explore and converge to similar solutions regardless of random initialization.

3. **Stochastic optimization**: Dual annealing uses random exploration in early phases. Different seeds explore different regions of the search space.

## The Fix

Updated `tests/test_optimization_quality.py` to:

1. **Accept `n_seeds` parameter**: Average results over multiple random seeds per configuration

2. **Use unique seeds per config**: Instead of `seed=42` for all configs, use `seed=base_seed + config_idx*1000 + seed_idx`

3. **Report uncertainty**: Track std deviation when using multiple seeds

### Usage:

```python
# Fast single-seed test (varies seeds across configs)
N_SEEDS = 1
BASE_SEED = 100

# Robust multi-seed average (RECOMMENDED)
N_SEEDS = 3  # Takes 3x longer
BASE_SEED = 100

# Random seeds (not reproducible)
N_SEEDS = 3
BASE_SEED = None
```

## Recommendations

### For Testing Optimization Parameters:
- Use **N_SEEDS ≥ 3** to get statistically robust estimates
- Don't trust single-seed results for stochastic optimization
- Report mean ± std when comparing configurations

### For Understanding the Results:
- **Higher maxfun is better** (on average) but takes longer
- Lower maxfun is faster but less reliable (high variance)
- The speed/quality tradeoff is real, but must be measured properly

### For RL Training:
You have two options:

**Option 1: Use higher maxfun for consistency**
```python
mpc_maxfun = 50  # More reliable, lower variance
mpc_T_opt = 300
mpc_dt_opt = 20
```
- Pro: Consistent performance episode-to-episode
- Con: Slower (but still 2-3x faster than maxfun=100)

**Option 2: Use lower maxfun and accept variance**
```python
mpc_maxfun = 10  # Faster but variable
mpc_T_opt = 200
mpc_dt_opt = 30
```
- Pro: 10-15x faster
- Con: Performance varies significantly between episodes
- Con: RL agent may struggle to learn from inconsistent MPC behavior

**Recommendation**: Start with Option 1 (maxfun=20-30) for stable learning, then try Option 2 if training is too slow.

## Key Lessons

1. **Always test stochastic algorithms with multiple seeds**
2. **Don't use the same seed for all test configurations**
3. **Report uncertainty (std dev) alongside mean**
4. **Single-seed "winners" may just be lucky**
5. **Variance matters** - consistent performance often beats higher average with high variance

## Files Modified

- ✅ `tests/test_optimization_quality.py` - Added multi-seed support
- ✅ `tests/test_maxfun_investigation.py` - Created to investigate the issue
- ✅ `SEED_BIAS_DISCOVERY.md` - This document

## Running the Fixed Test

```bash
# Quick test with single seed per config (but varied seeds)
# Edit test_optimization_quality.py: N_SEEDS=1, BASE_SEED=100
python tests/test_optimization_quality.py

# Robust test with 3 seeds per config (RECOMMENDED)
# Edit test_optimization_quality.py: N_SEEDS=3, BASE_SEED=100
python tests/test_optimization_quality.py  # Takes ~15-20 minutes

# Investigation test to check seed sensitivity
python tests/test_maxfun_investigation.py
```

## What We Learned

The optimization landscape for wake steering is:
- **Non-convex** with multiple local optima
- **Stochastic** due to dual annealing's random exploration
- **Sensitive to discretization** (dt_opt, T_opt affect problem structure)

Therefore:
- No single "best" configuration exists
- Trade-offs exist between speed, quality, and consistency
- Must measure performance statistically across multiple seeds
