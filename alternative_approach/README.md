# Alternative Approach (acados-based NMPC)

**Status:** ⚠️ **RESTARTING - November 2025**

This folder contains gradient-based MPC experiments built on acados, along with surrogate-modelling prototypes. Work was previously paused after identifying fundamental limitations with pure gradient-based approaches. We are now restarting work to implement and validate the recommended hybrid architecture.

## Quick Start

**New to this approach?** Start here:
1. Read [`RESTART_GUIDE.md`](RESTART_GUIDE.md) - Complete overview of what was done and current status
2. Read [`ROADMAP_2025.md`](ROADMAP_2025.md) - Detailed plan for continuing work
3. Read [`docs/INDEX.md`](docs/INDEX.md) - Comprehensive documentation guide

## What Was Discovered

The gradient-based MPC works correctly from a numerical perspective but has a **fundamental limitation**: it cannot find the global optimum due to:
- Delayed causality (wake delay 330s >> MPC horizon 100s)
- Weak gradients near zero yaw
- Highly nonlinear power landscape

**Result:** Pure gradient MPC achieves only 0.4% gain vs 15.1% optimal

**Solution:** Hybrid architecture (strategic planning + tactical MPC) - RECOMMENDED

## Structure

- **[`RESTART_GUIDE.md`](RESTART_GUIDE.md)** ⭐ **START HERE** - Complete restart guide
- **[`ROADMAP_2025.md`](ROADMAP_2025.md)** - Detailed roadmap for 2025
- `docs/` – Comprehensive investigation notes, limitations, and design proposals
  - [`docs/INDEX.md`](docs/INDEX.md) - Documentation guide
  - [`docs/LINEARIZATION_LIMITATION.md`](docs/LINEARIZATION_LIMITATION.md) - Why gradient MPC fails
  - [`docs/MPC_ALTERNATIVES.md`](docs/MPC_ALTERNATIVES.md) - 6 alternative approaches
  - [`docs/FINAL_RECOMMENDATIONS.md`](docs/FINAL_RECOMMENDATIONS.md) - Production recommendations
- `nmpc_windfarm_acados_fixed.py` – Primary acados controller implementation
- `examples/` – Working examples including hybrid architecture
  - `hybrid_mpc_example.py` - Two-layer architecture (strategic + tactical)
  - `demo_yaw_optimization.py` - Basic MPC demo
- `scripts/` – Utilities for diagnostics, surrogate dataset creation, and training
- `surrogate_module/` – Helper modules for dataset management, model training, and CasADi integration
- `tests/` – Validation and diagnostic test suites
  - `test_optimal_yaw.py` - Ground truth via grid search
  - `test_gradient_*.py` - Gradient debugging
- `mpcrl_archive/` – Older NMPC prototypes

## Current Focus

**Phase 1: Validation (Week 1-2)**
- Validate existing hybrid architecture
- Confirm ~15% power gain is achievable
- Multi-condition testing

See [`ROADMAP_2025.md`](ROADMAP_2025.md) for complete plan.

## Quick Test

```bash
# Test the hybrid architecture
cd alternative_approach/
python examples/hybrid_mpc_example.py

# Find optimal yaw (ground truth)
python tests/test_optimal_yaw.py

# Basic MPC demo
python examples/demo_yaw_optimization.py
```

## Key Insights

1. **Numerical stability** ✅ Solved via control normalization
2. **Fast solver** ✅ <1ms per solve with acados
3. **Fundamental limitation** ⚠️ Gradient MPC alone insufficient
4. **Solution identified** ✅ Hybrid architecture recommended
5. **Code ready** ✅ Implementation exists, needs validation

## Next Steps

See [`ROADMAP_2025.md`](ROADMAP_2025.md) for detailed plan:
1. Validate hybrid approach (1-2 weeks)
2. Build production implementation (3-4 weeks)
3. Explore surrogate models (6-8 weeks)
4. Compare with main approach (3-4 weeks)

