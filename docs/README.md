# MPCRL Documentation

This directory contains all documentation for the MPCRL (Model Predictive Control + Reinforcement Learning) project.

## 📚 Documentation Index

### Primary Documents

1. **[PAPER_OUTLINE_V2.md](PAPER_OUTLINE_V2.md)** - **START HERE**
   - Complete paper outline for the academic publication
   - Balanced MPC-RL narrative
   - Detailed experimental plan and expected results
   - Implementation roadmap (16-week timeline)
   - **Status**: Active, current version

### Technical Discoveries

2. **[WAKE_DELAY_FIX_SUMMARY.md](WAKE_DELAY_FIX_SUMMARY.md)**
   - Critical bug fix: Evaluation horizon must capture wake propagation
   - Explains why T_eval ≥ max_delay + T_AH is required
   - Documents the fix applied to all test files
   - **Key insight**: Short horizons underestimate wake steering benefits by 50%+

3. **[SEED_BIAS_DISCOVERY.md](SEED_BIAS_DISCOVERY.md)**
   - Critical finding: Single-seed tests can be misleading for stochastic optimization
   - Shows why multi-seed averaging is essential
   - Documents the "lucky seed" phenomenon
   - **Key insight**: Config that appeared 10kW better was actually 17kW worse on average!

### Implementation Guides

4. **[PERFORMANCE_OPTIMIZATION_GUIDE.md](PERFORMANCE_OPTIMIZATION_GUIDE.md)**
   - Comprehensive guide to optimizing MPC performance
   - Analysis of computational bottlenecks
   - Parameter recommendations (dt_opt, T_opt, maxfun)
   - Caching strategies and speedup techniques
   - **Result**: 6-8x speedup achieved

### Archived

5. **[PAPER_OUTLINE_V1_ARCHIVED.md](PAPER_OUTLINE_V1_ARCHIVED.md)**
   - Original paper outline (MPC-heavy version)
   - Kept for reference, superseded by V2
   - **Status**: Archived, use V2 instead

---

## 🎯 Quick Start

**If you're new to this project:**
1. Read the main [README.md](../README.md) in the root directory
2. Review [PAPER_OUTLINE_V2.md](PAPER_OUTLINE_V2.md) for the full picture
3. Check [WAKE_DELAY_FIX_SUMMARY.md](WAKE_DELAY_FIX_SUMMARY.md) and [SEED_BIAS_DISCOVERY.md](SEED_BIAS_DISCOVERY.md) for critical findings

**If you're implementing:**
1. Use [PERFORMANCE_OPTIMIZATION_GUIDE.md](PERFORMANCE_OPTIMIZATION_GUIDE.md) for parameter selection
2. Follow test scripts in `../tests/` directory
3. See [PAPER_OUTLINE_V2.md](PAPER_OUTLINE_V2.md) Section 10 for implementation roadmap

**If you're writing the paper:**
1. Follow structure in [PAPER_OUTLINE_V2.md](PAPER_OUTLINE_V2.md)
2. Reference technical discoveries for key findings
3. Use test scripts to generate figures

---

## 📊 Key Results Summary

### MPC Optimization Results
- **100 configurations tested** (5 dt_opt × 4 T_opt × 5 maxfun)
- **300 total experiments** (3 seeds per configuration)
- **Pareto optimal**: dt_opt=30, T_opt=300, maxfun=10
- **Speedup**: 10.8x faster than reference
- **Quality**: 99.78% of reference (< 1% loss)

### Critical Discoveries
1. **Wake delay requirement**: T_eval ≥ 200s for 3-turbine farm
2. **Seed sensitivity**: Low maxfun has ±21kW variance, high maxfun ±9kW
3. **Parameter interactions**: dt, T, maxfun not independent (emergent sweet spots)

### RL Integration (In Progress)
- **Target**: +2-5% additional gain over optimized MPC
- **Architecture**: Hybrid MPC-RL with bounded corrections (±5°)
- **Expected total gain**: 11.8% (MPC) + 2.6% (RL) = 14.4% vs greedy

---

## 🗂️ Document Organization

```
docs/
├── README.md                           (this file - documentation index)
├── PAPER_OUTLINE_V2.md                 (main paper outline)
├── WAKE_DELAY_FIX_SUMMARY.md          (wake delay discovery)
├── SEED_BIAS_DISCOVERY.md             (seed bias discovery)
├── PERFORMANCE_OPTIMIZATION_GUIDE.md  (implementation guide)
└── PAPER_OUTLINE_V1_ARCHIVED.md       (archived old version)
```

---

## 🔗 Related Directories

- `../tests/` - Test scripts for generating paper results
- `../results/` - Experimental results (data & figures)
- `../examples/` - Usage examples and tutorials
- `../mpcrl/` - Core library implementation

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{mpcrl2025,
  title={Learning-Enhanced Model Predictive Control for Wind Farm Wake Steering},
  author={[Your Name]},
  journal={[Target Venue]},
  year={2025},
  note={In preparation}
}
```

---

**Last Updated**: 2025-10-17
**Status**: Active development, Phase 1 (MPC) complete, Phase 2 (RL) in progress
