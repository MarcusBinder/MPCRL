# Documentation Index

**Complete guide to all documentation in this repository**

---

## 📖 Reading Guide

### For New Readers

Start here to understand the project:

1. **[README.md](README.md)** ⭐ **START HERE** ⭐
   - Project overview and goals
   - Two approaches (model-free vs model-based)
   - Current status and key findings
   - Quick start guide

---

### Understanding the acados Implementation

Read these in order to follow the implementation journey:

2. **[ACADOS_QP_DIAGNOSTICS.md](ACADOS_QP_DIAGNOSTICS.md)**
   - **What:** Original problem diagnosis
   - **When to read:** After README, before implementation details
   - **Key content:** QP solver failures, numerical conditioning issues
   - **Status:** Historical (problem now solved)

3. **[FINAL_FIX_SUMMARY.md](FINAL_FIX_SUMMARY.md)**
   - **What:** Solutions to numerical problems
   - **When to read:** After diagnostics
   - **Key content:**
     - Control normalization (main fix)
     - Gradient computation at delayed yaws
     - Symmetry breaking for wake deflection
   - **Status:** Current implementation

4. **[LINEARIZATION_LIMITATION.md](LINEARIZATION_LIMITATION.md)**
   - **What:** Why gradient-based MPC fails for this problem
   - **When to read:** After understanding the fixes
   - **Key content:**
     - Delayed causality problem (immediate cost, delayed benefit)
     - Nonlinear power function with weak gradients
     - MPC horizon << wake delay
   - **Status:** Fundamental finding

5. **[MPC_ALTERNATIVES.md](MPC_ALTERNATIVES.md)**
   - **What:** Analysis of 6 different MPC approaches
   - **When to read:** When considering alternatives
   - **Key content:**
     - Nonlinear MPC with surrogate models
     - Sample-based MPC (MPPI/CEM)
     - Hybrid global-local architecture ⭐ recommended
     - Multiple-starts approach
     - Learning-enhanced MPC
     - Trajectory optimization
   - **Status:** Design exploration

6. **[FINAL_RECOMMENDATIONS.md](FINAL_RECOMMENDATIONS.md)**
   - **What:** Production-ready architecture recommendations
   - **When to read:** When implementing a solution
   - **Key content:**
     - Two-layer architecture (strategic + tactical)
     - Lookup table + MPC tracking
     - Implementation timeline and roadmap
   - **Status:** Active recommendation

---

### Previous Work (MPC+RL)

If you want context on the original model-free approach:

7. **[archive/PAPER_OUTLINE_V2.md](archive/PAPER_OUTLINE_V2.md)**
   - **What:** Complete paper outline for MPC+RL approach
   - **Key content:** Balanced MPC-RL narrative, experimental plan
   - **Status:** Complete, for reference

8. **[archive/WAKE_DELAY_FIX_SUMMARY.md](archive/WAKE_DELAY_FIX_SUMMARY.md)**
   - **What:** Wake delay handling in model-free MPC
   - **Key finding:** Evaluation horizon must capture full wake propagation
   - **Status:** Historical

9. **[archive/SEED_BIAS_DISCOVERY.md](archive/SEED_BIAS_DISCOVERY.md)**
   - **What:** Importance of multi-seed averaging
   - **Key finding:** Single-seed tests can be misleading
   - **Status:** Methodological insight

10. **[archive/PERFORMANCE_OPTIMIZATION_GUIDE.md](archive/PERFORMANCE_OPTIMIZATION_GUIDE.md)**
    - **What:** Optimization guide for scipy-based MPC
    - **Key content:** Parameter tuning, caching strategies
    - **Result:** 6-8x speedup achieved
    - **Status:** For model-free implementation

11. **[archive/MPC_FIX_SUMMARY.md](archive/MPC_FIX_SUMMARY.md)**
    - **What:** Earlier fix attempts
    - **Status:** Superseded by FINAL_FIX_SUMMARY.md

---

## 📂 Document Organization

```
docs/
├── README.md                        ⭐ START HERE - Project overview
├── INDEX.md                         📖 This file - Documentation guide
│
├── acados Implementation Journey:
│   ├── ACADOS_QP_DIAGNOSTICS.md    Problem diagnosis
│   ├── FINAL_FIX_SUMMARY.md        Solutions (control normalization, etc.)
│   ├── LINEARIZATION_LIMITATION.md  Fundamental limitation explained
│   ├── MPC_ALTERNATIVES.md         6 alternative approaches analyzed
│   └── FINAL_RECOMMENDATIONS.md    Production recommendations
│
└── archive/                         Previous work (MPC+RL)
    ├── PAPER_OUTLINE_V2.md         MPC+RL paper outline
    ├── WAKE_DELAY_FIX_SUMMARY.md   Wake delay handling
    ├── SEED_BIAS_DISCOVERY.md      Multi-seed importance
    ├── PERFORMANCE_OPTIMIZATION_GUIDE.md  Scipy MPC optimization
    ├── MPC_FIX_SUMMARY.md          Early fix attempts
    └── [other archived docs]
```

---

## 🎯 Reading Paths

### Path 1: Quick Overview (15 minutes)

Just want to understand what we're doing?

1. [README.md](README.md) - Sections: Overview, Two Approaches, Key Findings
2. [LINEARIZATION_LIMITATION.md](LINEARIZATION_LIMITATION.md) - Section: The Problem

**Outcome:** Understand the project and main challenge

---

### Path 2: Implementation Details (1-2 hours)

Want to understand the code and run examples?

1. [README.md](README.md) - Full document
2. [FINAL_FIX_SUMMARY.md](FINAL_FIX_SUMMARY.md) - All sections
3. [README.md](README.md) - Section: Quick Start
4. Run examples in `../examples/`

**Outcome:** Understand implementation, able to run code

---

### Path 3: Deep Technical Understanding (3-4 hours)

Want to fully understand the technical challenges and solutions?

1. [README.md](README.md)
2. [ACADOS_QP_DIAGNOSTICS.md](ACADOS_QP_DIAGNOSTICS.md)
3. [FINAL_FIX_SUMMARY.md](FINAL_FIX_SUMMARY.md)
4. [LINEARIZATION_LIMITATION.md](LINEARIZATION_LIMITATION.md)
5. [MPC_ALTERNATIVES.md](MPC_ALTERNATIVES.md)
6. [FINAL_RECOMMENDATIONS.md](FINAL_RECOMMENDATIONS.md)

**Outcome:** Complete technical understanding, ready to contribute

---

### Path 4: Previous Work Context (2-3 hours)

Want to understand the MPC+RL background?

1. [README.md](README.md) - Section: Two Approaches
2. [archive/PAPER_OUTLINE_V2.md](archive/PAPER_OUTLINE_V2.md)
3. [archive/WAKE_DELAY_FIX_SUMMARY.md](archive/WAKE_DELAY_FIX_SUMMARY.md)
4. [archive/PERFORMANCE_OPTIMIZATION_GUIDE.md](archive/PERFORMANCE_OPTIMIZATION_GUIDE.md)

**Outcome:** Understand both approaches, can compare them

---

## 📊 Key Results Summary

### acados Implementation

| Metric | Value | Status |
|--------|-------|--------|
| **Solve time** | <1ms | ✅ Achieved |
| **QP convergence** | res_stat < 1e-9 | ✅ Achieved |
| **Numerical stability** | Well-conditioned | ✅ Achieved |
| **Optimal from cold start** | Cannot achieve | ❌ Fundamental limitation |
| **Power gain (cold start)** | -0.2% | ❌ Stuck in local minimum |
| **Hybrid architecture** | TBD | ⚙️ In progress |

### Model-Free MPC+RL (Previous Work)

| Metric | Value | Status |
|--------|-------|--------|
| **MPC gain** | +11.8% | ✅ Complete |
| **RL gain** | +2.6% | ✅ Complete |
| **Total gain** | +14.4% | ✅ Complete |
| **Solve time** | Seconds to minutes | ⚠️ Slow |
| **Real-time suitable** | No | ⚠️ Too slow |

---

## 🔍 Finding Specific Information

### How do I...

**...understand why the QP solver was failing?**
→ [ACADOS_QP_DIAGNOSTICS.md](ACADOS_QP_DIAGNOSTICS.md)

**...learn about the control normalization fix?**
→ [FINAL_FIX_SUMMARY.md](FINAL_FIX_SUMMARY.md) - Section: Control Normalization

**...understand why gradient MPC doesn't work?**
→ [LINEARIZATION_LIMITATION.md](LINEARIZATION_LIMITATION.md) - Sections: Root Cause, Why Longer Horizon Doesn't Help

**...see alternative approaches?**
→ [MPC_ALTERNATIVES.md](MPC_ALTERNATIVES.md) - All options analyzed with pros/cons

**...implement the hybrid architecture?**
→ [FINAL_RECOMMENDATIONS.md](FINAL_RECOMMENDATIONS.md) - Section: Recommended Architecture
→ `../examples/hybrid_mpc_example.py` - Working code example

**...run the code?**
→ [README.md](README.md) - Section: Quick Start

**...understand the original MPC+RL approach?**
→ [archive/PAPER_OUTLINE_V2.md](archive/PAPER_OUTLINE_V2.md)

**...tune MPC parameters?**
→ [README.md](README.md) - Section: Understanding the Code
→ [archive/PERFORMANCE_OPTIMIZATION_GUIDE.md](archive/PERFORMANCE_OPTIMIZATION_GUIDE.md) - For scipy-based MPC

---

## 🔑 Key Concepts

### Core Technical Findings

1. **Control Normalization** ([FINAL_FIX_SUMMARY.md](FINAL_FIX_SUMMARY.md))
   - Normalize controls to [-1, 1] for numerical stability
   - Scale cost terms accordingly
   - Critical for QP convergence

2. **Delayed Gradient** ([FINAL_FIX_SUMMARY.md](FINAL_FIX_SUMMARY.md))
   - Compute gradient at yaw angles from 330s ago
   - Power depends on delayed yaw, not current yaw
   - Essential for correct optimization direction

3. **Delayed Causality** ([LINEARIZATION_LIMITATION.md](LINEARIZATION_LIMITATION.md))
   - Immediate cost (yaw misalignment) vs delayed benefit (wake deflection)
   - MPC horizon (100s) << wake delay (330s)
   - **Fundamental limitation** of gradient-based MPC

4. **Hybrid Architecture** ([FINAL_RECOMMENDATIONS.md](FINAL_RECOMMENDATIONS.md))
   - Strategic layer: Find optimal (slow, global)
   - Tactical layer: Track it with MPC (fast, local)
   - Industry-standard solution

---

## 🛠️ Implementation Files

### Core Implementation
- `../nmpc_windfarm_acados_fixed.py` - Current acados implementation

### Examples
- `../examples/hybrid_mpc_example.py` - Two-layer architecture
- `../examples/demo_yaw_optimization.py` - Basic MPC demo
- `../examples/example_warm_start.py` - Warm vs cold start

### Tests
- `../tests/test_optimal_yaw.py` - Ground truth via grid search
- `../tests/test_gradient_*.py` - Gradient debugging
- `../tests/test_long_horizon.py` - Horizon experiments

---

## 📈 Timeline

| Date | Milestone | Documentation |
|------|-----------|---------------|
| Oct 17 | MPC+RL complete | archive/PAPER_OUTLINE_V2.md |
| Oct 20 | QP failures diagnosed | ACADOS_QP_DIAGNOSTICS.md |
| Oct 20 | Numerical stability fixed | FINAL_FIX_SUMMARY.md |
| Oct 20 | Limitation understood | LINEARIZATION_LIMITATION.md |
| Oct 20 | Alternatives explored | MPC_ALTERNATIVES.md |
| Oct 20 | Recommendations made | FINAL_RECOMMENDATIONS.md |
| **Now** | **Hybrid implementation** | *In progress* |

---

## 📧 Getting Help

1. **Start with README.md** - Covers 90% of questions
2. **Check relevant technical doc** - Use this index to find it
3. **Run examples** - See it working
4. **Read the code** - Well-commented implementation

---

**Last Updated:** 2025-10-20
**Status:** Documentation complete, hybrid architecture in progress
**Next:** Validate hybrid approach achieves +15% gain
