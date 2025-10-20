# Manual Physics Checks

These scripts were ad-hoc experiments created while debugging the MPC/wake pipeline. They are not part of the automated test suite, but remain useful for reproducing intermediate findings (e.g., gradient sanity checks, wake-model comparisons, manual optimisation sweeps).

Each script can be run directly:
```bash
python tests/manual/gradient_correctness.py
python tests/manual/wake_basic.py
# ...
```

They intentionally avoid the `test_*.py` naming convention so that `pytest` does not collect them by default. Feel free to delete individual files once the documented behaviour has been upstreamed into formal tests.
