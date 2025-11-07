# Normalization Issue with l4casadi

## Problem

The surrogate model has built-in normalization using PyTorch's `register_buffer`:

```python
self.register_buffer('input_mean', torch.zeros(input_dim))
self.register_buffer('input_std', torch.ones(input_dim))
```

When TorchScript traces the model for l4casadi, it doesn't properly trace these buffers, causing the CasADi version to give wrong predictions (off by ~14 MW mean error).

## Solution

**Option 1: External Normalization (Quick Fix)**

Normalize inputs before passing to l4casadi:

```python
# In nmpc_surrogate_casadi.py
# Manually normalize before calling surrogate
yaw_normalized = (yaw - mean_yaw) / std_yaw
wind_normalized = (wind - mean_wind) / std_wind
surrogate_input = ca.vertcat(yaw_normalized, wind_normalized)
```

**Option 2: Retrain Without Built-in Normalization (Clean Fix)**

Modify the model to not have built-in normalization, handle it in the training loop instead.

## Implementation: Option 1 (Quick Fix)

I'll implement this now - it's the fastest path to a working system.

