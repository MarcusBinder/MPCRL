# Data Directory

This directory stores wind field datasets used by the SAC + MPC workflow.

## Current Data Files

- **Hipersim_mann_l5.0_ae1.0000_g0.0_h0_128x128x128_4.000x8.00x8.00_s0001.nc**
  - Format: NetCDF
  - Size: ~25 MB
  - Description: High-resolution atmospheric simulation data for realistic wind farm modeling
  - Generated using Mann turbulence model

## Data Format

The NetCDF files contain 3D wind field data with turbulence characteristics suitable for wake modeling and wind farm simulations.

## Note

Surrogate datasets and related tooling have been archived under `alternative_approach/`. Data files (*.nc) remain excluded from git; share large files via external storage when necessary.
