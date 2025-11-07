#!/usr/bin/env python
"""
Run Complete Surrogate MPC Pipeline
====================================

Runs all steps: Generate → Train → Export → Validate → Test
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print("\n" + "="*70)
    print(f"▶ {description}")
    print("="*70)
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed!")
        print(f"Return code: {result.returncode}")
        sys.exit(1)

    print(f"\n✅ {description} complete!")

def main():
    print("="*70)
    print("Surrogate MPC Pipeline - Full Run")
    print("="*70)
    print()
    print("This will run all steps:")
    print("  1. Generate dataset (800 samples)")
    print("  2. Train model (50 epochs)")
    print("  3. Export to l4casadi")
    print("  4. Validate export")
    print("  5. Run MPC demo")
    print()
    print("Estimated time: ~5-10 minutes")
    print()

    response = input("Continue? [y/N] ").strip().lower()
    if response not in ['y', 'yes']:
        print("Cancelled.")
        sys.exit(0)

    # Step 1: Generate dataset
    run_command(
        ["python", "scripts/generate_dataset_v2.py", "--n_samples", "800"],
        "Step 1/5: Generate Dataset"
    )

    # Step 2: Train model
    run_command(
        ["python", "scripts/train_surrogate_v2.py", "--max_epochs", "50"],
        "Step 2/5: Train Model"
    )

    # Step 3: Export
    run_command(
        ["python", "scripts/export_l4casadi_model_v2.py"],
        "Step 3/5: Export to l4casadi"
    )

    # Step 4: Validate
    run_command(
        ["python", "validate_normalization.py"],
        "Step 4/5: Validate Export"
    )

    # Step 5: Run MPC
    run_command(
        ["python", "nmpc_surrogate_casadi.py"],
        "Step 5/5: Run MPC Demo"
    )

    print("\n" + "="*70)
    print("✅ FULL PIPELINE COMPLETE!")
    print("="*70)
    print()
    print("The surrogate MPC is now working end-to-end!")
    print()
    print("Next steps:")
    print("  - Generate more data (10k-100k samples) for better accuracy")
    print("  - Tune MPC parameters (horizon, weights, etc.)")
    print("  - Compare with other approaches")
    print("  - Integrate with WindGym")
    print()

if __name__ == '__main__':
    main()
