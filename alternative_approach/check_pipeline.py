#!/usr/bin/env python
"""
Pipeline Status Checker
=======================

Checks which steps of the surrogate MPC pipeline are complete.
"""

from pathlib import Path
import sys

def check_step(name: str, check_func, fix_command: str) -> bool:
    """Check if a step is complete."""
    result = check_func()
    status = "✅" if result else "❌"
    print(f"{status} {name}")
    if not result:
        print(f"   Fix: {fix_command}")
    return result

def main():
    print("="*70)
    print("Surrogate MPC Pipeline Status")
    print("="*70)
    print()

    all_good = True

    # Step 1: Dataset generated
    step1 = check_step(
        "Step 1: Dataset Generated",
        lambda: (Path("data/surrogate_dataset.h5").exists() and
                Path("data/surrogate_dataset_train.h5").exists() and
                Path("data/surrogate_dataset_val.h5").exists() and
                Path("data/surrogate_dataset_test.h5").exists()),
        "python scripts/generate_dataset_v2.py"
    )
    all_good = all_good and step1

    # Step 2: Model trained
    models_dir = Path("models")
    checkpoints_dir = models_dir / "checkpoints"

    # Check for any .ckpt file in checkpoints
    checkpoint_exists = False
    checkpoint_file = None
    if checkpoints_dir.exists():
        ckpt_files = list(checkpoints_dir.glob("*.ckpt"))
        if ckpt_files:
            checkpoint_exists = True
            checkpoint_file = ckpt_files[0]  # Get first one

    step2 = check_step(
        "Step 2: Model Trained",
        lambda: checkpoint_exists,
        "python scripts/train_surrogate_v2.py --max_epochs 100"
    )
    all_good = all_good and step2

    if checkpoint_exists:
        print(f"   Found checkpoint: {checkpoint_file}")

    # Step 3: Model exported
    step3 = check_step(
        "Step 3: Model Exported to l4casadi",
        lambda: Path("models/power_surrogate_casadi.pkl").exists(),
        "python scripts/export_l4casadi_model_v2.py"
    )
    all_good = all_good and step3

    print()
    print("="*70)

    if all_good:
        print("✅ All steps complete! You can now run:")
        print("   python nmpc_surrogate_casadi.py")
    else:
        print("❌ Some steps are missing. Run the commands above in order.")
        print()
        print("Quick start:")
        print("  1. python scripts/generate_dataset_v2.py")
        print("  2. python scripts/train_surrogate_v2.py --max_epochs 100")
        print("  3. python scripts/export_l4casadi_model_v2.py")
        print("  4. python nmpc_surrogate_casadi.py")

    print("="*70)

    # Show file sizes if they exist
    print()
    print("File sizes:")
    files_to_check = [
        "data/surrogate_dataset.h5",
        "data/surrogate_dataset_train.h5",
        "data/surrogate_dataset_val.h5",
        "data/surrogate_dataset_test.h5",
        "models/power_surrogate_casadi.pkl",
    ]

    for file_path in files_to_check:
        p = Path(file_path)
        if p.exists():
            size_mb = p.stat().st_size / (1024**2)
            print(f"  {file_path}: {size_mb:.1f} MB")

    if checkpoints_dir.exists():
        for ckpt in checkpoints_dir.glob("*.ckpt"):
            size_mb = ckpt.stat().st_size / (1024**2)
            print(f"  {ckpt}: {size_mb:.1f} MB")

if __name__ == '__main__':
    main()
