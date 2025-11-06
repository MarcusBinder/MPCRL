"""
Dataset Generation for Surrogate Model Training
================================================

Generates training data by running PyWake simulations with sampled yaw angles
and wind conditions. Uses Latin hypercube sampling for good coverage.

Usage:
    python generate_dataset_v2.py --n_samples 100000 --n_jobs 8
"""

import argparse
import json
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple

import numpy as np
import h5py
from scipy.stats import qmc  # Latin hypercube sampling

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmpc_windfarm_acados_fixed import build_pywake_model, pywake_farm_power, Farm, Wind


class DatasetGenerator:
    """Generate training dataset for surrogate model."""

    def __init__(
        self,
        n_samples: int = 100000,
        n_turbines: int = 4,
        spacing_D: float = 5.0,
        seed: int = 42
    ):
        self.n_samples = n_samples
        self.n_turbines = n_turbines
        self.spacing_D = spacing_D
        self.seed = seed

        # Setup farm
        self.D = 178.0
        self.x = np.array([0.0] + [i * spacing_D * self.D for i in range(1, n_turbines)])
        self.y = np.zeros(n_turbines)

        # Build PyWake model (will be rebuilt in each worker process)
        self.wf_model, self.layout = build_pywake_model(self.x, self.y, self.D)

        print(f"Dataset Generator initialized:")
        print(f"  Samples: {n_samples:,}")
        print(f"  Turbines: {n_turbines}")
        print(f"  Spacing: {spacing_D}D")
        print(f"  Farm layout: {self.x / self.D} D")

    def generate_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate input samples using Latin hypercube sampling."""

        print("\nGenerating samples using Latin Hypercube Sampling...")

        # Define bounds for each variable
        # yaw_0, yaw_1, yaw_2 (upstream turbines): [-30, 30]
        # yaw_3 (downstream): [0, 0] (always aligned)
        # wind_speed: [6, 12]
        # wind_direction: [260, 280] (mainly 270°, with some variation)

        n_features = 6  # yaw_0, yaw_1, yaw_2, yaw_3, wind_speed, wind_direction

        # Create Latin hypercube sampler
        sampler = qmc.LatinHypercube(d=n_features, seed=self.seed)
        samples = sampler.random(n=self.n_samples)

        # Scale to actual ranges
        l_bounds = np.array([-30, -30, -30, 0, 6, 260])  # Lower bounds
        u_bounds = np.array([30, 30, 30, 0, 12, 280])   # Upper bounds

        scaled_samples = qmc.scale(samples, l_bounds, u_bounds)

        # Split into yaw angles and wind conditions
        yaw_samples = scaled_samples[:, :4]  # First 4 columns
        wind_speed_samples = scaled_samples[:, 4]  # 5th column
        wind_dir_samples = scaled_samples[:, 5]  # 6th column

        # Combine wind conditions
        wind_samples = np.column_stack([wind_speed_samples, wind_dir_samples])

        print(f"  Generated {self.n_samples:,} samples")
        print(f"  Yaw range: [{yaw_samples.min():.1f}, {yaw_samples.max():.1f}]°")
        print(f"  Wind speed range: [{wind_speed_samples.min():.1f}, {wind_speed_samples.max():.1f}] m/s")
        print(f"  Wind direction range: [{wind_dir_samples.min():.1f}, {wind_dir_samples.max():.1f}]°")

        return yaw_samples, wind_samples

    def evaluate_sample(self, args: Tuple) -> float:
        """Evaluate a single sample with PyWake (for parallel execution)."""
        idx, yaw, wind_speed, wind_direction = args

        # Build PyWake model (each worker needs its own)
        wf_model, layout = build_pywake_model(self.x, self.y, self.D)

        # Compute power
        power = pywake_farm_power(wf_model, layout, wind_speed, wind_direction, yaw)

        return power

    def generate_dataset(self, n_jobs: int = None) -> Dict:
        """Generate full dataset with parallel PyWake evaluations."""

        if n_jobs is None:
            n_jobs = max(1, cpu_count() - 1)

        print(f"\nStarting dataset generation with {n_jobs} workers...")

        # Generate input samples
        yaw_samples, wind_samples = self.generate_samples()

        # Prepare arguments for parallel evaluation
        args_list = [
            (i, yaw_samples[i], wind_samples[i, 0], wind_samples[i, 1])
            for i in range(self.n_samples)
        ]

        # Parallel evaluation with progress tracking
        print("\nEvaluating samples with PyWake...")
        start_time = time.time()

        power_samples = []

        with Pool(processes=n_jobs) as pool:
            # Use imap for progress tracking
            for i, power in enumerate(pool.imap(self.evaluate_sample, args_list)):
                power_samples.append(power)

                # Progress update every 1000 samples
                if (i + 1) % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    eta = (self.n_samples - i - 1) / rate
                    print(f"  Progress: {i+1:,}/{self.n_samples:,} "
                          f"({100*(i+1)/self.n_samples:.1f}%) | "
                          f"Rate: {rate:.1f} samples/s | "
                          f"ETA: {eta/60:.1f} min")

        power_samples = np.array(power_samples)

        elapsed_total = time.time() - start_time
        print(f"\n✅ Dataset generation complete!")
        print(f"  Total time: {elapsed_total/60:.1f} minutes")
        print(f"  Average rate: {self.n_samples/elapsed_total:.1f} samples/s")

        # Combine into dataset
        dataset = {
            'yaw': yaw_samples,
            'wind_speed': wind_samples[:, 0],
            'wind_direction': wind_samples[:, 1],
            'power': power_samples,
            'metadata': {
                'n_samples': self.n_samples,
                'n_turbines': self.n_turbines,
                'spacing_D': self.spacing_D,
                'D': self.D,
                'x': self.x.tolist(),
                'y': self.y.tolist(),
                'seed': self.seed,
                'generation_time': elapsed_total,
            }
        }

        # Validation
        self.validate_dataset(dataset)

        return dataset

    def validate_dataset(self, dataset: Dict):
        """Validate dataset quality."""

        print("\nValidating dataset...")

        yaw = dataset['yaw']
        power = dataset['power']
        wind_speed = dataset['wind_speed']
        wind_direction = dataset['wind_direction']

        # Check for NaNs
        n_nans = np.isnan(power).sum()
        if n_nans > 0:
            print(f"  ⚠️  WARNING: {n_nans} NaN values found in power!")
        else:
            print(f"  ✅ No NaN values")

        # Check for outliers
        power_mean = power.mean()
        power_std = power.std()
        n_outliers = np.sum(np.abs(power - power_mean) > 5 * power_std)
        if n_outliers > 0:
            print(f"  ⚠️  WARNING: {n_outliers} outliers found (>5σ)")
        else:
            print(f"  ✅ No extreme outliers")

        # Statistics
        print(f"\nDataset statistics:")
        print(f"  Power:")
        print(f"    Mean: {power_mean/1e6:.2f} MW")
        print(f"    Std: {power_std/1e6:.2f} MW")
        print(f"    Min: {power.min()/1e6:.2f} MW")
        print(f"    Max: {power.max()/1e6:.2f} MW")

        print(f"  Yaw angles:")
        print(f"    Mean: {yaw.mean():.2f}°")
        print(f"    Std: {yaw.std():.2f}°")

        print(f"  Wind speed:")
        print(f"    Mean: {wind_speed.mean():.2f} m/s")
        print(f"    Std: {wind_speed.std():.2f} m/s")

        print(f"  Wind direction:")
        print(f"    Mean: {wind_direction.mean():.2f}°")
        print(f"    Std: {wind_direction.std():.2f}°")

    def save_dataset(self, dataset: Dict, output_path: str):
        """Save dataset to HDF5 file."""

        print(f"\nSaving dataset to {output_path}...")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, 'w') as f:
            # Save arrays
            f.create_dataset('yaw', data=dataset['yaw'], compression='gzip')
            f.create_dataset('wind_speed', data=dataset['wind_speed'], compression='gzip')
            f.create_dataset('wind_direction', data=dataset['wind_direction'], compression='gzip')
            f.create_dataset('power', data=dataset['power'], compression='gzip')

            # Save metadata as attributes
            for key, value in dataset['metadata'].items():
                if isinstance(value, list):
                    f.attrs[key] = json.dumps(value)
                else:
                    f.attrs[key] = value

        file_size = output_path.stat().st_size / (1024**2)  # MB
        print(f"  ✅ Saved {file_size:.1f} MB")

        # Also save metadata as JSON for easy reading
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(dataset['metadata'], f, indent=2)
        print(f"  ✅ Metadata saved to {metadata_path}")


def train_val_test_split(dataset_path: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Split dataset into train/val/test sets."""

    print(f"\nSplitting dataset into train/val/test...")
    print(f"  Train: {train_ratio*100:.0f}%")
    print(f"  Val: {val_ratio*100:.0f}%")
    print(f"  Test: {(1-train_ratio-val_ratio)*100:.0f}%")

    # Load dataset
    with h5py.File(dataset_path, 'r') as f:
        yaw = f['yaw'][:]
        wind_speed = f['wind_speed'][:]
        wind_direction = f['wind_direction'][:]
        power = f['power'][:]
        metadata = dict(f.attrs)

    n_samples = len(power)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    # Random shuffle
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_samples)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]

    # Create splits
    splits = {
        'train': {
            'yaw': yaw[train_indices],
            'wind_speed': wind_speed[train_indices],
            'wind_direction': wind_direction[train_indices],
            'power': power[train_indices],
        },
        'val': {
            'yaw': yaw[val_indices],
            'wind_speed': wind_speed[val_indices],
            'wind_direction': wind_direction[val_indices],
            'power': power[val_indices],
        },
        'test': {
            'yaw': yaw[test_indices],
            'wind_speed': wind_speed[test_indices],
            'wind_direction': wind_direction[test_indices],
            'power': power[test_indices],
        },
    }

    print(f"  Train: {len(train_indices):,} samples")
    print(f"  Val: {len(val_indices):,} samples")
    print(f"  Test: {len(test_indices):,} samples")

    # Save splits
    base_path = Path(dataset_path)
    for split_name, split_data in splits.items():
        split_path = base_path.parent / f"{base_path.stem}_{split_name}.h5"

        with h5py.File(split_path, 'w') as f:
            f.create_dataset('yaw', data=split_data['yaw'], compression='gzip')
            f.create_dataset('wind_speed', data=split_data['wind_speed'], compression='gzip')
            f.create_dataset('wind_direction', data=split_data['wind_direction'], compression='gzip')
            f.create_dataset('power', data=split_data['power'], compression='gzip')

            # Copy metadata
            for key, value in metadata.items():
                f.attrs[key] = value
            f.attrs['split'] = split_name

        print(f"  ✅ Saved {split_path}")

    return splits


def main():
    parser = argparse.ArgumentParser(description='Generate surrogate model training dataset')
    parser.add_argument('--n_samples', type=int, default=100000,
                        help='Number of samples to generate (default: 100000)')
    parser.add_argument('--n_jobs', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--output', type=str, default='data/surrogate_dataset.h5',
                        help='Output file path (default: data/surrogate_dataset.h5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no_split', action='store_true',
                        help='Skip train/val/test split')

    args = parser.parse_args()

    print("=" * 70)
    print("Surrogate Model Dataset Generation")
    print("=" * 70)

    # Generate dataset
    generator = DatasetGenerator(
        n_samples=args.n_samples,
        seed=args.seed
    )

    dataset = generator.generate_dataset(n_jobs=args.n_jobs)

    # Save
    generator.save_dataset(dataset, args.output)

    # Split into train/val/test
    if not args.no_split:
        train_val_test_split(args.output)

    print("\n" + "=" * 70)
    print("✅ Dataset generation complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Train surrogate model:")
    print(f"     python scripts/train_surrogate_v2.py --dataset {args.output}")


if __name__ == '__main__':
    main()
