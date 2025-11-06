"""
Train Surrogate Power Model
============================

Train a neural network to predict wind farm power from yaw angles and wind conditions.

Usage:
    python train_surrogate_v2.py --dataset data/surrogate_dataset_train.h5
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from surrogate_module.model import PowerSurrogate


class PowerDataset(Dataset):
    """PyTorch Dataset for power prediction."""

    def __init__(self, h5_path: str):
        """
        Load dataset from HDF5 file.

        Args:
            h5_path: Path to HDF5 file
        """
        with h5py.File(h5_path, 'r') as f:
            yaw = f['yaw'][:]
            wind_speed = f['wind_speed'][:]
            wind_direction = f['wind_direction'][:]
            power = f['power'][:]

        # Combine inputs: [yaw_0, yaw_1, yaw_2, yaw_3, wind_speed, wind_direction]
        self.X = np.column_stack([yaw, wind_speed, wind_direction])
        self.y = power.reshape(-1, 1)

        print(f"Loaded dataset from {h5_path}:")
        print(f"  Samples: {len(self.X):,}")
        print(f"  Input shape: {self.X.shape}")
        print(f"  Output shape: {self.y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y

    def get_statistics(self):
        """Compute dataset statistics for normalization."""
        return {
            'input_mean': self.X.mean(axis=0),
            'input_std': self.X.std(axis=0),
            'output_mean': self.y.mean(axis=0),
            'output_std': self.y.std(axis=0),
        }


class PowerSurrogateLightning(pl.LightningModule):
    """PyTorch Lightning wrapper for training."""

    def __init__(
        self,
        model: PowerSurrogate,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model.forward_normalized(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Normalize output for training
        y_normalized = (y - self.model.output_mean) / (self.model.output_std + 1e-8)

        # Predict
        y_pred = self.forward(x)

        # MSE loss
        loss = nn.functional.mse_loss(y_pred, y_normalized)

        # Log
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Predict (denormalized)
        y_pred = self.model(x, normalized=False)

        # Metrics on denormalized values
        mse = nn.functional.mse_loss(y_pred, y)
        mae = torch.abs(y_pred - y).mean()

        # Relative error
        relative_error = (torch.abs(y_pred - y) / (torch.abs(y) + 1e-8)).mean()

        # Log
        self.log('val_mse', mse, prog_bar=True)
        self.log('val_mae', mae, prog_bar=True)
        self.log('val_relative_error', relative_error, prog_bar=True)

        return {'val_loss': mse, 'val_mae': mae}

    def test_step(self, batch, batch_idx):
        x, y = batch

        # Predict
        y_pred = self.model(x, normalized=False)

        # Metrics
        mse = nn.functional.mse_loss(y_pred, y)
        mae = torch.abs(y_pred - y).mean()
        relative_error = (torch.abs(y_pred - y) / (torch.abs(y) + 1e-8)).mean()

        # R² score
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot

        self.log('test_mse', mse)
        self.log('test_mae', mae)
        self.log('test_relative_error', relative_error)
        self.log('test_r2', r2)

        return {'test_loss': mse, 'test_mae': mae, 'test_r2': r2}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=20
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_mse',
            }
        }


def train(
    train_dataset_path: str,
    val_dataset_path: str,
    output_dir: str = 'models',
    batch_size: int = 256,
    max_epochs: int = 1000,
    learning_rate: float = 1e-3,
    gpus: int = 0,
):
    """Train surrogate model."""

    print("=" * 70)
    print("Surrogate Model Training")
    print("=" * 70)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = PowerDataset(train_dataset_path)
    val_dataset = PowerDataset(val_dataset_path)

    # Get statistics for normalization
    stats = train_dataset.get_statistics()
    print("\nDataset statistics:")
    print(f"  Input mean: {stats['input_mean']}")
    print(f"  Input std: {stats['input_std']}")
    print(f"  Output mean: {stats['output_mean'][0]/1e6:.2f} MW")
    print(f"  Output std: {stats['output_std'][0]/1e6:.2f} MW")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    print("\nCreating model...")
    model = PowerSurrogate(
        input_dim=6,
        hidden_dims=[64, 64, 32],
        output_dim=1,
        activation='tanh'
    )

    # Set normalization
    model.set_normalization(
        stats['input_mean'],
        stats['input_std'],
        stats['output_mean'],
        stats['output_std']
    )

    # Wrap in Lightning module
    pl_model = PowerSurrogateLightning(model, learning_rate=learning_rate)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / 'checkpoints',
        filename='best-{epoch:02d}-{val_mae:.2f}',
        monitor='val_mae',
        mode='min',
        save_top_k=3,
    )

    early_stop_callback = EarlyStopping(
        monitor='val_mae',
        patience=50,
        mode='min',
        verbose=True
    )

    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir / 'logs',
        name='power_surrogate'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator='gpu' if gpus > 0 else 'cpu',
        devices=gpus if gpus > 0 else 1,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
    )

    # Train
    print("\nStarting training...")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {'GPU' if gpus > 0 else 'CPU'}")

    start_time = time.time()
    trainer.fit(pl_model, train_loader, val_loader)
    train_time = time.time() - start_time

    print(f"\n✅ Training complete!")
    print(f"  Total time: {train_time/60:.1f} minutes")
    print(f"  Best model: {checkpoint_callback.best_model_path}")

    # Load best model
    best_model_path = checkpoint_callback.best_model_path
    pl_model = PowerSurrogateLightning.load_from_checkpoint(
        best_model_path,
        model=model
    )

    # Save final model
    final_model_path = output_dir / 'power_surrogate.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.get_config(),
        'normalization': {
            'input_mean': stats['input_mean'].tolist(),
            'input_std': stats['input_std'].tolist(),
            'output_mean': stats['output_mean'].tolist(),
            'output_std': stats['output_std'].tolist(),
        },
        'training': {
            'train_dataset': train_dataset_path,
            'val_dataset': val_dataset_path,
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'learning_rate': learning_rate,
            'train_time': train_time,
        }
    }, final_model_path)

    print(f"  ✅ Model saved to {final_model_path}")

    # Save config as JSON
    config_path = output_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'model': model.get_config(),
            'training': {
                'batch_size': batch_size,
                'max_epochs': max_epochs,
                'learning_rate': learning_rate,
            },
            'normalization': {
                'input_mean': stats['input_mean'].tolist(),
                'input_std': stats['input_std'].tolist(),
                'output_mean': stats['output_mean'].tolist(),
                'output_std': stats['output_std'].tolist(),
            }
        }, f, indent=2)
    print(f"  ✅ Config saved to {config_path}")

    return model, trainer


def test(model: PowerSurrogate, test_dataset_path: str, batch_size: int = 256):
    """Test model on test set."""

    print("\n" + "=" * 70)
    print("Testing Model")
    print("=" * 70)

    test_dataset = PowerDataset(test_dataset_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    pl_model = PowerSurrogateLightning(model)

    trainer = pl.Trainer(accelerator='cpu', devices=1)
    results = trainer.test(pl_model, test_loader)

    print("\n✅ Test Results:")
    print(f"  MAE: {results[0]['test_mae']/1e6:.3f} MW")
    print(f"  MSE: {results[0]['test_mse']/1e12:.3f} MW²")
    print(f"  R²: {results[0]['test_r2']:.4f}")
    print(f"  Relative Error: {results[0]['test_relative_error']*100:.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train power surrogate model')
    parser.add_argument('--train_dataset', type=str,
                        default='data/surrogate_dataset_train.h5',
                        help='Training dataset path')
    parser.add_argument('--val_dataset', type=str,
                        default='data/surrogate_dataset_val.h5',
                        help='Validation dataset path')
    parser.add_argument('--test_dataset', type=str,
                        default='data/surrogate_dataset_test.h5',
                        help='Test dataset path')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=1000,
                        help='Maximum epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--gpus', type=int, default=0,
                        help='Number of GPUs (0 for CPU)')

    args = parser.parse_args()

    # Train
    model, trainer = train(
        train_dataset_path=args.train_dataset,
        val_dataset_path=args.val_dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        gpus=args.gpus,
    )

    # Test
    test(model, args.test_dataset, batch_size=args.batch_size)

    print("\n" + "=" * 70)
    print("✅ Training and testing complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Export model for l4casadi:")
    print(f"     python scripts/export_l4casadi_model.py --model {args.output_dir}/power_surrogate.pth")


if __name__ == '__main__':
    main()
