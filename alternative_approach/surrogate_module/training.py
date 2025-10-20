from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "NormalizationStats":
        return cls(mean=np.array(data["mean"]), std=np.array(data["std"]))


def load_dataset_npz(path: Path) -> dict:
    """Lightweight loader duplicating mpcrl.surrogate.dataset.load_dataset_npz."""
    npz = np.load(path, allow_pickle=False)
    meta_path = path.with_suffix(".json")
    meta = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
    return {
        "features": npz["features"],
        "targets": npz["targets"],
        "feature_names": npz["feature_names"],
        "metadata": meta,
    }


def _compute_stats(arr: np.ndarray, eps: float = 1e-8) -> NormalizationStats:
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std < eps, eps, std)
    return NormalizationStats(mean=mean, std=std)


class SurrogateMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_layers: Sequence[int], activation: str = "tanh"):
        super().__init__()
        if not hidden_layers:
            raise ValueError("hidden_layers must be non-empty")

        act_cls = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "softplus": nn.Softplus,
        }.get(activation.lower())
        if act_cls is None:
            raise ValueError(f"Unsupported activation '{activation}'")

        layers: List[nn.Module] = []
        last_dim = in_dim
        for width in hidden_layers:
            layers.append(nn.Linear(last_dim, width))
            layers.append(act_cls())
            last_dim = width
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def _prepare_datasets(
    features: np.ndarray,
    targets: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    seed: int,
):
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("Validation and test fractions must be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")

    full_dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(targets.astype(np.float32)),
    )
    n_total = len(full_dataset)
    n_val = int(n_total * val_fraction)
    n_test = int(n_total * test_fraction)
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )
    return train_set, val_set, test_set


def train_surrogate(
    dataset_path: Path,
    output_dir: Path,
    hidden_layers: Sequence[int] = (128, 128, 128),
    activation: str = "tanh",
    epochs: int = 300,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """
    Train an MLP surrogate on the generated dataset.

    Returns a dict with training statistics and paths to saved artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset_npz(dataset_path)
    features = data["features"]
    targets = data["targets"][:, None]  # ensure 2D

    feature_stats = _compute_stats(features)
    target_stats = _compute_stats(targets)

    features_norm = (features - feature_stats.mean) / feature_stats.std
    targets_norm = (targets - target_stats.mean) / target_stats.std

    train_set, val_set, test_set = _prepare_datasets(
        features_norm,
        targets_norm,
        val_fraction,
        test_fraction,
        seed,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = SurrogateMLP(
        in_dim=features_norm.shape[1],
        hidden_layers=hidden_layers,
        activation=activation,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= max(1, len(val_loader.dataset))

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on the test set using denormalised units (MW)
    model.eval()
    preds_list = []
    targets_list = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb)
            preds_list.append(preds.cpu().numpy())
            targets_list.append(yb.cpu().numpy())
    if preds_list:
        preds_norm = np.vstack(preds_list)
        targets_norm = np.vstack(targets_list)
        preds_denorm = preds_norm * target_stats.std + target_stats.mean
        targets_denorm = targets_norm * target_stats.std + target_stats.mean
        mae = float(np.mean(np.abs(preds_denorm - targets_denorm)))
        mape = float(
            np.mean(np.abs((preds_denorm - targets_denorm) / np.maximum(1e-6, targets_denorm)))
        )
    else:
        mae = float("nan")
        mape = float("nan")

    model_path = output_dir / "surrogate_model.pt"
    torch.save(model.state_dict(), model_path)

    stats_payload = {
        "feature_stats": feature_stats.to_dict(),
        "target_stats": target_stats.to_dict(),
        "training_history": history,
        "best_val_loss": best_val_loss,
        "mae_W": mae,
        "mape": mape,
        "hidden_layers": list(hidden_layers),
        "activation": activation,
        "dataset": str(dataset_path),
    }
    with (output_dir / "training_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats_payload, fh, indent=2)

    weights_npz_path = output_dir / "surrogate_weights.npz"
    export_state_dict(model, weights_npz_path)

    return {
        "model_path": model_path,
        "stats_path": output_dir / "training_stats.json",
        "weights_path": weights_npz_path,
        "metrics": {"mae_W": mae, "mape": mape},
    }


def export_state_dict(model: nn.Module, output_path: Path) -> None:
    """Export the model weights and biases to an NPZ file for CasADi embedding."""
    data = {}
    for name, param in model.state_dict().items():
        data[name] = param.cpu().numpy()
    np.savez_compressed(output_path, **data)
