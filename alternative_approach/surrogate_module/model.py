"""
Neural Network Model for Power Surrogate
=========================================

PyTorch implementation of the power prediction model.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict


class PowerSurrogate(nn.Module):
    """
    Neural network to predict total farm power from yaw angles and wind conditions.

    Input: [yaw_0, yaw_1, yaw_2, yaw_3, wind_speed, wind_direction]  # 6 features
    Output: [total_power]  # 1 output

    Architecture: 6 -> 64 -> 64 -> 32 -> 1 with Tanh activations
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: list = [64, 64, 32],
        output_dim: int = 1,
        activation: str = 'tanh'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Select activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim

        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Normalization parameters (will be set during training)
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        self.register_buffer('output_mean', torch.zeros(output_dim))
        self.register_buffer('output_std', torch.ones(output_dim))

        print(f"PowerSurrogate initialized:")
        print(f"  Input: {input_dim}")
        print(f"  Hidden: {hidden_dims}")
        print(f"  Output: {output_dim}")
        print(f"  Activation: {activation}")
        print(f"  Total parameters: {self.count_parameters():,}")

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_normalization(
        self,
        input_mean: np.ndarray,
        input_std: np.ndarray,
        output_mean: np.ndarray,
        output_std: np.ndarray
    ):
        """Set normalization parameters."""
        self.input_mean = torch.tensor(input_mean, dtype=torch.float32)
        self.input_std = torch.tensor(input_std, dtype=torch.float32)
        self.output_mean = torch.tensor(output_mean, dtype=torch.float32)
        self.output_std = torch.tensor(output_std, dtype=torch.float32)

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using stored statistics."""
        return (x - self.input_mean) / (self.input_std + 1e-8)

    def denormalize_output(self, y: torch.Tensor) -> torch.Tensor:
        """Denormalize output using stored statistics."""
        return y * self.output_std + self.output_mean

    def forward(self, x: torch.Tensor, normalized: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, 6]
            normalized: If True, input is already normalized

        Returns:
            Power prediction [batch_size, 1] (denormalized)
        """
        if not normalized:
            x = self.normalize_input(x)

        # Forward through network
        y = self.network(x)

        # Denormalize output
        y = self.denormalize_output(y)

        return y

    def forward_normalized(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with normalized inputs and outputs (for training)."""
        x = self.normalize_input(x)
        y = self.network(x)
        return y

    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'activation': 'tanh',
            'n_parameters': self.count_parameters(),
        }

    def predict(self, yaw: np.ndarray, wind_speed: float, wind_direction: float) -> float:
        """
        Predict power for given inputs (convenience method).

        Args:
            yaw: Yaw angles [4]
            wind_speed: Wind speed (m/s)
            wind_direction: Wind direction (degrees)

        Returns:
            Predicted power (W)
        """
        # Prepare input
        x = np.concatenate([yaw, [wind_speed, wind_direction]])
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

        # Predict
        with torch.no_grad():
            y_tensor = self.forward(x_tensor)

        return float(y_tensor.item())


def create_model(config: Dict = None) -> PowerSurrogate:
    """Create model from configuration."""

    if config is None:
        config = {
            'input_dim': 6,
            'hidden_dims': [64, 64, 32],
            'output_dim': 1,
            'activation': 'tanh'
        }

    return PowerSurrogate(**config)
