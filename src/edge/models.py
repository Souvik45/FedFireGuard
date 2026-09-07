"""Edge TinyML module: LSTM Autoencoder for on-device anomaly detection.

Transmits only model gradients to the FL server — raw sensor data never leaves the node.
Designed for quantization-aware training targeting embedded MCU deployment (Cortex-M4).
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence LSTM Autoencoder for multivariate sensor anomaly detection.

    Architecture:
        Encoder: 2-layer LSTM compresses input sequence into a fixed latent vector.
        Decoder: 2-layer LSTM reconstructs the input sequence from the latent vector.

    High reconstruction error at inference time signals a pre-ignition anomaly —
    the deviation from learned normal microclimate behaviour is the detection signal.

    Args:
        input_dim:   Number of sensor features (default 7: temp, hum, wind_speed,
                     wind_dir, co2, pm25, soil_moisture).
        hidden_dim:  LSTM hidden state size. Keep ≤ 32 for MCU deployment.
        num_layers:  Stacked LSTM depth for encoder and decoder (default 2).
        dropout:     Dropout between LSTM layers when num_layers > 1 (default 0.1).
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        lstm_dropout = dropout if num_layers > 1 else 0.0

        # Encoder: compresses time-series window → latent context vector
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # Bottleneck: project hidden state → compact latent representation
        self.latent_proj = nn.Linear(hidden_dim, hidden_dim)

        # Decoder: reconstructs full sequence from repeated latent vector
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # Output projection: map decoder hidden states → original feature space
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequence to latent context.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            latent: Bottleneck representation (batch, hidden_dim).
            enc_hidden: Full encoder hidden state tuple for skip connections.
        """
        _, (h_n, c_n) = self.encoder(x)
        # Take final layer hidden state as sequence summary
        latent = torch.tanh(self.latent_proj(h_n[-1]))
        return latent, (h_n, c_n)

    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode latent vector back to full sequence.

        Args:
            latent: Bottleneck tensor (batch, hidden_dim).
            seq_len: Target reconstruction length (must match input seq_len).

        Returns:
            Reconstructed sequence (batch, seq_len, input_dim).
        """
        # Repeat latent vector across time steps as decoder input
        repeated = latent.unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder(repeated)
        reconstruction = self.output_proj(dec_out)
        return reconstruction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencoder forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Reconstruction tensor of same shape as x.
        """
        seq_len = x.size(1)
        latent, _ = self.encode(x)
        reconstruction = self.decode(latent, seq_len)
        return reconstruction

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mean MSE reconstruction loss over a batch.

        Used during federated training with Opacus DP-SGD.

        Args:
            x: Input batch (batch, seq_len, input_dim).

        Returns:
            Scalar mean MSE loss.
        """
        x_hat = self.forward(x)
        return nn.functional.mse_loss(x_hat, x, reduction="mean")

    def quantize_int8(self) -> "LSTMAutoencoder":
        """Apply PyTorch dynamic INT8 quantization for MCU/TinyML deployment.

        Returns the quantized model in-place (eval mode required).
        Note: LSTM quantization is CPU-only in PyTorch's dynamic quant backend.
        """
        self.eval()
        quantized = torch.quantization.quantize_dynamic(
            self,
            qconfig_spec={nn.LSTM, nn.Linear},
            dtype=torch.qint8,
        )
        return quantized


def compute_reconstruction_errors(
    model: LSTMAutoencoder,
    windows: torch.Tensor,
) -> np.ndarray:
    """Compute per-window MSE reconstruction error for anomaly scoring.

    Higher error = greater deviation from learned normal behaviour = higher fire risk.
    Runs in inference mode (no gradient tracking) for edge deployment efficiency.

    Args:
        model:   Trained LSTMAutoencoder in eval mode.
        windows: Sliding-window tensor of shape (num_windows, seq_len, input_dim).

    Returns:
        NumPy array of shape (num_windows,) containing per-window MSE scores.
        Values are non-negative; typical normal range < 0.05, anomaly > 0.3.
    """
    model.eval()
    errors = []

    with torch.no_grad():
        # Process in small batches to stay within MCU memory budget
        batch_size = 32
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            reconstruction = model(batch)
            # Per-window MSE: mean over (seq_len, input_dim) dimensions
            mse = ((reconstruction - batch) ** 2).mean(dim=(1, 2))
            errors.append(mse.cpu().numpy())

    return np.concatenate(errors, axis=0)


def build_edge_model(config: dict) -> LSTMAutoencoder:
    """Instantiate LSTMAutoencoder from a config dictionary (loaded from base.yaml).

    Args:
        config: Dict with optional keys: input_dim, hidden_dim, num_layers, dropout.

    Returns:
        Initialised LSTMAutoencoder ready for federated training.
    """
    return LSTMAutoencoder(
        input_dim=config.get("input_dim", 7),
        hidden_dim=config.get("hidden_dim", 32),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.1),
    )
