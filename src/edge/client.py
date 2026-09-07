"""IoT Edge Client encapsulating local model training and privacy-safe gradient extraction."""
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Any
from src.edge.models import LSTMAutoencoder, compute_reconstruction_errors

class EdgeNodeClient:
    """Represents a single distributed IoT sensor node running Edge AI locally.
    
    Ensures absolute data governance: raw sensor readings never leave the node.
    Only parameter gradient differences / model weights are exported during federated rounds.
    """

    def __init__(
        self,
        node_id: int,
        zone_id: int,
        input_dim: int = 7,
        hidden_dim: int = 16,
        seq_len: int = 6,
        lr: float = 0.01,
        device: str = "cpu"
    ):
        self.node_id = node_id
        self.zone_id = zone_id
        self.seq_len = seq_len
        self.device = device
        self.input_dim = input_dim
        
        self.model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.local_data_tensor: Optional[torch.Tensor] = None
        self.local_labels: Optional[np.ndarray] = None
        self._initial_state_dict: Optional[Dict[str, torch.Tensor]] = None

    def load_dataframe(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> None:
        """Transform node multivariate time series into sliding temporal window tensors."""
        if feature_cols is None:
            feature_cols = ["temperature", "humidity", "wind_speed", "wind_direction", "co2", "pm25", "soil_moisture"]
            
        data_matrix = df[feature_cols].values.astype(np.float32)
        labels = df["fire_label"].values.astype(np.int64) if "fire_label" in df.columns else np.zeros(len(df), dtype=np.int64)
        
        # Standard-scale features locally per node to stabilize LSTM gradient convergence
        mean = np.mean(data_matrix, axis=0, keepdims=True)
        std = np.std(data_matrix, axis=0, keepdims=True) + 1e-6
        normalized_data = (data_matrix - mean) / std
        
        num_windows = len(normalized_data) - self.seq_len + 1
        if num_windows <= 0:
            raise ValueError(f"DataFrame length ({len(df)}) must be >= sequence length ({self.seq_len})")
            
        windows = []
        win_labels = []
        for i in range(num_windows):
            windows.append(normalized_data[i : i + self.seq_len])
            # A window is anomalous/fire if any timestamp inside it has fire_label == 1
            win_labels.append(1 if np.any(labels[i : i + self.seq_len] == 1) else 0)
            
        self.local_data_tensor = torch.tensor(np.array(windows), dtype=torch.float32).to(self.device)
        self.local_labels = np.array(win_labels, dtype=np.int64)

    def synchronize_with_server(self, global_state_dict: Dict[str, torch.Tensor]) -> None:
        """Broadcast receiver: update local model weights from federated aggregation round."""
        self.model.load_state_dict(copy.deepcopy(global_state_dict))
        self._initial_state_dict = copy.deepcopy(global_state_dict)

    def train_local_epoch(self, batch_size: int = 16, epochs: int = 1, only_normal_data: bool = True) -> float:
        """Train unsupervised LSTM Autoencoder on local sensor stream."""
        if self.local_data_tensor is None:
            raise RuntimeError("No local data loaded into EdgeNodeClient.")
            
        self.model.train()
        if self._initial_state_dict is None:
            self._initial_state_dict = copy.deepcopy(self.model.state_dict())

        # Unsupervised training: fit reconstruction on normal baseline windows (label == 0)
        train_tensors = self.local_data_tensor
        if only_normal_data and self.local_labels is not None:
            normal_mask = (self.local_labels == 0)
            if np.any(normal_mask):
                train_tensors = train_tensors[normal_mask]

        dataset_size = len(train_tensors)
        total_loss = 0.0
        num_batches = 0
        
        for ep in range(epochs):
            indices = torch.randperm(dataset_size)
            for start_idx in range(0, dataset_size, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                batch = train_tensors[batch_idx]
                
                self.optimizer.zero_grad()
                recon = self.model(batch)
                loss = self.criterion(recon, batch)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / max(num_batches, 1)

    def extract_gradients(self) -> Dict[str, torch.Tensor]:
        """Extract pseudo-gradients / parameter deltas (W_local - W_global).
        
        GUARANTEE: Returns ONLY weight gradient tensors. No raw time series data is exposed.
        This interface forms the strict boundary of decentralized privacy protection.
        """
        if self._initial_state_dict is None:
            raise RuntimeError("Initial model state not tracked. Call synchronize_with_server first.")
            
        gradients = {}
        current_state = self.model.state_dict()
        for param_name, current_val in current_state.items():
            initial_val = self._initial_state_dict[param_name]
            # Delta gradient vector representation
            gradients[param_name] = (current_val.cpu() - initial_val.cpu()).clone().detach()
            
        return gradients

    def compute_anomaly_scores(self) -> np.ndarray:
        """Run anomaly scoring over all local windows to flag wildfire precursors."""
        if self.local_data_tensor is None:
            raise RuntimeError("No local data loaded into EdgeNodeClient.")
        return compute_reconstruction_errors(self.model, self.local_data_tensor)
