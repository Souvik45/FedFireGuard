"""Centralized raw-data learning baseline (Baseline 1) for comparative evaluation."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from src.edge.models import LSTMAutoencoder, compute_reconstruction_errors

class CentralizedLSTMBaseline:
    """Baseline 1: Centralized pooled model where all IoT node sensor data is shipped
    to a single server without privacy protections or local edge computation.
    
    Acts as an upper-bound computational baseline and demonstrates bandwidth requirements
    of legacy cloud pipelines compared to decentralized federated edge architectures.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 16,
        seq_len: int = 6,
        lr: float = 0.01,
        device: str = "cpu"
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.device = device
        
        self.model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.pooled_data_tensor: Optional[torch.Tensor] = None
        self.pooled_labels: Optional[np.ndarray] = None
        self.node_window_slices: Dict[int, Tuple[int, int]] = {}
        self.total_raw_bytes_shipped: int = 0

    def load_and_pool_data(self, node_dfs: Dict[int, pd.DataFrame], feature_cols: Optional[List[str]] = None) -> None:
        """Pool all raw time-series data across IoT nodes into a centralized database."""
        if feature_cols is None:
            feature_cols = ["temperature", "humidity", "wind_speed", "wind_direction", "co2", "pm25", "soil_moisture"]
            
        all_windows = []
        all_labels = []
        current_idx = 0
        
        for node_id, df in sorted(node_dfs.items()):
            # Calculate simulated communication bandwidth payload for raw data transfer
            raw_matrix = df[feature_cols].values.astype(np.float32)
            self.total_raw_bytes_shipped += raw_matrix.nbytes
            
            labels = df["fire_label"].values.astype(np.int64) if "fire_label" in df.columns else np.zeros(len(df), dtype=np.int64)
            
            # Centralized global standardization across pooled features
            mean = np.mean(raw_matrix, axis=0, keepdims=True)
            std = np.std(raw_matrix, axis=0, keepdims=True) + 1e-6
            norm_data = (raw_matrix - mean) / std
            
            num_win = len(norm_data) - self.seq_len + 1
            for i in range(max(0, num_win)):
                all_windows.append(norm_data[i : i + self.seq_len])
                all_labels.append(1 if np.any(labels[i : i + self.seq_len] == 1) else 0)
                
            self.node_window_slices[node_id] = (current_idx, current_idx + max(0, num_win))
            current_idx += max(0, num_win)
            
        if all_windows:
            self.pooled_data_tensor = torch.tensor(np.array(all_windows), dtype=torch.float32).to(self.device)
            self.pooled_labels = np.array(all_labels, dtype=np.int64)
        else:
            self.pooled_data_tensor = torch.empty((0, self.seq_len, self.input_dim), device=self.device)
            self.pooled_labels = np.empty((0,), dtype=np.int64)

    def train_pooled_model(self, epochs: int = 5, batch_size: int = 32, only_normal_data: bool = True) -> float:
        """Train the centralized autoencoder on all normal pooled sensor windows."""
        if self.pooled_data_tensor is None or len(self.pooled_data_tensor) == 0:
            raise RuntimeError("No pooled data available for training.")
            
        self.model.train()
        train_tensors = self.pooled_data_tensor
        if only_normal_data and self.pooled_labels is not None:
            normal_mask = (self.pooled_labels == 0)
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
                
        return total_loss / max(1, num_batches)

    def evaluate_node_anomaly_scores(self) -> Dict[int, np.ndarray]:
        """Compute anomaly scores per individual node from the pooled global model."""
        if self.pooled_data_tensor is None:
            raise RuntimeError("No data pooled.")
            
        scores_by_node = {}
        all_scores = compute_reconstruction_errors(self.model, self.pooled_data_tensor)
        
        for node_id, (start_idx, end_idx) in self.node_window_slices.items():
            scores_by_node[node_id] = all_scores[start_idx : end_idx]
            
        return scores_by_node

    def get_bandwidth_overhead_kb(self) -> float:
        """Return total kilovolt bytes of raw time series shipped across network."""
        return round(self.total_raw_bytes_shipped / 1024.0, 2)
