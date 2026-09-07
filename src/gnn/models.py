"""Spatio-Temporal Graph Attention Network (GAT) with dynamic node dropout masking."""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from typing import Dict, Optional, Any

class SpatioTemporalGAT(nn.Module):
    """Graph Attention Network predicting 6-hour wildfire spread belief maps per zone.
    
    Designed to degrade gracefully under high sensor burnout/dropout (up to 30%+ of nodes offline)
    by leveraging self-attention mechanisms to dynamically reweight operational neighboring sensors.
    """

    def __init__(self, in_features: int = 9, hidden_dim: int = 32, num_heads: int = 2, num_zones: int = 2):
        super(SpatioTemporalGAT, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_zones = num_zones

        # GAT Layer 1: Multi-head directional graph attention
        self.gat1 = GATConv(
            in_channels=in_features,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=True,
            edge_dim=1  # Receives our directed wind/distance edge weights
        )
        
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.2)

        # GAT Layer 2: Aggregation down to concise embedding
        self.gat2 = GATConv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            edge_dim=1
        )
        
        # Classifier mapping aggregated zone representations to 6-hour fire probability belief map
        self.zone_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        zone_ids: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass generating per-zone fire belief probability array.
        
        Args:
            x: Node feature tensor (N, 9).
            edge_index: Graph adjacency connections (2, E).
            edge_attr: Directional wind/distance edge weights (E, 1).
            zone_ids: Zone index per node (N,).
            active_mask: Optional binary tensor indicator (N,) where 1=active, 0=offline/masked.
            
        Returns:
            Tensor of shape (num_zones,) representing predicted wildfire spread probabilities over 6 hours.
        """
        # Explicit node masking: Zero out features of offline/burned sensors
        if active_mask is not None:
            x = x * active_mask.unsqueeze(-1).to(x.device)
            # Override operational indicator channel (index 8) directly to 0
            x[:, 8] = active_mask.to(x.device).float()

        # Graph attention transformations
        h = self.gat1(x, edge_index, edge_attr=edge_attr)
        h = self.elu(h)
        h = self.dropout(h)
        
        h = self.gat2(h, edge_index, edge_attr=edge_attr)
        h = self.elu(h)

        # Zone aggregation: Average representations of operational nodes within each geographic zone
        zone_embeddings = []
        for z in range(self.num_zones):
            z_mask = (zone_ids == z)
            if active_mask is not None:
                z_mask = z_mask & (active_mask == 1)
                
            if torch.any(z_mask):
                z_emb = torch.mean(h[z_mask], dim=0)
            else:
                # Fallback if an entire zone is temporarily uncontactable: use global mean or zeros
                if torch.any(zone_ids == z):
                    z_emb = torch.mean(h[zone_ids == z], dim=0)
                else:
                    z_emb = torch.zeros(self.hidden_dim, device=x.device)
            zone_embeddings.append(z_emb)

        zone_matrix = torch.stack(zone_embeddings, dim=0)
        probs = self.zone_classifier(zone_matrix).squeeze(-1)
        return probs
