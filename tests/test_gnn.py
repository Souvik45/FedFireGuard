import os
import pytest
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from src.gnn.graph_builder import WildfireGraphBuilder
from src.gnn.models import SpatioTemporalGAT
from src.gnn.trainer import GNNFirePredictor
from src.data.generator import WildfireDataGenerator

def test_directional_wind_edge_weights():
    """Verify edge weights are directional and higher when wind blows from source toward destination."""
    builder = WildfireGraphBuilder(max_distance_km=25.0, wind_amplification_gamma=3.0)
    
    # Node 0 at (0, 0), Node 1 at (10, 0) -> due East of Node 0
    meta = {
        0: {"x_km": 0.0, "y_km": 0.0, "zone_id": 0},
        1: {"x_km": 10.0, "y_km": 0.0, "zone_id": 1}
    }
    
    # Wind at Node 0 blows due East (90 degrees meteorological) directly toward Node 1
    # Wind at Node 1 blows due North (0 degrees meteorological), orthogonal/opposing Node 0
    readings = {
        0: {"wind_direction": 90.0, "wind_speed": 20.0, "anomaly_score": 0.5},
        1: {"wind_direction": 0.0, "wind_speed": 5.0, "anomaly_score": 0.1}
    }
    
    graph = builder.build_graph(meta, readings, zone_targets={0: 1.0, 1: 0.0})
    
    # Check graph adjacency dimensions and asymmetry
    assert graph.x.shape == (2, 9)
    assert graph.edge_index.shape[0] == 2
    
    # Find weight for edge 0 -> 1 vs edge 1 -> 0
    edge_src = graph.edge_index[0].numpy()
    edge_dst = graph.edge_index[1].numpy()
    weights = graph.edge_attr.squeeze(-1).numpy()
    
    w_0_to_1 = weights[(edge_src == 0) & (edge_dst == 1)][0]
    w_1_to_0 = weights[(edge_src == 1) & (edge_dst == 0)][0]
    
    print(f"Directed edge weight 0->1 (with wind): {w_0_to_1} | 1->0 (against wind): {w_1_to_0}")
    assert w_0_to_1 > w_1_to_0 * 1.5, f"Wind-aligned edge {w_0_to_1} failed to significantly outweigh reverse edge {w_1_to_0}"

def test_gat_forward_and_belief_map():
    """Verify GAT architecture maps node graph to accurate per-zone belief map probabilities."""
    model = SpatioTemporalGAT(in_features=9, hidden_dim=16, num_zones=2)
    x = torch.randn(6, 9)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]], dtype=torch.long)
    edge_attr = torch.ones((6, 1))
    zone_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    
    probs = model(x, edge_index, edge_attr, zone_ids)
    assert probs.shape == (2,), f"Expected 2 zone predictions, got shape {probs.shape}"
    assert torch.all((probs >= 0.0) & (probs <= 1.0)), "Probs must be bound in [0, 1] via Sigmoid"

def test_graceful_degradation_under_30_percent_dropout(tmp_path):
    """CRITICAL TEST: Verify GNN degrades gracefully (no collapse) under 30% simulated node dropout."""
    out_dir = str(tmp_path / "gnn_dropout_logs")
    
    # Generate synthetic training & testing graph sequences
    train_graphs = []
    test_graphs = []
    
    for idx in range(30):
        # Create graphs with clear correlation: high anomaly & temp in Zone 1 -> target=1.0
        x = torch.zeros(6, 9)
        if idx % 2 == 0:
            # Zone 0 fire active
            x[0:3, 0] = 5.0  # Anomaly score
            x[0:3, 1] = 35.0 # Temp
            y = torch.tensor([1.0, 0.0])
        else:
            # Zone 1 fire active
            x[3:6, 0] = 5.0
            x[3:6, 1] = 38.0
            y = torch.tensor([0.0, 1.0])
            
        edge_index = torch.tensor([[0,0,1,1,2,2,3,3,4,4,5,5], [0,1,1,2,2,0,3,4,4,5,5,3]], dtype=torch.long)
        edge_attr = torch.ones((12, 1)) * 1.5
        zone_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        
        g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        g.zone_id = zone_ids
        
        if idx < 20:
            train_graphs.append(g)
        else:
            test_graphs.append(g)
            
    predictor = GNNFirePredictor(num_zones=2, in_features=9, hidden_dim=16, lr=0.02)
    losses = predictor.fit(train_graphs, epochs=20, augment_dropout=0.15)
    assert losses[-1] < losses[0], f"GNN loss failed to converge during training: {losses[0]} -> {losses[-1]}"
    
    df_resilience = predictor.evaluate_dropout_resilience(
        test_graphs,
        dropout_rates=[0.0, 0.3],
        num_trials=10,
        output_dir=out_dir
    )
    
    assert os.path.exists(os.path.join(out_dir, "gnn_dropout_resilience.json"))
    
    auc_0 = df_resilience[df_resilience["dropout_rate"] == 0.0]["mean_auc_roc"].values[0]
    auc_30 = df_resilience[df_resilience["dropout_rate"] == 0.3]["mean_auc_roc"].values[0]
    
    print(f"Baseline GNN AUC (0% dropout): {auc_0} | Graceful Degradation AUC (30% dropout): {auc_30}")
    
    # Verify model maintains robust predictive accuracy (>0.75 AUC) at 30% dropout without breaking
    assert auc_30 >= 0.75, f"Model collapsed under 30% dropout: AUC fell to {auc_30}"
    assert auc_30 >= auc_0 * 0.80, f"30% dropout AUC ({auc_30}) degraded by more than 20% vs baseline ({auc_0})"
