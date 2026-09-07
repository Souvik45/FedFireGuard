import os
import shutil
import pytest
import numpy as np
import torch
import pandas as pd
from src.data.generator import WildfireDataGenerator
from src.edge.client import EdgeNodeClient
from src.federated.centralized import CentralizedLSTMBaseline
from src.federated.strategies import StandardFedAvgStrategy, ClusteredFedAvgStrategy
from src.federated.server import FLSimulationServer
from src.federated.metrics import compute_cosine_divergence, FLDivergenceLogger

def test_centralized_baseline(tmp_path):
    """Verify Baseline 1 pools data across zones, trains model, and tracks raw data bandwidth."""
    gen = WildfireDataGenerator(num_nodes=3, num_zones=2, time_steps=40, seed=42)
    dfs = gen.generate_all_nodes()
    
    baseline = CentralizedLSTMBaseline(input_dim=7, hidden_dim=10, seq_len=5)
    baseline.load_and_pool_data(dfs)
    
    assert len(baseline.pooled_data_tensor) > 0
    loss = baseline.train_pooled_model(epochs=2, batch_size=16)
    assert loss > 0.0
    
    scores = baseline.evaluate_node_anomaly_scores()
    assert len(scores) == 3
    
    bw_kb = baseline.get_bandwidth_overhead_kb()
    assert bw_kb > 0.0, "Centralized raw baseline must log positive communication bandwidth consumption!"

def test_fedavg_strategy_math():
    """Verify Standard FedAvg computes sample-weighted consensus identically for all nodes."""
    strategy = StandardFedAvgStrategy()
    g1 = {"weight": torch.tensor([[2.0, 4.0]])}
    g2 = {"weight": torch.tensor([[10.0, 20.0]])}
    
    # Client 1 has 1 sample, Client 2 has 3 samples -> weights 0.25 and 0.75
    # Expect 0.25 * [2,4] + 0.75 * [10,20] = [0.5 + 7.5, 1.0 + 15.0] = [8.0, 16.0]
    result = strategy.aggregate_gradients([g1, g2], [10, 30])
    assert len(result) == 2
    assert torch.allclose(result[0]["weight"], torch.tensor([[8.0, 16.0]]))
    assert torch.allclose(result[1]["weight"], torch.tensor([[8.0, 16.0]]))

def test_clustered_fedavg_strategy_math():
    """Verify Clustered FedAvg assigns higher weight to peer gradients with aligned directions."""
    strategy = ClusteredFedAvgStrategy(temperature=10.0)
    # Node 0 and Node 1 point in identical directions (+X); Node 2 points in opposite direction (-X)
    g0 = {"w": torch.tensor([1.0, 0.0])}
    g1 = {"w": torch.tensor([2.0, 0.0])}
    g2 = {"w": torch.tensor([-10.0, 0.0])}
    
    results = strategy.aggregate_gradients([g0, g1, g2], [10, 10, 10])
    
    # Due to high cosine similarity between g0 and g1 (+1.0) vs g0 and g2 (-1.0),
    # Node 0's aggregated update should be dominated by g0 and g1 (positive X value)
    assert results[0]["w"][0] > 0.0, f"Clustered aggregation should suppress opposing gradient! Got {results[0]['w']}"
    # Conversely, Node 2 should isolate mostly to its own negative direction
    assert results[2]["w"][0] < 0.0

def test_gradient_divergence_comparison(tmp_path):
    """CRITICAL TEST: Verify non-IID gradient divergence is logged and visibly worse for standard FedAvg than clustered FedAvg."""
    log_dir = str(tmp_path / "fl_test_logs")
    
    # Generate non-IID microclimate data (4 nodes across 2 severe contrast microclimate zones)
    gen = WildfireDataGenerator(num_nodes=4, num_zones=2, time_steps=60, seed=123)
    dfs = gen.generate_all_nodes()
    
    # Setup clients for Standard FedAvg
    clients_std = [EdgeNodeClient(node_id=i, zone_id=dfs[i]["zone_id"].iloc[0], seq_len=5) for i in range(4)]
    for i, client in enumerate(clients_std):
        client.load_dataframe(dfs[i])
        
    server_std = FLSimulationServer(clients=clients_std, strategy_type="standard", logger_dir=os.path.join(log_dir, "std"))
    summary_std = server_std.train_federated_loop(total_rounds=3, local_epochs=1)
    std_divergences = [round_meta["mean_divergence"] for round_meta in summary_std]
    
    # Setup clients for Clustered FedAvg starting from identical seed weights
    clients_cluster = [EdgeNodeClient(node_id=i, zone_id=dfs[i]["zone_id"].iloc[0], seq_len=5) for i in range(4)]
    for i, client in enumerate(clients_cluster):
        client.load_dataframe(dfs[i])
        # Force identical initial weights as std clients for fair scientific test
        client.model.load_state_dict(clients_std[0]._initial_state_dict)
        
    server_cluster = FLSimulationServer(clients=clients_cluster, strategy_type="clustered", logger_dir=os.path.join(log_dir, "cluster"), temperature=5.0)
    summary_cluster = server_cluster.train_federated_loop(total_rounds=3, local_epochs=1)
    cluster_divergences = [round_meta["mean_divergence"] for round_meta in summary_cluster]
    
    # ASSERT CORE RESEARCH PROMPT REQUIREMENT:
    # Gradient divergence must be logged and lower under Clustered FedAvg than under Standard FedAvg
    assert os.path.exists(os.path.join(log_dir, "std", "fl_divergence_history.json"))
    assert os.path.exists(os.path.join(log_dir, "cluster", "fl_divergence_history.json"))
    
    mean_std_div = np.mean(std_divergences)
    mean_cluster_div = np.mean(cluster_divergences)
    print(f"Mean Standard FedAvg Divergence: {mean_std_div} | Mean Clustered Divergence: {mean_cluster_div}")
    
    assert mean_cluster_div < mean_std_div, (
        f"Clustered FedAvg divergence ({mean_cluster_div}) failed to improve over Standard FedAvg ({mean_std_div}) on non-IID zones!"
    )
