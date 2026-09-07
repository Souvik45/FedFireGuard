import torch
import pytest
import numpy as np
import pandas as pd
from src.edge.models import LSTMAutoencoder, compute_reconstruction_errors
from src.edge.quantize import quantize_int8, benchmark_model_size_kb, benchmark_latency
from src.edge.client import EdgeNodeClient
from src.data.generator import WildfireDataGenerator

def test_lstm_autoencoder_shapes():
    """Verify forward pass reconstruction shapes match input tensors."""
    model = LSTMAutoencoder(input_dim=7, hidden_dim=12, num_layers=1)
    batch = torch.randn(8, 6, 7)
    recon = model(batch)
    assert recon.shape == (8, 6, 7), f"Expected reconstruction shape (8,6,7) got {recon.shape}"

def test_anomaly_score_on_fire_injection():
    """Verify untrained/normal-trained autoencoder yields higher anomaly score on fire signature."""
    gen = WildfireDataGenerator(num_nodes=1, num_zones=1, time_steps=60, seed=42)
    scen = [{"node_id": 0, "start_step": 25, "duration": 15, "magnitude_scale": 2.0}]
    dfs = gen.generate_all_nodes(fire_scenarios=scen)
    
    client = EdgeNodeClient(node_id=0, zone_id=0, seq_len=6, lr=0.01)
    client.load_dataframe(dfs[0])
    
    # Train locally on normal data
    client.synchronize_with_server(client.model.state_dict())
    client.train_local_epoch(epochs=3, batch_size=8, only_normal_data=True)
    
    scores = client.compute_anomaly_scores()
    labels = client.local_labels
    
    normal_scores = scores[labels == 0]
    fire_scores = scores[labels == 1]
    
    assert len(fire_scores) > 0
    # Average anomaly score during fire injection should exceed normal baseline score
    assert np.mean(fire_scores) > np.mean(normal_scores), f"Fire anomaly score {np.mean(fire_scores)} not higher than normal {np.mean(normal_scores)}"

def test_quantization_and_benchmarking():
    """Verify int8 post-training quantization reduces model size and latency benchmarking runs cleanly."""
    model = LSTMAutoencoder(input_dim=7, hidden_dim=16)
    size_orig_kb = benchmark_model_size_kb(model)
    
    q_model = quantize_int8(model)
    size_quant_kb = benchmark_model_size_kb(q_model)
    
    # Check that int8 quantized checkpoint is smaller or comparable (for tiny models metadata weight overhead applies)
    assert size_quant_kb <= size_orig_kb
    
    bench = benchmark_latency(q_model, input_dim=7, seq_len=6, num_runs=20)
    assert "mean_latency_ms" in bench
    assert "p95_latency_ms" in bench
    assert bench["mean_latency_ms"] > 0.0

def test_gradient_extraction_interface():
    """Verify that extracted gradients contain only parameter weights and no raw data."""
    gen = WildfireDataGenerator(num_nodes=1, num_zones=1, time_steps=30, seed=42)
    dfs = gen.generate_all_nodes()
    
    client = EdgeNodeClient(node_id=0, zone_id=0, seq_len=5)
    client.load_dataframe(dfs[0])
    
    init_weights = copy_weights(client.model.state_dict())
    client.synchronize_with_server(init_weights)
    
    loss = client.train_local_epoch(epochs=1, batch_size=10)
    assert loss >= 0.0
    
    grads = client.extract_gradients()
    assert isinstance(grads, dict)
    assert "encoder_lstm.weight_ih_l0" in grads
    
    # Ensure gradients are non-zero after training step
    norm = torch.norm(grads["encoder_lstm.weight_ih_l0"])
    assert norm > 0.0
    
    # Guarantee no raw dataframe / time series values leaked in output dictionary
    for k, v in grads.items():
        assert isinstance(v, torch.Tensor), f"Extracted item {k} is not a weight tensor!"
        assert "temperature" not in k and "pm25" not in k

def copy_weights(state_dict):
    import copy
    return copy.deepcopy(state_dict)
