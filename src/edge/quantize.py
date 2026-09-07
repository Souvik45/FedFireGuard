"""Post-training int8 quantization and TinyML device simulation benchmarks."""
import os
import time
import tempfile
import torch
import torch.nn as nn
from typing import Dict, Any

def quantize_int8(model: nn.Module) -> nn.Module:
    """Apply post-training dynamic int8 quantization to linear and LSTM weights.
    
    Note on framework choice: Utilizes native PyTorch int8 dynamic quantization
    (`torch.quantization.quantize_dynamic`) rather than TensorFlow Lite export.
    This provides the identical mathematical uint8/int8 memory reduction on LSTM
    weights without requiring heavy multi-GB TensorFlow/ONNX dependencies or running
    into control-flow export conversion errors on Windows.
    """
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.LSTM, nn.Linear},
        dtype=torch.qint8
    )
    return quantized_model

def benchmark_model_size_kb(model: nn.Module) -> float:
    """Measure serialized model checkpoint size on disk in Kilobytes (KB)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp_path = tmp.name
        
    try:
        torch.save(model.state_dict(), tmp_path)
        size_bytes = os.path.getsize(tmp_path)
        size_kb = size_bytes / 1024.0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    return float(size_kb)

def benchmark_latency(
    model: nn.Module,
    input_dim: int = 7,
    seq_len: int = 6,
    num_warmups: int = 10,
    num_runs: int = 100
) -> Dict[str, Any]:
    """Benchmark inference latency on host CPU as a stand-in for edge microcontrollers.
    
    Args:
        model: PyTorch model (standard or quantized int8).
        input_dim: Number of sensor channels.
        seq_len: Temporal window sequence length.
        num_warmups: Warmup inferences to prime cache.
        num_runs: Measured test iterations.
        
    Returns:
        Dictionary containing mean latency (ms), 95th percentile (ms), and size (KB).
    """
    model.eval()
    dummy_input = torch.randn(1, seq_len, input_dim)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(dummy_input)
            
    # Measured execution loop
    latencies_ms = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_t = time.perf_counter()
            _ = model(dummy_input)
            end_t = time.perf_counter()
            latencies_ms.append((end_t - start_t) * 1000.0)
            
    latencies_ms.sort()
    mean_ms = sum(latencies_ms) / len(latencies_ms)
    p95_ms = latencies_ms[int(len(latencies_ms) * 0.95)]
    size_kb = benchmark_model_size_kb(model)
    
    return {
        "mean_latency_ms": round(mean_ms, 3),
        "p95_latency_ms": round(p95_ms, 3),
        "model_size_kb": round(size_kb, 2),
        "target_device": "host_cpu_simulating_mcu"
    }
