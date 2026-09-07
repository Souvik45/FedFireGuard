import pytest
from src.dashboard.app import load_simulated_dashboard_data

def test_streamlit_dashboard_data_loading_and_imports():
    """CRITICAL TEST: Verify Streamlit dashboard imports cleanly without missing dependencies or syntax errors."""
    data = load_simulated_dashboard_data(seed=42)
    assert "node_dfs" in data
    assert "metadata" in data
    assert "scores" in data
    assert "explanations" in data
    assert len(data["metadata"]) == 6
    assert 0 in data["explanations"]
