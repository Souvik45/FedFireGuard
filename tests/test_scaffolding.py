import os
import yaml
import pytest
import src

def test_src_imports():
    """Verify that core packages can be imported successfully."""
    import src.data
    import src.edge
    import src.federated
    import src.privacy
    import src.gnn
    import src.rl
    import src.baselines
    import src.explain
    import src.dashboard
    import src.eval
    assert src.__version__ == "0.1.0"

def test_base_config():
    """Verify that base.yaml is readable and has required fields."""
    config_path = os.path.join(os.path.dirname(__file__), "../configs/base.yaml")
    assert os.path.exists(config_path), f"Config file not found at {config_path}"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    assert config["seed"] == 42
    assert config["simulation"]["num_nodes"] == 8
    assert config["simulation"]["num_zones"] == 2
    assert len(config["simulation"]["features"]) == 7
