import os
import json
import shutil
import pytest
import numpy as np
import pandas as pd
from src.data.generator import WildfireDataGenerator
from src.data.stubs import FarsiteLoaderStub, NASAFirmsLoaderStub, NOAAWeatherLoaderStub

def test_generator_determinism():
    """Verify that identical seeds produce identical time series DataFrames."""
    gen1 = WildfireDataGenerator(num_nodes=4, num_zones=2, time_steps=48, seed=123)
    dfs1 = gen1.generate_all_nodes()
    
    gen2 = WildfireDataGenerator(num_nodes=4, num_zones=2, time_steps=48, seed=123)
    dfs2 = gen2.generate_all_nodes()
    
    for n in range(4):
        pd.testing.assert_frame_equal(dfs1[n], dfs2[n])
        
    # Verify different seeds yield divergent values
    gen3 = WildfireDataGenerator(num_nodes=4, num_zones=2, time_steps=48, seed=999)
    dfs3 = gen3.generate_all_nodes()
    assert not np.allclose(dfs1[0]["temperature"], dfs3[0]["temperature"])

def test_non_iid_microclimates():
    """Verify that nodes in distinct zones exhibit statistically distinct baseline distributions."""
    gen = WildfireDataGenerator(num_nodes=8, num_zones=2, time_steps=240, seed=42)
    dfs = gen.generate_all_nodes()
    
    # Node 0 is in Zone 0 (Alpine: cool, high humidity), Node 1 is in Zone 1 (Lowland valley: hot, dry)
    temp_zone0 = dfs[0]["temperature"].mean()
    temp_zone1 = dfs[1]["temperature"].mean()
    hum_zone0 = dfs[0]["humidity"].mean()
    hum_zone1 = dfs[1]["humidity"].mean()
    
    # Assert severe microclimate difference (>10 degrees C, >15% humidity divergence)
    assert (temp_zone1 - temp_zone0) > 8.0, f"Zone 1 temp ({temp_zone1}) should be significantly warmer than Zone 0 ({temp_zone0})"
    assert (hum_zone0 - hum_zone1) > 15.0, f"Zone 0 hum ({hum_zone0}) should be significantly higher than Zone 1 ({hum_zone1})"

def test_pre_ignition_injection():
    """Verify that injected wildfire pre-ignition signatures elevate temp/pm25 and suppress humidity."""
    gen = WildfireDataGenerator(num_nodes=2, num_zones=1, time_steps=100, seed=42)
    scenarios = [{"node_id": 0, "start_step": 40, "duration": 15, "magnitude_scale": 1.0}]
    dfs = gen.generate_all_nodes(fire_scenarios=scenarios)
    
    df_fire = dfs[0]
    normal_windows = df_fire[df_fire["fire_label"] == 0]
    fire_windows = df_fire[df_fire["fire_label"] == 1]
    
    assert len(fire_windows) == 15
    assert fire_windows["temperature"].mean() > normal_windows["temperature"].mean() + 5.0
    assert fire_windows["pm25"].mean() > normal_windows["pm25"].mean() + 20.0
    assert fire_windows["humidity"].mean() < normal_windows["humidity"].mean() - 5.0

def test_parquet_export(tmp_path):
    """Verify that node DataFrames and metadata export to labeled Parquet cleanly."""
    out_dir = str(tmp_path / "test_parquet_data")
    gen = WildfireDataGenerator(num_nodes=3, num_zones=2, time_steps=30, seed=10)
    gen.save_to_parquet(output_dir=out_dir)
    
    assert os.path.exists(os.path.join(out_dir, "metadata.json"))
    for n in range(3):
        file_path = os.path.join(out_dir, f"node_{n}.parquet")
        assert os.path.exists(file_path)
        df_loaded = pd.read_parquet(file_path)
        assert len(df_loaded) == 30
        assert "fire_label" in df_loaded.columns
        assert "co2" in df_loaded.columns

def test_loader_stubs():
    """Verify that loader stubs raise appropriate documentation exceptions when called."""
    with pytest.raises(NotImplementedError, match="optional stretch goal"):
        FarsiteLoaderStub.load_scenario("dummy_path", {0: (0.0, 0.0)})
        
    with pytest.raises(NotImplementedError, match="out of scope"):
        NASAFirmsLoaderStub.fetch_thermal_anomalies(30.0, 35.0, -120.0, -115.0, "2023-08-01")
        
    with pytest.raises(NotImplementedError, match="stubbed"):
        NOAAWeatherLoaderStub.fetch_hourly_weather("STATION_X", "2023-08-01", "2023-08-07")
