import pytest
from src.baselines.threshold_rules import ThresholdRuleSystem

def test_threshold_rule_system_baseline():
    """Verify Baseline 3 triggers evacuations on raw pre-ignition spikes and remains clear on baselines."""
    rules = ThresholdRuleSystem(temp_evac=35.0, hum_evac=25.0, pm_evac=50.0)
    
    # Severe fire signature readings
    fire_readings = {"temperature": 38.0, "humidity": 15.0, "pm25": 80.0}
    assert rules.evaluate_node_step(fire_readings) == 3
    
    # Mild warning readings
    warn_readings = {"temperature": 33.0, "humidity": 30.0, "pm25": 12.0}
    assert rules.evaluate_node_step(warn_readings) >= 1
    
    # Clear alpine baseline
    clear_readings = {"temperature": 16.0, "humidity": 65.0, "pm25": 8.0}
    assert rules.evaluate_node_step(clear_readings) == 0

def test_zone_level_aggregation_rules():
    """Verify zone directives select maximum sensor severity within a geographic cluster."""
    rules = ThresholdRuleSystem()
    zone_data = {
        0: [
            {"temperature": 18.0, "humidity": 70.0, "pm25": 5.0},
            {"temperature": 38.0, "humidity": 15.0, "pm25": 85.0}  # One sensor detected severe flame front
        ],
        1: [
            {"temperature": 20.0, "humidity": 50.0, "pm25": 10.0},
            {"temperature": 21.0, "humidity": 48.0, "pm25": 11.0}   # All clear
        ]
    }
    
    directives = rules.evaluate_zone_directives(zone_data)
    assert directives[0]["alert_level"] == 3
    assert directives[0]["resource_dispatch"] == 2
    
    assert directives[1]["alert_level"] == 0
    assert directives[1]["resource_dispatch"] == 0
