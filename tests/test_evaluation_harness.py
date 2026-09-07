import os
import pytest
import pandas as pd
from src.eval.run_benchmarks import BenchmarkHarness

def test_evaluation_harness_and_figure_generation(tmp_path):
    """CRITICAL TEST: Verify evaluation script generates all 6 plots and 2 tables without manual intervention."""
    rep_dir = str(tmp_path / "reports")
    log_dir = str(tmp_path / "logs")
    
    harness = BenchmarkHarness(output_dir=rep_dir, logs_dir=log_dir)
    
    # Run Table 1 & Table 2 generation
    df_t1 = harness.run_system_benchmarks()
    df_t2 = harness.run_ablation_study()
    
    assert os.path.exists(os.path.join(rep_dir, "table_1_system_benchmarks.csv"))
    assert os.path.exists(os.path.join(rep_dir, "table_2_ablation_study.csv"))
    assert len(df_t1) == 5, f"Expected 5 benchmark systems in Table 1, got {len(df_t1)}"
    assert len(df_t2) == 4, f"Expected 4 ablation configurations in Table 2, got {len(df_t2)}"
    
    # Run Figure Generation (6 plots)
    harness.generate_research_plots()
    
    expected_figures = [
        "fig_fl_divergence.png",
        "fig_privacy_utility.png",
        "fig_detection_delay.png",
        "fig_gnn_burnout_resilience.png",
        "fig_rl_timeline.png",
        "fig_shap_comparison.png"
    ]
    for fig_name in expected_figures:
        fig_path = os.path.join(rep_dir, fig_name)
        assert os.path.exists(fig_path), f"Mandatory research figure {fig_name} was not generated in reports/!"
        assert os.path.getsize(fig_path) > 1000, f"Generated image {fig_name} appears empty or unrendered!"

    # Verify Markdown report compilation
    report_text = harness.generate_final_report_markdown(df_t1, df_t2)
    assert os.path.exists(os.path.join(rep_dir, "final_research_report.md"))
    assert "Non-IID Zone-Clustered Federated Averaging" in report_text
    assert "PyTorch Native Int8 Quantization vs. TensorFlow Lite" in report_text
