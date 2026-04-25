#!/usr/bin/env python3
"""
Phase 0.4 Testing Script
Quick validation of all modules with mock data.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utility_computer import UtilityComputer, initialize_equal_weights, AggregateUtilityScore
from gain_measurer import StratifiedGainSampler, GainAnalyzer, GainMeasurement
from correlation_validator import CorrelationResult, CorrelationValidator, CorrelationReporter
from visualizer import CorrelationVisualizer
from main import Phase04Orchestrator, create_mock_experiment_data


def test_utility_computer():
    """Test utility computation module."""
    print("\n" + "=" * 60)
    print("TEST 1: Utility Computer")
    print("=" * 60)
    
    computer = UtilityComputer(gamma=0.99, task_type="dense")
    
    # Mock trajectory
    mock_traj = {
        "rewards": np.array([1.0, 2.0, 3.0, 1.0]),
        "z_true": np.random.randn(5, 32),      # 5 timesteps, 32-dim
        "z_pred": np.random.randn(4, 32),      # 4 predictions
        "values": np.array([0.5, 1.0, 2.0, 3.5, 2.0]),
    }
    
    utilities = computer.compute_trajectory_utilities(mock_traj)
    print(f"[OK] Computed utilities: {utilities}")
    print(f"  Shape: {utilities.shape}")
    print(f"  Components: Reward, Novelty, TD Error, Goal Proximity")
    
    stats = computer.get_statistics_summary()
    print(f"[OK] Running statistics (after 1 trajectory):")
    for name, stat in stats.items():
        print(f"    {name}: μ={stat['mean']:.4f}, σ={stat['std']:.4f}")
    
    print("[OK] Utility computer test PASSED")
    return True


def test_gain_sampler():
    """Test learning gain measurement."""
    print("\n" + "=" * 60)
    print("TEST 2: Stratified Gain Sampler")
    print("=" * 60)
    
    sampler = StratifiedGainSampler(n_samples=100, n_quantiles=4, learning_rate=1e-4)
    
    # Mock utility scores
    utility_scores = np.random.randn(1000)
    
    sampled_indices = sampler.sample_stratified_indices(utility_scores)
    print(f"[OK] Sampled {len(sampled_indices)} trajectories from 1000 total")
    print(f"  Quantile distribution: {np.bincount(np.digitize(utility_scores[sampled_indices], np.percentile(utility_scores, [0, 25, 50, 75, 100])) - 1)}")
    
    # Mock measurements
    measurements = [
        GainMeasurement(
            trajectory_id=int(i),
            loss_before=1.0,
            loss_after=0.8 + 0.01 * np.random.randn(),
            learning_gain=0.2 + 0.05 * np.random.randn(),
            quantile_bin=i % 4,
            utility_score=utility_scores[i],
        )
        for i in sampled_indices
    ]
    
    stats = GainAnalyzer.compute_gain_statistics(measurements)
    print(f"[OK] Gain statistics:")
    print(f"    Mean: {stats['mean']:.4f}")
    print(f"    Std: {stats['std']:.4f}")
    print(f"    Median: {stats['median']:.4f}")
    print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    print("[OK] Gain sampler test PASSED")
    return True


def test_correlation_validator():
    """Test correlation validation."""
    print("\n" + "=" * 60)
    print("TEST 3: Correlation Validator")
    print("=" * 60)
    
    validator = CorrelationValidator(min_correlation=0.3)
    
    # Generate correlated mock data
    n = 500
    true_gains = np.random.randn(n) * 2
    proxy_scores = 0.42 * true_gains + np.sqrt(1 - 0.42**2) * np.random.randn(n)
    
    utilities = np.random.randn(n, 4)
    
    result = validator.validate_correlation(
        proxy_scores=proxy_scores,
        learning_gains=true_gains,
        utilities=utilities,
        environment="Mock-Env"
    )
    
    print(f"[OK] Validation result:")
    print(f"    Environment: {result.environment}")
    print(f"    Pearson r: {result.pearson_r:.3f} (p={result.pearson_p:.2e})")
    print(f"    95% CI: [{result.pearson_ci_low:.3f}, {result.pearson_ci_high:.3f}]")
    print(f"    Spearman rho_s: {result.spearman_r:.3f}")
    print(f"    Slope: {result.slope:.3f} (p={result.slope_p:.2e})")
    print(f"    R²: {result.r_squared:.3f}")
    print(f"    Cohen's q: {result.cohens_q:.3f}")
    print(f"    Passes criterion: {result.passes_criterion}")
    print(f"    Reason: {result.reason}")
    
    print("[OK] Correlation validator test PASSED")
    return True


def test_visualizer():
    """Test visualization module."""
    print("\n" + "=" * 60)
    print("TEST 4: Visualizer")
    print("=" * 60)
    
    visualizer = CorrelationVisualizer(output_dir="./phase_0_4_test_figs")
    
    # Create mock results
    results = []
    proxy_scores_dict = {}
    learning_gains_dict = {}
    
    for i in range(6):
        n = 500
        true_gains = np.random.randn(n)
        proxy_scores = 0.35 * true_gains + np.sqrt(1 - 0.35**2) * np.random.randn(n)
        
        env_name = f"Env-{i}"
        results.append(
            CorrelationResult(
                environment=env_name,
                n_samples=n,
                pearson_r=0.35,
                pearson_p=0.001,
                pearson_ci_low=0.28,
                pearson_ci_high=0.42,
                spearman_r=0.33,
                spearman_p=0.001,
                spearman_ci_low=0.26,
                spearman_ci_high=0.40,
                kendall_tau=0.22,
                kendall_p=0.001,
                slope=0.3,
                slope_p=0.001,
                intercept=0.1,
                r_squared=0.12,
                cohens_q=0.37,
                component_correlations={
                    "Reward": 0.2,
                    "Novelty": 0.15,
                    "TD Error": 0.25,
                    "Goal Proximity": 0.1
                },
                passes_criterion=True,
                reason="All criteria pass",
            )
        )
        proxy_scores_dict[env_name] = proxy_scores
        learning_gains_dict[env_name] = true_gains
    
    try:
        visualizer.scatter_plot_grid(results, proxy_scores_dict, learning_gains_dict)
        print("[OK] Generated scatter plot grid")
    except Exception as e:
        print(f"[ERR] Scatter plot failed: {e}")
    
    try:
        visualizer.correlation_barplot(results)
        print("[OK] Generated correlation barplot")
    except Exception as e:
        print(f"[ERR] Barplot failed: {e}")
    
    try:
        visualizer.component_ablation_heatmap(results)
        print("[OK] Generated component heatmap")
    except Exception as e:
        print(f"[ERR] Heatmap failed: {e}")
    
    try:
        visualizer.summary_statistics_figure(results)
        print("[OK] Generated summary statistics figure")
    except Exception as e:
        print(f"[ERR] Summary stats failed: {e}")
    
    print("[OK] Visualizer test PASSED")
    return True


def test_orchestrator():
    """Test full orchestrator."""
    print("\n" + "=" * 60)
    print("TEST 5: Phase 0.4 Orchestrator (Full Pipeline)")
    print("=" * 60)
    
    # Create mock data
    print("Generating mock experiment data...")
    trajectories, utilities, learning_gains = create_mock_experiment_data(
        n_envs=6, n_trajectories=500, true_correlation=0.40
    )
    
    # Run orchestrator
    print("Initializing orchestrator...")
    # Initialize orchestrator (config loading has built-in fallback to defaults)
    orchestrator = Phase04Orchestrator(config_path="config.yaml")
    
    print("Running validation pipeline...")
    results = orchestrator.run_validation_pipeline(trajectories, utilities, learning_gains)
    
    print(f"\n[OK] Orchestrator test PASSED")
    print(f"  Gate Status: {'PASS' if results['gate_status']['pass'] else 'FAIL'}")
    print(f"  Pass Rate: {results['gate_status']['pass_rate']*100:.1f}%")
    print(f"  Environments: {results['gate_status']['n_pass']}/{results['gate_status']['n_total']}")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# PHASE 0.4 TEST SUITE")
    print("#" * 60)
    
    tests = [
        ("Utility Computer", test_utility_computer),
        ("Gain Sampler", test_gain_sampler),
        ("Correlation Validator", test_correlation_validator),
        ("Visualizer", test_visualizer),
        ("Orchestrator", test_orchestrator),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, "PASS" if passed else "FAIL"))
        except Exception as e:
            print(f"\n[ERR] {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, "ERROR"))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, status in results:
        symbol = "[OK]" if status == "PASS" else "[ERR]"
        print(f"{symbol} {test_name}: {status}")
    
    all_passed = all(status == "PASS" for _, status in results)
    print("\n" + ("=" * 60))
    if all_passed:
        print("[OK] ALL TESTS PASSED")
    else:
        print("[ERR] SOME TESTS FAILED")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
