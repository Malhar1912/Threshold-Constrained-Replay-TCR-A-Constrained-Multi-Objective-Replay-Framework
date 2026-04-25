"""
Phase 0.4: Main Orchestrator
Coordinates the full correlation validation experiment.
"""

import yaml
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json

# Import submodules
from utility_computer import UtilityComputer, initialize_equal_weights
from gain_measurer import StratifiedGainSampler, GainAnalyzer
from correlation_validator import CorrelationValidator, CorrelationReporter
from visualizer import CorrelationVisualizer


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class Phase04Orchestrator:
    """Main orchestrator for Phase 0.4 validation experiment."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize orchestrator with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config['output']['base_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.utility_computer = UtilityComputer(
            gamma=self.config['agent']['gamma'],
            ema_decay=self.config['utility']['normalization']['ema_decay'],
            epsilon=self.config['utility']['normalization']['epsilon'],
        )
        
        self.gain_sampler = StratifiedGainSampler(
            n_samples=self.config['learning_gain']['n_samples'],
            n_quantiles=len(self.config['learning_gain']['stratified_quantiles']) - 1,
            learning_rate=self.config['learning_gain']['learning_rate'],
        )
        
        self.validator = CorrelationValidator(
            significance_level=self.config['correlation']['significance_level'],
            ci_level=self.config['correlation']['ci_level'],
            bootstrap_samples=self.config['correlation']['bootstrap_samples'],
            min_correlation=self.config['acceptance_criteria']['correlation_threshold'],
            spearman_tolerance=self.config['acceptance_criteria']['spearman_tolerance'],
        )
        
        self.visualizer = CorrelationVisualizer(
            output_dir=str(self.output_dir / "figures"),
            dpi=self.config['visualization']['dpi'],
        )
        
        self.results = []
        self.experiment_metadata = {
            'start_time': datetime.now().isoformat(),
            'config_path': config_path,
            'config': self.config,
        }
    
    @staticmethod
    def _get_default_config() -> Dict:
        """Return default configuration."""
        return {
            'output': {
                'base_dir': 'results',
                'save_dataframe_csv': True,
                'save_results_json': True,
                'save_summary_txt': True,
            },
            'agent': {'gamma': 0.99},
            'utility': {
                'normalization': {
                    'ema_decay': 0.99,
                    'epsilon': 1e-8,
                }
            },
            'learning_gain': {
                'n_samples': 500,
                'stratified_quantiles': [0, 0.25, 0.5, 0.75, 1.0],
                'learning_rate': 0.001,
            },
            'correlation': {
                'significance_level': 0.05,
                'ci_level': 0.95,
                'bootstrap_samples': 10000,
            },
            'acceptance_criteria': {
                'correlation_threshold': 0.3,
                'spearman_tolerance': 0.05,
                'min_environments': 6,
                'min_env_pass_rate': 1.0,  # All environments must pass
            },
            'visualization': {
                'dpi': 300,
                'figures': {
                    'scatter_plot': {'filename': '01_proxy_vs_gain_scatter.png'},
                    'correlation_barplot': {'filename': '02_correlation_barplot.png'},
                    'component_heatmap': {'filename': '03_component_ablation_heatmap.png'},
                }
            },
            'weight_vector': {
                'strategy': 'equal_weighting',
                'values': [0.5, 0.5, 0.5, 0.5],
            }
        }
    
    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """Load configuration from YAML file with fallback to defaults."""
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file {config_path} not found; using default configuration")
            return Phase04Orchestrator._get_default_config()
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}; using defaults")
            return Phase04Orchestrator._get_default_config()
    
    def run_validation_pipeline(self, 
                               trajectories_per_env: Dict[str, List],
                               utilities_per_env: Dict[str, np.ndarray],
                               learning_gains_per_env: Dict[str, np.ndarray]) -> Dict:
        """
        Run full validation pipeline for Phase 0.4.
        
        Args:
            trajectories_per_env: Dict mapping env name to trajectory list
            utilities_per_env: Dict mapping env name to utility matrices (N, 4)
            learning_gains_per_env: Dict mapping env name to learning gain vectors (N,)
            
        Returns:
            Aggregated validation results
        """
        logger.info("=" * 60)
        logger.info("PHASE 0.4: PROXY APPROXIMATION VALIDITY GATE")
        logger.info("=" * 60)
        
        # Get weight vector
        weights = self._initialize_weights(utilities_per_env)
        logger.info(f"Weight vector: {weights}")
        
        proxy_scores_dict = {}
        self.results = []
        
        # Validate each environment
        for env_name in utilities_per_env.keys():
            logger.info(f"\nValidating environment: {env_name}")
            
            utilities = utilities_per_env[env_name]
            learning_gains = learning_gains_per_env[env_name]
            
            # Compute proxy scores
            proxy_scores = utilities @ weights
            proxy_scores_dict[env_name] = proxy_scores
            
            # Run correlation validation
            result = self.validator.validate_correlation(
                proxy_scores=proxy_scores,
                learning_gains=learning_gains,
                utilities=utilities,
                environment=env_name,
            )
            
            self.results.append(result)
            logger.info(self.validator.format_result(result))
        
        # Aggregate results
        aggregate = CorrelationReporter.aggregate_results(self.results)
        CorrelationReporter.print_summary(aggregate)
        
        # Generate visualizations
        learning_gains_dict = {env: learning_gains_per_env[env] 
                              for env in utilities_per_env.keys()}
        
        self._generate_visualizations(proxy_scores_dict, learning_gains_dict)
        
        # Save results
        self._save_results(aggregate, weights)
        
        # Determine overall pass/fail
        pass_rate = aggregate['pass_rate']
        min_pass_rate = self.config['acceptance_criteria']['min_env_pass_rate']
        
        gate_status = {
            'pass': pass_rate >= min_pass_rate,
            'pass_rate': pass_rate,
            'min_required': min_pass_rate,
            'n_pass': aggregate['n_pass'],
            'n_total': aggregate['n_environments'],
        }
        
        logger.info("\n" + "=" * 60)
        if gate_status['pass']:
            logger.info("✓ GATE PASSED: Proceed to Phases 1-7")
        else:
            logger.info("✗ GATE FAILED: Re-engineer utility components and retry")
        logger.info("=" * 60 + "\n")
        
        return {
            'aggregate': aggregate,
            'gate_status': gate_status,
            'results': self.results,
            'weights': weights,
            'proxy_scores': proxy_scores_dict,
        }
    
    def _initialize_weights(self, utilities_per_env: Dict[str, np.ndarray]) -> np.ndarray:
        """Initialize weight vector."""
        strategy = self.config['weight_vector']['strategy']
        
        # Determine dimensionality
        first_utilities = next(iter(utilities_per_env.values()))
        d = first_utilities.shape[1] if len(first_utilities.shape) > 1 else 1
        
        if strategy == "equal_weighting":
            return initialize_equal_weights(d)
        elif strategy == "regression_optimized":
            # Would require fitting on pooled data
            logger.warning("Regression-optimized weights not yet implemented; using equal weights")
            return initialize_equal_weights(d)
        else:
            raise ValueError(f"Unknown weight strategy: {strategy}")
    
    def _generate_visualizations(self, 
                                proxy_scores_dict: Dict[str, np.ndarray],
                                learning_gains_dict: Dict[str, np.ndarray]):
        """Generate all visualization figures."""
        logger.info("\nGenerating visualizations...")
        
        try:
            self.visualizer.scatter_plot_grid(
                self.results,
                proxy_scores_dict,
                learning_gains_dict,
                filename=self.config['visualization']['figures']['scatter_plot']['filename'],
            )
        except Exception as e:
            logger.warning(f"Failed to generate scatter plot grid: {e}")
        
        try:
            self.visualizer.correlation_barplot(
                self.results,
                threshold=self.config['acceptance_criteria']['correlation_threshold'],
                filename=self.config['visualization']['figures']['correlation_barplot']['filename'],
            )
        except Exception as e:
            logger.warning(f"Failed to generate correlation barplot: {e}")
        
        try:
            self.visualizer.component_ablation_heatmap(
                self.results,
                filename=self.config['visualization']['figures']['component_heatmap']['filename'],
            )
        except Exception as e:
            logger.warning(f"Failed to generate component heatmap: {e}")
        
        try:
            self.visualizer.summary_statistics_figure(
                self.results,
                filename="04_summary_statistics.png",
            )
        except Exception as e:
            logger.warning(f"Failed to generate summary statistics figure: {e}")
        
        logger.info(f"Visualizations saved to {self.visualizer.output_dir}")
    
    def _save_results(self, aggregate: Dict, weights: np.ndarray):
        """Save results to CSV, JSON, and text formats."""
        # CSV results
        if self.config['output']['save_dataframe_csv']:
            csv_path = self.output_dir / "results.csv"
            try:
                import pandas as pd
                data = []
                for result in self.results:
                    data.append({
                        'environment': result.environment,
                        'n_samples': result.n_samples,
                        'pearson_r': result.pearson_r,
                        'pearson_p': result.pearson_p,
                        'pearson_ci_low': result.pearson_ci_low,
                        'pearson_ci_high': result.pearson_ci_high,
                        'spearman_r': result.spearman_r,
                        'kendall_tau': result.kendall_tau,
                        'slope': result.slope,
                        'r_squared': result.r_squared,
                        'cohens_q': result.cohens_q,
                        'passes_criterion': result.passes_criterion,
                    })
                df = pd.DataFrame(data)
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved results to {csv_path}")
            except Exception as e:
                logger.warning(f"Failed to save CSV: {e}")
        
        # JSON summary
        if self.config['output']['save_results_json']:
            json_path = self.output_dir / "results.json"
            try:
                summary = {
                    'experiment': self.experiment_metadata,
                    'aggregate_results': {
                        'n_environments': aggregate['n_environments'],
                        'n_pass': aggregate['n_pass'],
                        'pass_rate': aggregate['pass_rate'],
                        'mean_pearson_r': float(aggregate['mean_pearson_r']),
                        'std_pearson_r': float(aggregate['std_pearson_r']),
                        'median_pearson_r': float(aggregate['median_pearson_r']),
                    },
                    'weight_vector': weights.tolist(),
                }
                with open(json_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                logger.info(f"Saved JSON summary to {json_path}")
            except Exception as e:
                logger.warning(f"Failed to save JSON: {e}")
        
        # Text summary
        if self.config['output']['save_summary_txt']:
            txt_path = self.output_dir / "summary.txt"
            try:
                with open(txt_path, 'w') as f:
                    f.write("=" * 70 + "\n")
                    f.write("PHASE 0.4: PROXY APPROXIMATION VALIDITY GATE - RESULTS\n")
                    f.write("=" * 70 + "\n\n")
                    
                    f.write(f"Experiment: {self.experiment_metadata['start_time']}\n")
                    f.write(f"Environments: {aggregate['n_environments']}\n")
                    f.write(f"Pass rate: {aggregate['pass_rate']*100:.1f}% ({aggregate['n_pass']}/{aggregate['n_environments']})\n")
                    f.write(f"Mean Pearson r: {aggregate['mean_pearson_r']:.3f} ± {aggregate['std_pearson_r']:.3f}\n")
                    f.write(f"Median Pearson r: {aggregate['median_pearson_r']:.3f}\n\n")
                    
                    f.write("Per-Environment Results:\n")
                    f.write("-" * 70 + "\n")
                    for result in self.results:
                        f.write(self.validator.format_result(result) + "\n\n")
                
                logger.info(f"Saved text summary to {txt_path}")
            except Exception as e:
                logger.warning(f"Failed to save text summary: {e}")


def create_mock_experiment_data(n_envs: int = 8,
                               n_trajectories: int = 500,
                               true_correlation: float = 0.42) -> tuple:
    """
    Create mock data for testing Phase 0.4 orchestrator.
    
    CRITICAL: Uses EQUAL WEIGHTS to match orchestrator's default initialization.
    This ensures the mock correlations actually validate properly.
    
    Args:
        n_envs: Number of environments
        n_trajectories: Trajectories per environment
        true_correlation: True correlation between proxy and gains (default 0.42)
        
    Returns:
        (trajectories, utilities, learning_gains) dicts
    """
    env_names = ["Pong", "Breakout", "Seaquest", "SpaceInvaders", 
                 "walker-walk", "cheetah-run", "finger-spin", "reacher-hard"][:n_envs]
    
    trajectories_dict = {}
    utilities_dict = {}
    learning_gains_dict = {}
    
    # CRITICAL: Use EQUAL WEIGHTS (matching orchestrator default)
    # Equal weight for 4 components: w = [0.5, 0.5, 0.5, 0.5] normalized
    equal_weights = np.array([0.5, 0.5, 0.5, 0.5])
    
    for env_name in env_names:
        # Generate realistic utility vectors (mean 0, std 1 after normalization)
        utilities = np.random.randn(n_trajectories, 4)
        
        # Compute proxy scores using EQUAL weights (this is what orchestrator will use)
        proxy_scores = utilities @ equal_weights
        
        # Generate learning gains correlated with proxy
        # ΔL = r * proxy + sqrt(1 - r²) * noise
        noise = np.sqrt(1.0 - true_correlation**2) * np.random.randn(n_trajectories)
        learning_gains = true_correlation * proxy_scores + noise
        
        trajectories_dict[env_name] = [{'id': i} for i in range(n_trajectories)]
        utilities_dict[env_name] = utilities.astype(np.float32)
        learning_gains_dict[env_name] = learning_gains.astype(np.float32)
    
    return trajectories_dict, utilities_dict, learning_gains_dict


if __name__ == "__main__":
    logger.info("Phase 0.4: Proxy Approximation Validity Experiment")
    
    # Create mock data
    logger.info("Creating mock experiment data...")
    trajectories, utilities, learning_gains = create_mock_experiment_data(
        n_envs=8, n_trajectories=500, true_correlation=0.42
    )
    
    # Run orchestrator
    orchestrator = Phase04Orchestrator(config_path="config.yaml")
    results = orchestrator.run_validation_pipeline(trajectories, utilities, learning_gains)
    
    logger.info("✓ Phase 0.4 experiment completed successfully")
    print(f"\nGate Status: {'PASS' if results['gate_status']['pass'] else 'FAIL'}")
