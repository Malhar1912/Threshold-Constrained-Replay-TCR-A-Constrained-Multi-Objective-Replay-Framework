"""
Visualization Module
Generate publication-quality figures for correlation analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import logging
from pathlib import Path

from correlation_validator import CorrelationResult

logger = logging.getLogger(__name__)


class CorrelationVisualizer:
    """Generate figures for correlation validation results."""
    
    def __init__(self,
                 output_dir: str = "./results/figures",
                 dpi: int = 300,
                 figsize_base: tuple = (4, 3)):
        """
        Args:
            output_dir: Directory to save figures
            dpi: Resolution in dots per inch
            figsize_base: Base figure size (width, height)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize_base = figsize_base
        
        # Set style
        sns.set_style("darkgrid")
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 11
        plt.rcParams['legend.fontsize'] = 9
    
    def scatter_plot_grid(self,
                         results: List[CorrelationResult],
                         proxy_scores_dict: Dict[str, np.ndarray],
                         learning_gains_dict: Dict[str, np.ndarray],
                         filename: str = "01_proxy_vs_gain_scatter.png"):
        """
        Generate 2×3 grid of scatter plots (proxy scores vs learning gains).
        
        Args:
            results: List of CorrelationResult objects
            proxy_scores_dict: Dict mapping environment name to proxy scores
            learning_gains_dict: Dict mapping environment name to learning gains
            filename: Output filename
        """
        n_envs = len(results)
        n_cols = 3
        n_rows = (n_envs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten() if n_envs > 1 else [axes]
        
        for idx, result in enumerate(results):
            ax = axes[idx]
            env_name = result.environment
            
            x = proxy_scores_dict.get(env_name, np.array([]))
            y = learning_gains_dict.get(env_name, np.array([]))
            
            if len(x) == 0 or len(y) == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                ax.set_title(f"{env_name} (no data)")
                continue
            
            # Scatter plot with density coloring
            scatter = ax.scatter(x, y, c=np.arange(len(x)), cmap='viridis',
                               alpha=0.6, s=40, edgecolors='none')
            
            # Fitted regression line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = p(x_line)
            ax.plot(x_line, y_line, 'r--', linewidth=2, label='Linear fit')
            
            # 95% confidence band (approximate)
            residuals = y - p(x)
            std_residuals = np.std(residuals)
            confidence_band = 1.96 * std_residuals
            ax.fill_between(x_line, y_line - confidence_band, y_line + confidence_band,
                           alpha=0.2, color='red', label='95% CI')
            
            # Labels and title
            status = "✓ PASS" if result.passes_criterion else "✗ FAIL"
            title = f"{env_name} (n={result.n_samples})\n{status}: ρ={result.pearson_r:.3f}, p={result.pearson_p:.2e}"
            ax.set_title(title, fontsize=10, weight='bold')
            ax.set_xlabel("Proxy Score w^T U(τ)")
            ax.set_ylabel("Learning Gain ΔL̂(τ)")
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(results), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved scatter plot grid to {output_path}")
        plt.close()
    
    def correlation_barplot(self,
                           results: List[CorrelationResult],
                           threshold: float = 0.3,
                           filename: str = "02_correlation_barplot.png"):
        """
        Generate barplot of correlation coefficients across environments.
        
        Args:
            results: List of CorrelationResult objects
            threshold: Horizontal line at this correlation value
            filename: Output filename
        """
        env_names = [r.environment for r in results]
        pearson_rs = [r.pearson_r for r in results]
        pearson_cis_low = [r.pearson_ci_low for r in results]
        pearson_cis_high = [r.pearson_ci_high for r in results]
        
        errors = [
            np.array(pearson_rs) - np.array(pearson_cis_low),
            np.array(pearson_cis_high) - np.array(pearson_rs),
        ]
        
        # Color bars by pass/fail
        colors = ['green' if r.passes_criterion else 'red' for r in results]
        
        fig, ax = plt.subplots(figsize=(max(len(env_names), 6), 4))
        
        bars = ax.bar(range(len(env_names)), pearson_rs, yerr=errors, 
                     capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Threshold line
        ax.axhline(y=threshold, color='black', linestyle='--', linewidth=2,
                  label=f'Pass threshold (ρ={threshold})')
        
        # Labels
        ax.set_xlabel("Environment")
        ax.set_ylabel("Pearson Correlation (ρ)")
        ax.set_title("Proxy Validity: Correlation Across Environments")
        ax.set_xticks(range(len(env_names)))
        ax.set_xticklabels(env_names, rotation=45, ha='right')
        ax.set_ylim([-0.1, 1.0])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved barplot to {output_path}")
        plt.close()
    
    def component_ablation_heatmap(self,
                                   results: List[CorrelationResult],
                                   filename: str = "03_component_ablation_heatmap.png"):
        """
        Generate heatmap of per-component correlations.
        
        Rows: Environments
        Cols: Utility components (Reward, Novelty, TD Error, Goal Proximity)
        Values: Correlation coefficients
        
        Args:
            results: List of CorrelationResult objects
            filename: Output filename
        """
        # Extract data
        env_names = [r.environment for r in results]
        component_names = list(results[0].component_correlations.keys())
        
        data = np.zeros((len(env_names), len(component_names)))
        for i, result in enumerate(results):
            for j, comp_name in enumerate(component_names):
                data[i, j] = result.component_correlations.get(comp_name, 0.0)
        
        # Generate heatmap
        fig, ax = plt.subplots(figsize=(len(component_names) * 1.5 + 1, len(env_names) * 0.8 + 1))
        
        sns.heatmap(data, 
                   xticklabels=component_names,
                   yticklabels=env_names,
                   annot=True, fmt='.2f',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Correlation (ρ)'},
                   vmin=0, vmax=1,
                   ax=ax)
        
        ax.set_title("Per-Component Correlation Ablation")
        ax.set_xlabel("Utility Components")
        ax.set_ylabel("Environmental Tasks")
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved heatmap to {output_path}")
        plt.close()
    
    def summary_statistics_figure(self,
                                  results: List[CorrelationResult],
                                  filename: str = "04_summary_statistics.png"):
        """
        Generate summary statistics figure with multiple subplots.
        
        Shows:
        - Distribution of Pearson correlations
        - Slope distribution
        - Pass/fail breakdown
        - Statistical summary table
        """
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        pearson_rs = [r.pearson_r for r in results]
        slopes = [r.slope for r in results]
        pass_count = sum(1 for r in results if r.passes_criterion)
        
        # 1. Correlation distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(pearson_rs, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(pearson_rs), color='red', linestyle='--', linewidth=2, label=f"Mean={np.mean(pearson_rs):.3f}")
        ax1.set_xlabel("Pearson ρ")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Correlations")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Slope distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(slopes, bins=10, color='lightcoral', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(slopes), color='red', linestyle='--', linewidth=2, label=f"Mean={np.mean(slopes):.3f}")
        ax2.set_xlabel("Regression Slope (β)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of Slopes")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Pass/fail breakdown
        ax3 = fig.add_subplot(gs[0, 2])
        labels = ['Pass', 'Fail']
        sizes = [pass_count, len(results) - pass_count]
        colors = ['green', 'red']
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title(f"Pass Rate: {pass_count}/{len(results)}")
        
        # 4. Statistical summary table
        ax4 = fig.add_subplot(gs[1, :])
        ax4.axis('off')
        
        summary_data = [
            ["Metric", "Value"],
            ["Mean Correlation", f"{np.mean(pearson_rs):.3f} ± {np.std(pearson_rs):.3f}"],
            ["Median Correlation", f"{np.median(pearson_rs):.3f}"],
            ["Min-Max Correlation", f"[{np.min(pearson_rs):.3f}, {np.max(pearson_rs):.3f}]"],
            ["Mean Slope", f"{np.mean(slopes):.3f} ± {np.std(slopes):.3f}"],
            ["Pass Rate", f"{pass_count}/{len(results)} ({100*pass_count/len(results):.1f}%)"],
        ]
        
        table = ax4.table(cellText=summary_data, loc='center', cellLoc='left',
                         colWidths=[0.3, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Header styling
        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor('lightgray')
            table[(0, i)].set_text_props(weight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved summary statistics figure to {output_path}")
        plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example: generate mock figures
    visualizer = CorrelationVisualizer(output_dir="./test_figs")
    
    # Mock results
    results = [
        CorrelationResult(
            environment=f"Env{i}",
            n_samples=500,
            pearson_r=0.3 + 0.1 * i + np.random.randn() * 0.05,
            pearson_p=0.01,
            pearson_ci_low=0.25,
            pearson_ci_high=0.45,
            spearman_r=0.32,
            spearman_p=0.01,
            spearman_ci_low=0.27,
            spearman_ci_high=0.47,
            kendall_tau=0.22,
            kendall_p=0.01,
            slope=0.5,
            slope_p=0.01,
            intercept=0.1,
            r_squared=0.15,
            cohens_q=0.31,
            component_correlations={"Reward": 0.2, "Novelty": 0.15, "TD Error": 0.25, "Goal Proximity": 0.1},
            passes_criterion=True,
            reason="All criteria pass",
        )
        for i in range(6)
    ]
    
    # Mock data
    proxy_scores_dict = {r.environment: np.random.randn(500) for r in results}
    learning_gains_dict = {r.environment: np.random.randn(500) for r in results}
    
    # Generate figures
    visualizer.scatter_plot_grid(results, proxy_scores_dict, learning_gains_dict)
    visualizer.correlation_barplot(results)
    visualizer.component_ablation_heatmap(results)
    visualizer.summary_statistics_figure(results)
    
    print("✓ Mock figures generated successfully")
