"""
Correlation Analysis Module
Computes and validates correlation between proxy scores and empirical learning gains.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Result of correlation analysis."""
    environment: str
    n_samples: int
    
    # Primary metric: Pearson
    pearson_r: float
    pearson_p: float
    pearson_ci_low: float
    pearson_ci_high: float
    
    # Robustness: Spearman
    spearman_r: float
    spearman_p: float
    spearman_ci_low: float
    spearman_ci_high: float
    
    # Robustness: Kendall
    kendall_tau: float
    kendall_p: float
    
    # Regression
    slope: float
    slope_p: float
    intercept: float
    r_squared: float
    
    # Effect size
    cohens_q: float
    
    # Per-component
    component_correlations: Dict[str, float]
    
    # Pass/fail
    passes_criterion: bool
    reason: str


class CorrelationValidator:
    """Compute and validate correlations with statistical rigor."""
    
    def __init__(self,
                 significance_level: float = 0.05,
                 ci_level: float = 0.95,
                 bootstrap_samples: int = 10000,
                 min_correlation: float = 0.3,
                 spearman_tolerance: float = 0.05):
        """
        Args:
            significance_level: Alpha for hypothesis tests
            ci_level: Confidence level for intervals (e.g., 0.95 for 95% CI)
            bootstrap_samples: Number of bootstrap resamples
            min_correlation: Minimum correlation required to pass
            spearman_tolerance: Max allowed difference |pearson - spearman|
        """
        self.significance_level = significance_level
        self.ci_level = ci_level
        self.bootstrap_samples = bootstrap_samples
        self.min_correlation = min_correlation
        self.spearman_tolerance = spearman_tolerance
    
    @staticmethod
    def pearson_correlation(x: np.ndarray,
                           y: np.ndarray) -> Tuple[float, float]:
        """
        Compute Pearson correlation with p-value.
        
        Args:
            x: Shape (N,) - proxy scores
            y: Shape (N,) - learning gains
            
        Returns:
            (correlation, p_value)
        """
        from scipy.stats import pearsonr
        r, p = pearsonr(x, y)
        return float(r), float(p)
    
    @staticmethod
    def spearman_correlation(x: np.ndarray,
                            y: np.ndarray) -> Tuple[float, float]:
        """Spearman rank correlation (robust to outliers)."""
        from scipy.stats import spearmanr
        r, p = spearmanr(x, y)
        return float(r), float(p)
    
    @staticmethod
    def kendall_correlation(x: np.ndarray,
                           y: np.ndarray) -> Tuple[float, float]:
        """Kendall's tau correlation (pairwise concordance)."""
        from scipy.stats import kendalltau
        tau, p = kendalltau(x, y)
        return float(tau), float(p)
    
    def bootstrap_ci(self,
                    x: np.ndarray,
                    y: np.ndarray,
                    correlation_fn=None) -> Tuple[float, float]:
        """
        Compute 95% confidence interval via bootstrap.
        
        Args:
            x, y: Data arrays
            correlation_fn: Function to compute correlation
            
        Returns:
            (ci_low, ci_high)
        """
        if correlation_fn is None:
            correlation_fn = self.pearson_correlation
        
        n = len(x)
        bootstrap_correlations = []
        
        for _ in range(self.bootstrap_samples):
            # Resample with replacement
            indices = np.random.choice(n, n, replace=True)
            x_boot = x[indices]
            y_boot = y[indices]
            
            # Compute correlation on bootstrap sample
            r, _ = correlation_fn(x_boot, y_boot)
            bootstrap_correlations.append(r)
        
        bootstrap_correlations = np.array(bootstrap_correlations)
        ci_low = np.percentile(bootstrap_correlations, (1 - self.ci_level) / 2 * 100)
        ci_high = np.percentile(bootstrap_correlations, (1 + self.ci_level) / 2 * 100)
        
        return float(ci_low), float(ci_high)
    
    @staticmethod
    def linear_regression(x: np.ndarray,
                         y: np.ndarray) -> Tuple[float, float, float, float, float]:
        """
        Fit linear regression y = α + β*x.
        
        Returns:
            (slope, intercept, slope_p_value, r_squared, residual_std)
        """
        from scipy.stats import linregress
        result = linregress(x, y)
        
        # Compute R-squared
        y_pred = result.slope * x + result.intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return float(result.slope), float(result.intercept), float(result.pvalue), float(r_squared), float(result.rvalue)
    
    @staticmethod
    def cohens_q(r: float) -> float:
        """
        Compute Cohen's q effect size from correlation coefficient.
        
        q = 0.5 * ln((1+r)/(1-r))
        
        Interpretation:
          q ≈ 0.1: small
          q ≈ 0.3: medium
          q ≈ 0.5: large
        """
        r = np.clip(r, -0.999, 0.999)  # Avoid division by zero
        q = 0.5 * np.log((1 + r) / (1 - r))
        return float(q)
    
    def compute_component_correlations(self,
                                      utilities: np.ndarray,
                                      learning_gains: np.ndarray) -> Dict[str, float]:
        """
        Compute correlation for each utility component individually.
        
        Args:
            utilities: Shape (N, d) - utility vectors
            learning_gains: Shape (N,) - learning gains
            
        Returns:
            Dict mapping component name to correlation
        """
        component_names = ["Reward", "Novelty", "TD Error", "Goal Proximity"]
        correlations = {}
        
        for i, name in enumerate(component_names):
            if i < utilities.shape[1]:
                r, _ = self.pearson_correlation(utilities[:, i], learning_gains)
                correlations[name] = float(r)
        
        return correlations
    
    def validate_correlation(self,
                            proxy_scores: np.ndarray,
                            learning_gains: np.ndarray,
                            utilities: Optional[np.ndarray] = None,
                            environment: str = "Unknown") -> CorrelationResult:
        """
        Full correlation validation pipeline.
        
        Args:
            proxy_scores: Shape (N,) - w^T U(τ)
            learning_gains: Shape (N,) - ΔL̂(τ)
            utilities: Shape (N, d) - component utility vectors (for ablation)
            environment: Name of environment for reporting
            
        Returns:
            CorrelationResult object
        """
        n = len(proxy_scores)
        
        # Ensure valid data
        valid_mask = (np.isfinite(proxy_scores) & np.isfinite(learning_gains))
        proxy_scores = proxy_scores[valid_mask]
        learning_gains = learning_gains[valid_mask]
        if utilities is not None:
            utilities = utilities[valid_mask]
        
        n_valid = len(proxy_scores)
        logger.info(f"Valid samples for {environment}: {n_valid}/{n}")
        
        # Compute correlations
        pearson_r, pearson_p = self.pearson_correlation(proxy_scores, learning_gains)
        spearman_r, spearman_p = self.spearman_correlation(proxy_scores, learning_gains)
        kendall_tau, kendall_p = self.kendall_correlation(proxy_scores, learning_gains)
        
        # Confidence intervals
        pearson_ci_low, pearson_ci_high = self.bootstrap_ci(proxy_scores, learning_gains)
        spearman_ci_low, spearman_ci_high = self.bootstrap_ci(proxy_scores, learning_gains,
                                                               self.spearman_correlation)
        
        # Regression
        slope, intercept, slope_p, r_squared, _ = self.linear_regression(proxy_scores, learning_gains)
        
        # Effect size
        cohens_q = self.cohens_q(pearson_r)
        
        # Component correlations
        component_corrs = {}
        if utilities is not None:
            component_corrs = self.compute_component_correlations(utilities, learning_gains)
        
        # Determine pass/fail
        passes = (
            (pearson_r > self.min_correlation) and
            (pearson_p < self.significance_level) and
            (abs(pearson_r - spearman_r) < self.spearman_tolerance) and
            (slope > 0) and (slope_p < 0.05)
        )
        
        reason = ""
        if not passes:
            fails = []
            if pearson_r <= self.min_correlation:
                fails.append(f"r={pearson_r:.3f} ≤ {self.min_correlation}")
            if pearson_p >= self.significance_level:
                fails.append(f"p={pearson_p:.4f} ≥ {self.significance_level}")
            if abs(pearson_r - spearman_r) >= self.spearman_tolerance:
                fails.append(f"|pearson-spearman|={abs(pearson_r - spearman_r):.3f} too large")
            if slope <= 0:
                fails.append(f"slope={slope:.3f} ≤ 0")
            reason = "; ".join(fails)
        else:
            reason = "All criteria pass"
        
        result = CorrelationResult(
            environment=environment,
            n_samples=n_valid,
            pearson_r=pearson_r,
            pearson_p=pearson_p,
            pearson_ci_low=pearson_ci_low,
            pearson_ci_high=pearson_ci_high,
            spearman_r=spearman_r,
            spearman_p=spearman_p,
            spearman_ci_low=spearman_ci_low,
            spearman_ci_high=spearman_ci_high,
            kendall_tau=kendall_tau,
            kendall_p=kendall_p,
            slope=slope,
            slope_p=slope_p,
            intercept=intercept,
            r_squared=r_squared,
            cohens_q=cohens_q,
            component_correlations=component_corrs,
            passes_criterion=passes,
            reason=reason,
        )
        
        return result
    
    @staticmethod
    def format_result(result: CorrelationResult) -> str:
        """Format result for printing."""
        status = "✓ PASS" if result.passes_criterion else "✗ FAIL"
        
        lines = [
            f"\n{status} Environment: {result.environment}",
            f"{'─' * 50}",
            f"Sample Size:              n = {result.n_samples}",
            f"Pearson r:                ρ = {result.pearson_r:.3f} (95% CI: [{result.pearson_ci_low:.3f}, {result.pearson_ci_high:.3f}])",
            f"P-value:                  p = {result.pearson_p:.2e}",
            f"Spearman ρ_s:             ρ_s = {result.spearman_r:.3f} (CI: [{result.spearman_ci_low:.3f}, {result.spearman_ci_high:.3f}])",
            f"Kendall τ:                τ = {result.kendall_tau:.3f} (p = {result.kendall_p:.2e})",
            f"\nRegression:",
            f"  Slope (β):              β = {result.slope:.3f} (p = {result.slope_p:.2e})",
            f"  Intercept (α):          α = {result.intercept:.3f}",
            f"  R² (goodness-of-fit):   R² = {result.r_squared:.3f}",
            f"\nEffect Size:",
            f"  Cohen's q:              q = {result.cohens_q:.3f}",
            f"\nPer-Component Correlations:",
        ]
        
        for comp_name, comp_r in result.component_correlations.items():
            lines.append(f"  {comp_name:20s}: ρ = {comp_r:.3f}")
        
        lines.append(f"\nDecision: {result.reason}")
        
        return "\n".join(lines)


class CorrelationReporter:
    """Generate summary reports across multiple environments."""
    
    @staticmethod
    def aggregate_results(results: List[CorrelationResult]) -> Dict:
        """Compute aggregate statistics across environments."""
        pass_count = sum(1 for r in results if r.passes_criterion)
        pass_rate = pass_count / len(results) if results else 0
        
        pearson_rs = [r.pearson_r for r in results]
        
        return {
            "n_environments": len(results),
            "n_pass": pass_count,
            "pass_rate": pass_rate,
            "mean_pearson_r": np.mean(pearson_rs),
            "std_pearson_r": np.std(pearson_rs),
            "median_pearson_r": np.median(pearson_rs),
            "results": results,
        }
    
    @staticmethod
    def print_summary(aggregate: Dict):
        """Print summary report."""
        print("\n" + "=" * 60)
        print("CORRELATION VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Environments tested:  {aggregate['n_environments']}")
        print(f"Passed criterion:     {aggregate['n_pass']}/{aggregate['n_environments']} ({aggregate['pass_rate']*100:.1f}%)")
        print(f"Mean Pearson r:       {aggregate['mean_pearson_r']:.3f} ± {aggregate['std_pearson_r']:.3f}")
        print(f"Median Pearson r:     {aggregate['median_pearson_r']:.3f}")
        print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    validator = CorrelationValidator(min_correlation=0.3)
    
    # Mock data
    n = 500
    true_gains = np.random.randn(n) * 2
    proxy_scores = 0.4 * true_gains + np.random.randn(n) * 1.5
    
    result = validator.validate_correlation(proxy_scores, true_gains, environment="Pong")
    print(CorrelationValidator.format_result(result))
