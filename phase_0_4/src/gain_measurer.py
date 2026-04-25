"""
Learning Gain Measurement Module
Measures empirical learning gain ΔL̂(τ) via stratified SGD sampling.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GainMeasurement:
    """Single learning gain measurement result."""
    trajectory_id: int
    loss_before: float
    loss_after: float
    learning_gain: float
    quantile_bin: int
    utility_score: float


class StratifiedGainSampler:
    """Stratified sampling of learning gains across utility quantiles."""
    
    def __init__(self,
                 n_samples: int = 500,
                 n_quantiles: int = 4,
                 learning_rate: float = 1e-4):
        """
        Args:
            n_samples: Total number of trajectories to measure
            n_quantiles: Number of quantile bins for stratification
            learning_rate: SGD step size for gain measurement
        """
        self.n_samples = n_samples
        self.n_quantiles = n_quantiles
        self.samples_per_quantile = n_samples // n_quantiles
        self.learning_rate = learning_rate
        
        logger.info(f"Stratified sampler: {n_samples} samples across {n_quantiles} quantiles")
    
    def stratify_by_utility(self,
                           utility_scores: np.ndarray,
                           n_quantiles: Optional[int] = None) -> List[np.ndarray]:
        """
        Partition indices into quantile-based strata.
        
        Args:
            utility_scores: Shape (N,) - aggregate utility scores
            n_quantiles: Override default quantile count
            
        Returns:
            List of index arrays, one per quantile
        """
        n_q = n_quantiles or self.n_quantiles
        quantiles = np.percentile(utility_scores, 
                                  np.linspace(0, 100, n_q + 1))
        
        strata = []
        for q_idx in range(n_q):
            mask = ((utility_scores >= quantiles[q_idx]) & 
                   (utility_scores < quantiles[q_idx + 1]))
            stratum_indices = np.where(mask)[0]
            strata.append(stratum_indices)
            logger.debug(f"Quantile {q_idx}: {len(stratum_indices)} trajectories")
        
        return strata
    
    def sample_stratified_indices(self,
                                 utility_scores: np.ndarray) -> np.ndarray:
        """
        Sample trajectory indices stratified by utility quantiles.
        
        Args:
            utility_scores: Shape (N,) - aggregate scores
            
        Returns:
            Shape (n_samples,) - sampled trajectory indices
        """
        strata = self.stratify_by_utility(utility_scores)
        sampled_indices = []
        
        for q_idx, stratum in enumerate(strata):
            if len(stratum) == 0:
                logger.warning(f"Empty stratum {q_idx}")
                continue
            
            # Sample without replacement from this quantile
            sample_size = min(self.samples_per_quantile, len(stratum))
            q_samples = np.random.choice(stratum, sample_size, replace=False)
            sampled_indices.extend(q_samples)
        
        sampled_indices = np.array(sampled_indices[:self.n_samples])
        logger.info(f"Sampled {len(sampled_indices)} trajectories for gain measurement")
        return sampled_indices
    
    def measure_gain_single(self,
                           trajectory: Dict,
                           loss_fn: Callable[[Dict], float],
                           model: object,
                           trajectory_id: int,
                           utility_score: float,
                           quantile_bin: int) -> GainMeasurement:
        """
        Measure learning gain for a single trajectory.
        
        ΔL̂(τ) = L(θ_0; τ) - L(θ_1; τ)
        
        Where:
          - θ_0: initial parameters
          - θ_1: parameters after one SGD step on τ
        
        Args:
            trajectory: The trajectory dict
            loss_fn: Function to compute loss: callable(model, trajectory) -> scalar
            model: Model object with train_step method
            trajectory_id: Identifier for this trajectory
            utility_score: Pre-computed w^T U(τ)
            quantile_bin: Which quantile this trajectory came from
            
        Returns:
            GainMeasurement result
        """
        # Compute baseline loss (before update)
        model.eval()
        loss_before = loss_fn(model, trajectory)
        
        # Single SGD step on this trajectory
        model.train()
        grads = model.compute_gradients(trajectory)
        model.update_params(grads, self.learning_rate)
        
        # Compute updated loss
        model.eval()
        loss_after = loss_fn(model, trajectory)
        
        # Learning gain
        gain = loss_before - loss_after
        
        measurement = GainMeasurement(
            trajectory_id=trajectory_id,
            loss_before=loss_before,
            loss_after=loss_after,
            learning_gain=float(gain),
            quantile_bin=quantile_bin,
            utility_score=utility_score,
        )
        
        # Important: restore model checkpoint to avoid parameter drift
        model.restore_checkpoint()
        
        return measurement
    
    def measure_gains_batch(self,
                           trajectories: List[Dict],
                           sampled_indices: np.ndarray,
                           utility_scores: np.ndarray,
                           loss_fn: Callable,
                           model: object,
                           strata: List[np.ndarray]) -> List[GainMeasurement]:
        """
        Measure learning gains for a stratified sample of trajectories.
        
        Args:
            trajectories: Full list of trajectories
            sampled_indices: Indices to measure (from stratified sampling)
            utility_scores: Aggregate scores for each trajectory
            loss_fn: Loss function
            model: Model object
            strata: Quantile strata (for bin assignment)
            
        Returns:
            List of GainMeasurement results
        """
        measurements = []
        
        for sample_idx, traj_id in enumerate(sampled_indices):
            # Determine which quantile bin this trajectory belongs to
            quantile_bin = 0
            for q_idx, stratum in enumerate(strata):
                if traj_id in stratum:
                    quantile_bin = q_idx
                    break
            
            traj = trajectories[traj_id]
            util_score = utility_scores[traj_id]
            
            # Measure gain
            measurement = self.measure_gain_single(
                trajectory=traj,
                loss_fn=loss_fn,
                model=model,
                trajectory_id=traj_id,
                utility_score=util_score,
                quantile_bin=quantile_bin,
            )
            measurements.append(measurement)
            
            if (sample_idx + 1) % 50 == 0:
                logger.info(f"Measured gains: {sample_idx + 1} / {len(sampled_indices)}")
        
        return measurements


class LossComputers:
    """Pre-defined loss functions for different RL algorithms."""
    
    @staticmethod
    def dreamer_imagination_loss(model, trajectory: Dict) -> float:
        """
        Imagination loss for DreamerV3 (world model training).
        
        L_imagination = λ_model * L_model + λ_value * L_value + λ_policy * L_policy
        """
        # This would typically use model.imagine_loss() or similar
        # For now, placeholder implementation
        
        z_true = trajectory.get("z_true")
        z_pred = trajectory.get("z_pred")
        values = trajectory.get("values")
        rewards = trajectory.get("rewards")
        
        if z_true is None or z_pred is None:
            return 0.0
        
        # Model prediction loss
        l_model = np.mean((z_true[1:] - z_pred) ** 2)
        
        # Value loss (bootstrapped)
        if values is not None and len(rewards) > 0:
            v_target = rewards + 0.99 * values[1:]
            l_value = np.mean((values[:-1] - v_target) ** 2)
        else:
            l_value = 0.0
        
        # Combined loss
        total_loss = 0.6 * l_model + 0.4 * l_value
        return float(total_loss)
    
    @staticmethod
    def behavioral_cloning_loss(model, trajectory: Dict) -> float:
        """
        Behavioral cloning loss (policy imitation).
        
        L_bc = E[-log π(a_t | z_t)]
        """
        # Placeholder: compute cross-entropy loss
        # In real implementation, would use model.compute_policy_loss()
        return 0.1  # dummy value
    
    @staticmethod
    def combined_dreamer_loss(model, trajectory: Dict,
                             weights: Tuple[float, float, float] = (0.6, 0.3, 0.1)) -> float:
        """
        Combined loss: imagination + behavioral cloning + regularization.
        """
        w_imag, w_bc, w_reg = weights
        
        l_imag = LossComputers.dreamer_imagination_loss(model, trajectory)
        l_bc = LossComputers.behavioral_cloning_loss(model, trajectory)
        l_reg = 0.0  # placeholder for regularization
        
        return w_imag * l_imag + w_bc * l_bc + w_reg * l_reg


class GainAnalyzer:
    """Analyze and report learning gain statistics."""
    
    @staticmethod
    def compute_gain_statistics(measurements: List[GainMeasurement]) -> Dict:
        """
        Compute summary statistics over measurements.
        
        Returns:
            Dict with keys: {mean, std, median, quantiles, per_bin}
        """
        gains = np.array([m.learning_gain for m in measurements])
        
        stats = {
            "n_measurements": len(measurements),
            "mean": np.mean(gains),
            "std": np.std(gains),
            "median": np.median(gains),
            "min": np.min(gains),
            "max": np.max(gains),
            "q25": np.percentile(gains, 25),
            "q75": np.percentile(gains, 75),
        }
        
        # Per-quantile analysis
        bins = {}
        for m in measurements:
            b = m.quantile_bin
            if b not in bins:
                bins[b] = []
            bins[b].append(m.learning_gain)
        
        stats["per_bin"] = {}
        for b, gains_b in bins.items():
            stats["per_bin"][b] = {
                "mean": np.mean(gains_b),
                "std": np.std(gains_b),
                "n": len(gains_b),
            }
        
        return stats
    
    @staticmethod
    def align_gains_with_utilities(measurements: List[GainMeasurement]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract aligned arrays of utility scores and learning gains.
        
        Returns:
            (utility_scores, learning_gains) both shape (N,)
        """
        utility_scores = np.array([m.utility_score for m in measurements])
        learning_gains = np.array([m.learning_gain for m in measurements])
        return utility_scores, learning_gains
    
    @staticmethod
    def filter_outliers(measurements: List[GainMeasurement],
                       percentile_low: float = 10,
                       percentile_high: float = 90) -> List[GainMeasurement]:
        """
        Filter outlier gains (for robustness checks).
        
        Args:
            measurements: Original measurements
            percentile_low: Lower bound percentile
            percentile_high: Upper bound percentile
            
        Returns:
            Filtered measurements (excluding outliers)
        """
        gains = np.array([m.learning_gain for m in measurements])
        low_bound = np.percentile(gains, percentile_low)
        high_bound = np.percentile(gains, percentile_high)
        
        filtered = [m for m in measurements
                   if low_bound <= m.learning_gain <= high_bound]
        
        logger.info(f"Filtered from {len(measurements)} to {len(filtered)} "
                   f"measurements (removed {len(measurements) - len(filtered)} outliers)")
        
        return filtered


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    sampler = StratifiedGainSampler(n_samples=100, n_quantiles=4)
    
    # Mock utility scores
    utility_scores = np.random.randn(1000)
    
    # Stratify
    sampled_indices = sampler.sample_stratified_indices(utility_scores)
    print(f"Sampled indices: {sampled_indices[:10]}")
    
    # Mock measurements
    measurements = [
        GainMeasurement(
            trajectory_id=i,
            loss_before=1.0,
            loss_after=0.8,
            learning_gain=0.2,
            quantile_bin=i % 4,
            utility_score=utility_scores[i],
        )
        for i in sampled_indices
    ]
    
    stats = GainAnalyzer.compute_gain_statistics(measurements)
    print(f"Gain statistics: {stats}")
