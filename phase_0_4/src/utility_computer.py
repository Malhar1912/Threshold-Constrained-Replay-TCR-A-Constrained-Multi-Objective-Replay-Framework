"""
Utility Vector Computation Module
Computes multi-objective utility vectors U(τ) for trajectories using existing forward passes.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class UtilityStats:
    """Running statistics for utility normalization via EMA."""
    mean: float = 0.0
    var: float = 1.0
    count: int = 0
    
    def update(self, value: float, ema_decay: float = 0.01):
        """Update stats using exponential moving average."""
        delta = value - self.mean
        self.mean += ema_decay * delta
        self.var = (1 - ema_decay) * (self.var + ema_decay * delta**2) + ema_decay * (value - self.mean)**2
        self.count += 1
    
    def normalize(self, value: float, epsilon: float = 1e-8) -> float:
        """Normalize value using running stats: (x - μ) / (σ + ε)"""
        std = np.sqrt(self.var) + epsilon
        return (value - self.mean) / std


class UtilityComputer:
    """Compute utility vectors for trajectories without extra computational overhead."""
    
    def __init__(self, 
                 gamma: float = 0.99,
                 ema_decay: float = 0.01,
                 epsilon: float = 1e-8,
                 task_type: str = "dense"):
        """
        Args:
            gamma: Discount factor
            ema_decay: EMA decay for normalization
            epsilon: Numerical stability constant
            task_type: "sparse" or "dense" (controls goal proximity component)
        """
        self.gamma = gamma
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.task_type = task_type
        
        # Running statistics for normalization
        self.stats = {
            "reward": UtilityStats(),
            "novelty": UtilityStats(),
            "td_error": UtilityStats(),
            "goal_proximity": UtilityStats(),
        }
    
    def compute_reward_utility(self, 
                               rewards: np.ndarray) -> float:
        """
        Compute discounted episodic reward.
        
        R(τ) = (1/T) Σ_t γ^t r_t
        
        Args:
            rewards: Shape (T,) - reward sequence
            
        Returns:
            Scalar reward utility
        """
        T = len(rewards)
        discounts = self.gamma ** np.arange(T)
        reward_util = np.sum(rewards * discounts) / T
        return float(reward_util)
    
    def compute_novelty_utility(self,
                               z_true: np.ndarray,
                               z_pred: np.ndarray) -> float:
        """
        Compute world model prediction error (novelty).
        
        N(τ) = (1/T) Σ_t ||z_{t+1}^true - ẑ_{t+1}||^2
        
        Args:
            z_true: Shape (T+1, D) - true state embeddings
            z_pred: Shape (T, D) - predicted embeddings from world model
            
        Returns:
            Scalar novelty utility
        """
        if len(z_true) != len(z_pred) + 1:
            raise ValueError(f"Shape mismatch: z_true={z_true.shape}, z_pred={z_pred.shape}")
        
        # Compare predicted vs actual next states
        z_true_next = z_true[1:]  # shape (T, D)
        l2_errors = np.linalg.norm(z_true_next - z_pred, axis=1)
        novelty_util = np.mean(l2_errors ** 2)
        return float(novelty_util)
    
    def compute_td_error_utility(self,
                                values: np.ndarray,
                                rewards: np.ndarray) -> float:
        """
        Compute TD error (value function uncertainty).
        
        ΔV(τ) = (1/T) Σ_t |V(z_t) - (r_t + γ V(z_{t+1}))|
        
        Args:
            values: Shape (T+1,) - value function estimates
            rewards: Shape (T,) - rewards
            
        Returns:
            Scalar TD error utility
        """
        if len(values) != len(rewards) + 1:
            raise ValueError(f"Shape mismatch: values={values.shape}, rewards={rewards.shape}")
        
        v_current = values[:-1]  # V(z_t)
        v_next = values[1:]      # V(z_{t+1})
        
        # Bootstrap targets: r_t + γ V(z_{t+1})
        bootstrapped_targets = rewards + self.gamma * v_next
        td_errors = np.abs(v_current - bootstrapped_targets)
        td_error_util = np.mean(td_errors)
        return float(td_error_util)
    
    def compute_goal_proximity_utility(self,
                                       z_final: np.ndarray,
                                       z_goal: Optional[np.ndarray] = None) -> float:
        """
        Compute goal proximity (for sparse-reward tasks only).
        
        G(τ) = -||z_T - z_goal||_2
        
        Args:
            z_final: Final state embedding
            z_goal: Goal state embedding (if None, returns 0)
            
        Returns:
            Scalar goal proximity utility
        """
        if self.task_type == "dense" or z_goal is None:
            return 0.0
        
        distance = np.linalg.norm(z_final - z_goal)
        goal_util = -distance
        return float(goal_util)
    
    def compute_trajectory_utilities(self,
                                    trajectory: Dict,
                                    z_goal: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute full utility vector for a trajectory.
        
        U(τ) = [R̂(τ), N̂(τ), ΔV̂(τ), Ĝ(τ)]
        
        Args:
            trajectory: Dict with keys {rewards, z_true, z_pred, values}
            z_goal: Goal state embedding (optional, for sparse tasks)
            
        Returns:
            4-dimensional normalized utility vector
        """
        # Extract trajectory components
        rewards = trajectory.get("rewards", np.array([]))
        z_true = trajectory.get("z_true", np.array([]))
        z_pred = trajectory.get("z_pred", np.array([]))
        values = trajectory.get("values", np.array([]))
        
        # Compute raw utilities
        raw_utilities = []
        
        # 1. Reward
        r_util = self.compute_reward_utility(rewards)
        raw_utilities.append(("reward", r_util))
        
        # 2. Novelty
        n_util = self.compute_novelty_utility(z_true, z_pred)
        raw_utilities.append(("novelty", n_util))
        
        # 3. TD Error
        td_util = self.compute_td_error_utility(values, rewards)
        raw_utilities.append(("td_error", td_util))
        
        # 4. Goal Proximity
        g_util = self.compute_goal_proximity_utility(z_true[-1], z_goal)
        raw_utilities.append(("goal_proximity", g_util))
        
        # Normalize using running stats
        normalized_utils = []
        for name, util_value in raw_utilities:
            # Update running statistics
            self.stats[name].update(util_value, self.ema_decay)
            # Normalize
            normalized_util = self.stats[name].normalize(util_value, self.epsilon)
            normalized_utils.append(normalized_util)
        
        return np.array(normalized_utils, dtype=np.float32)
    
    def compute_batch_utilities(self,
                               trajectories: List[Dict],
                               z_goals: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        Compute utility vectors for a batch of trajectories.
        
        Args:
            trajectories: List of trajectory dicts
            z_goals: List of goal embeddings (optional)
            
        Returns:
            Shape (N, 4) array of normalized utilities
        """
        n_trajectories = len(trajectories)
        utilities = np.zeros((n_trajectories, 4), dtype=np.float32)
        
        for i, traj in enumerate(trajectories):
            z_goal = z_goals[i] if z_goals else None
            utilities[i] = self.compute_trajectory_utilities(traj, z_goal)
        
        logger.info(f"Computed utilities for {n_trajectories} trajectories")
        return utilities
    
    def get_statistics_summary(self) -> Dict:
        """Return summary of running normalization statistics."""
        summary = {}
        for name, stat in self.stats.items():
            summary[name] = {
                "mean": stat.mean,
                "std": np.sqrt(stat.var),
                "count": stat.count,
            }
        return summary
    
    def reset_statistics(self):
        """Reset running statistics (for new environment/seed)."""
        for name in self.stats:
            self.stats[name] = UtilityStats()
        logger.info("Reset utility normalization statistics")


class AggregateUtilityScore:
    """Compute aggregate utility score w^T U(τ) for sampling."""
    
    @staticmethod
    def compute_score(utilities: np.ndarray,
                     weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute weighted aggregate: s_i = w^T U_i
        
        Args:
            utilities: Shape (N, d) - utility vectors per trajectory
            weights: Shape (d,) - weight vector (if None, use uniform)
            
        Returns:
            Shape (N,) - aggregate scores
        """
        if weights is None:
            weights = np.ones(utilities.shape[1]) / np.sqrt(utilities.shape[1])
        
        # Ensure shapes match
        if utilities.shape[1] != len(weights):
            raise ValueError(f"Utility dim {utilities.shape[1]} != weight dim {len(weights)}")
        
        scores = utilities @ weights
        return scores


# Utility functions
def initialize_equal_weights(n_components: int) -> np.ndarray:
    """Initialize equal weight vector: w = 1_d / sqrt(d)"""
    return np.ones(n_components) / np.sqrt(n_components)


def fit_weights_regression(utilities: np.ndarray,
                          learning_gains: np.ndarray) -> np.ndarray:
    """
    Fit weight vector via least-squares regression.
    
    w* = argmin_w ||w^T U - ΔL||^2
    """
    from scipy.linalg import lstsq
    w, _, _, _ = lstsq(utilities, learning_gains)
    return w


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    computer = UtilityComputer(gamma=0.99, task_type="dense")
    
    # Mock trajectory
    mock_traj = {
        "rewards": np.array([1.0, 2.0, 3.0]),
        "z_true": np.random.randn(4, 32),      # 4 timesteps, 32-dim embeddings
        "z_pred": np.random.randn(3, 32),      # 3 predictions
        "values": np.array([0.5, 1.0, 2.0, 3.5]),  # 4 value estimates
    }
    
    utilities = computer.compute_trajectory_utilities(mock_traj)
    print(f"Utilities: {utilities}")
    print(f"Stats: {computer.get_statistics_summary()}")
