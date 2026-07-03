import torch
from .stability_checks import StabilityChecker
from .trajectory_logger import TrajectoryLogger
from .transition_wrapper import TransitionWrapper

class RolloutEngine:
    def __init__(self, core_model, execution_mode="deterministic", noise_std=0.0):
        """
        Engine designed for multi-step temporal simulation of cell states.
        Built to support future Uncertainty Propagation and Explainability.
        """
        self.transition = TransitionWrapper(core_model, execution_mode, noise_std)
        self.checker = StabilityChecker(variance_threshold=100.0, max_norm=50.0)
        self.logger = TrajectoryLogger()
        
    def rollout(self, z0, edge_index, steps, log_trajectory=True, enforce_stability=True):
        """
        Unrolls the GNN state over time `steps`.
        z0: [batch_size, num_genes, latent_dim] initial state
        Returns final state, and optionally the full logged trajectory.
        """
        if log_trajectory:
            self.logger.clear()
            self.logger.log(0, z0)
            
        z_t = z0
        for step in range(1, steps + 1):
            # 1. Transition to t+1
            z_next = self.transition(z_t, edge_index)
            
            # 2. Stability Checks (early divergence detection)
            if enforce_stability:
                is_stable = self.checker.check(z_next, step)
                if not is_stable:
                    print(f"Rollout halted early at step {step} due to instability.")
                    break
            
            # 3. Log State
            if log_trajectory:
                self.logger.log(step, z_next)
                
            z_t = z_next
            
        return z_t, self.logger.get_trajectory() if log_trajectory else None
