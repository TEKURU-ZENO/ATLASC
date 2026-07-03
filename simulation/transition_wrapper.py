import torch
import torch.nn as nn

class TransitionWrapper(nn.Module):
    """
    Wraps the core step logic to support Ensembles, Monte Carlo dropouts, or Noise.
    This separates the mathematical "step" from the execution paradigm.
    """
    def __init__(self, core_model, execution_mode="deterministic", noise_std=0.0):
        super().__init__()
        self.core_model = core_model # e.g. the GNN
        self.execution_mode = execution_mode
        self.noise_std = noise_std
        
    def forward(self, z, edge_index):
        if self.execution_mode == "deterministic":
            return self._deterministic_step(z, edge_index)
        elif self.execution_mode == "noisy":
            return self._noisy_step(z, edge_index)
        elif self.execution_mode == "ensemble":
            return self._ensemble_step(z, edge_index)
        else:
            raise ValueError(f"Unknown mode: {self.execution_mode}")
            
    def _deterministic_step(self, z, edge_index):
        # Normal inference step
        return self.core_model(z, edge_index)
        
    def _noisy_step(self, z, edge_index):
        # Inject standard gaussian noise for Monte-Carlo rollout uncertainty
        noise = torch.randn_like(z) * self.noise_std
        z_noisy = z + noise
        return self.core_model(z_noisy, edge_index)
        
    def _ensemble_step(self, z, edge_index):
        # Placeholder for ensemble support. 
        # Future: self.core_model is a ModuleList and we average / track variance
        pass
