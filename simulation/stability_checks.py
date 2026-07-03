import torch

class StabilityChecker:
    def __init__(self, variance_threshold=100.0, max_norm=50.0):
        self.variance_threshold = variance_threshold
        self.max_norm = max_norm
        
    def check(self, z, step):
        """
        Monitors a latent state trajectory for divergence.
        """
        # Feature variance
        var = torch.var(z)
        if var > self.variance_threshold:
            print(f"WARNING: Trajectory diverging at step {step}. Latent variance ({var:.2f}) exceeded threshold.")
            return False
            
        # Max norm
        norm = torch.norm(z, p=2, dim=-1).mean()
        if norm > self.max_norm:
            print(f"WARNING: Trajectory exploding at step {step}. Mean L2 norm ({norm:.2f}) exceeded threshold.")
            return False
            
        # NaN checks
        if torch.isnan(z).any():
            print(f"CRITICAL: Trajectory collapsed to NaN at step {step}.")
            return False
            
        return True
