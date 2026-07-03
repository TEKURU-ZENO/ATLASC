import torch

class TrajectoryLogger:
    def __init__(self):
        self.history = []
        
    def log(self, step, z, external_features=None):
        """
        Logs the state for analysis, explainability, or visualization.
        """
        log_entry = {
            'step': step,
            'z': z.detach().cpu().clone()
        }
        if external_features is not None:
            log_entry['features'] = external_features.detach().cpu().clone()
            
        self.history.append(log_entry)
        
    def get_trajectory(self):
        """
        Returns tensor of shape [steps, batch_size, num_genes, latent_dim]
        """
        if not self.history:
            return None
        return torch.stack([entry['z'] for entry in self.history])
        
    def clear(self):
        self.history = []
