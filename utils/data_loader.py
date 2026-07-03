import os
import torch
from torch.utils.data import Dataset

class NormanDataset(Dataset):
    def __init__(self, processed_dir="data/processed", use_cell_states=False):
        """
        Loads the precomputed Norman 2019 dataset tensors.
        If use_cell_states is True, x is the specific cell's state. Otherwise it's the global basal.
        """
        super().__init__()
        print(f"Loading precomputed tensors from {processed_dir} into memory...")
        
        self.gene_names = torch.load(os.path.join(processed_dir, "gene_names.pt"), weights_only=False)
        self.interventions = torch.load(os.path.join(processed_dir, "interventions.pt"), weights_only=True)
        self.basal_state = torch.load(os.path.join(processed_dir, "basal_state.pt"), weights_only=True)
        self.X = torch.load(os.path.join(processed_dir, "expression_matrix.pt"), weights_only=True)
        self.use_cell_states = use_cell_states
        
        self.num_samples = self.X.shape[0]
        self.num_genes = self.X.shape[1]
        
        print("Dataset ready.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.use_cell_states:
            x = self.X[idx].clone()
        else:
            x = self.basal_state.clone()
            
        intervened_genes = self.interventions[idx]
        target = self.X[idx]
        
        return x, intervened_genes, target

def get_trrust_graph(processed_dir="data/processed"):
    """
    Returns the true biological human transcription factor network parsed from TRRUST.
    Shape: [2, E] where E = mapped edges + self-loops.
    """
    path = os.path.join(processed_dir, "trrust_edge_index.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run scripts/download_trrust.py first.")
    
    edge_index = torch.load(path, weights_only=True)
    return edge_index
