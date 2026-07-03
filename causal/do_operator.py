import torch

def apply_intervention(latent_state, gene_indices_batch, value=0.0):
    """
    Applies intervention to specific genes in the graph.
    latent_state: [batch_size, num_genes, latent_dim]
    gene_indices_batch: Tensor of shape [batch_size, max_interventions], padded with -1
    """
    # Ensure z is completely decoupled from the computation graph tracking previous states
    z = latent_state.clone()
    
    # Handle both [batch, genes, dim] and ensemble [K, batch, genes, dim] natively
    if z.dim() == 4:
        # z: [K, batch, genes, dim]
        K, batch_size, num_genes, latent_dim = z.size()
        for k in range(K):
            for b in range(batch_size):
                for idx in gene_indices_batch[b]:
                    if idx.item() != -1:
                        z[k, b, idx.item(), :] = value
    else:
        # z: [batch, genes, dim]
        for b in range(gene_indices_batch.size(0)):
            for idx in gene_indices_batch[b]:
                if idx.item() != -1:
                    z[b, idx.item(), :] = value
            
    return z
