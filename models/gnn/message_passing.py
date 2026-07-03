import torch
import torch.nn as nn

class MessagePassingLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear_msg = nn.Linear(in_channels, out_channels)
        self.linear_update = nn.Linear(in_channels + out_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels) # Structural bound on variance
        self.relu = nn.ReLU()

        # Initialize weights small to prevent cascading explosion
        nn.init.xavier_uniform_(self.linear_msg.weight, gain=0.1)
        nn.init.xavier_uniform_(self.linear_update.weight, gain=0.1)

    def forward(self, x, edge_index):
        # x: [batch_size, num_nodes, in_channels]
        batch_size, num_nodes, _ = x.size()
        src, dst = edge_index
        
        out_msg = self.linear_msg(x)
        messages = out_msg[:, src, :]
        
        agg_messages = torch.zeros(batch_size, num_nodes, out_msg.size(-1), device=x.device)
        dst_expanded = dst.view(1, -1, 1).expand(batch_size, -1, out_msg.size(-1))
        agg_messages.scatter_add_(1, dst_expanded, messages)
        
        combined = torch.cat([x, agg_messages], dim=-1)
        updated_x = self.linear_update(combined)
        
        # Apply LayerNorm strictly before activation
        updated_x = self.norm(updated_x)
        updated_x = self.relu(updated_x)
        
        return updated_x
