import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden_dim, n_nodes, output_dim):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.n_nodes     = n_nodes
        self.output_dim  = output_dim