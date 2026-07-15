import torch
import torch.nn as nn
from agents.agent import Agent

class MLPC(nn.Module, Agent):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, max_action=10, **kwargs):
        super().__init__(**kwargs)

        self.layers     = nn.ModuleList()
        self.activation = nn.ReLU()

        self.layers.append(nn.Linear(input_dim, hidden_dim))

        for i in range(n_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, output_dim*2)) # For each output, we calculate mean and std

        self.init_weights()
        self.type = 'continuous'
        self.max_action = max_action
        self.min_log_std = torch.log(0.3 * 2 * max_action) # 30% of the action range
        self.max_log_std = -4

    def init_weights(self):
        # Normal hidden-layer initialization
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Make initial logits exactly zero
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Not applying the activation function to the last layer
            if i < len(self.layers) - 1:
                x = self.activation(x)

        return x
    
    def predict(self, x):
        device = next(self.parameters()).device
        x = torch.as_tensor(x, dtype=torch.float32, device=device)

        with torch.no_grad():
            return self(x)


