import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim):
        super().__init__()

        self.layers     = nn.ModuleList()
        self.activation = nn.ReLU()
        self.softmax    = nn.Softmax(dim=0)

        self.layers.append(nn.Linear(input_dim, hidden_dim))

        for i in range(n_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Not applying the activation function to the last layer
            if i < len(self.layers) - 1:
                x = self.activation(x)
        # To get the probabilities for the last layer
        x = self.softmax(x)
        return x
    
    def predict(self, x):
        device = next(self.parameters()).device
        x = torch.as_tensor(x, dtype=torch.float32, device=device)

        with torch.no_grad():
            return self(x)


