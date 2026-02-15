import torch.nn as nn



class GeLU(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        """
        Feed-Forward Network (FFN) module.

        Args:
            embed_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        Forward pass through the FFN.
        
        Args:
            x: Input tensor of shape (B, T, N, P) where:
               B = batch size
               T = number of frames
               N = number of patches
               P = embedding dimension
        Returns:
            torch.Tensor: Output tensor of shape (B, T, N, P)
        """

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        
        return x