import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        """
        Root Mean Square Layer Normalization.

        Args:
            dim: Dimension of the input features
            eps: Small value to prevent division by zero
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            torch.Tensor: Normalized tensor of shape (..., dim)
        """
        # Compute RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize by RMS and scale by learnable weight
        x_norm = x / rms

        return self.weight * x_norm
