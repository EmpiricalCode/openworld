import torch
import torch.nn as nn


class FSQ(nn.Module):
    def __init__(self, latent_dim, num_bins):
        """
        Finite Scalar Quantization (FSQ).

        Args:
            latent_dim: Number of latent dimensions
            num_bins: List of integers specifying the number of quantization levels for each dimension.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_bins = num_bins
        self.codebook_size = num_bins ** latent_dim
        powers = torch.tensor([num_bins ** i for i in range(latent_dim)])
        self.register_buffer('powers', powers)

    def forward(self, x):
        """
        Forward pass for quantization.

        Args:
            x: Input tensor of shape (B, T, N, latent_dim) where:
               B = batch size
               T = number of frames
               N = number of patches
               latent_dim = latent dimension

        Returns:
            quantized: Quantized tensor of shape (B, T, N, latent_dim)
        """
        
        # Scale input to [-1, 1]
        x = torch.tanh(x)

        # Snap to hypercube
        # Scale to [0, num_bins - 1]
        x_scaled = (x + 1) / 2 * (self.num_bins - 1)

        # Round to nearest integer
        x_quantized = torch.round(x_scaled)

        # Straight-through estimator: forward uses quantized, backward uses continuous
        x_quantized = x_scaled + (x_quantized - x_scaled).detach()

        # Scale back to [-1, 1]
        x = x_quantized / (self.num_bins - 1) * 2 - 1

        return x

    def latent_to_index(self, z):
        """
        Convert quantized latent vectors to integer token indices.

        Args:
            z: Quantized latents in [-1, 1], shape (..., latent_dim)

        Returns:
            indices: Integer token IDs, shape (...)
        """
        # Convert from [-1, 1] back to bin indices [0, num_bins-1]
        bin_indices = torch.round((z + 1) / 2 * (self.num_bins - 1)).long()  # (..., latent_dim)

        indices = (bin_indices * self.powers).sum(dim=-1)  # (...)

        return indices

    def index_to_latent(self, indices):
        """
        Convert integer token indices back to quantized latent vectors.

        Args:
            indices: Integer token IDs, shape (...)

        Returns:
            z: Quantized latents in [-1, 1], shape (..., latent_dim)
        """
        bin_indices = (indices.unsqueeze(-1) // self.powers) % self.num_bins  # (..., latent_dim)

        # Convert bin indices [0, num_bins-1] back to [-1, 1]
        z = bin_indices.float() / (self.num_bins - 1) * 2 - 1

        return z