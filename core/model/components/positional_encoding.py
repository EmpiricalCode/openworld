import torch
import torch.nn as nn
import math


class SinusoidalEncoding(nn.Module):
    def __init__(self, length, embed_dim, base=10000.0):
        """
        Creates 1D sinusoidal positional encodings.

        Args:
            length: Maximum sequence length
            embed_dim: Embedding dimension
            base: Base frequency for encoding
        """
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim

        pe = self._create_positional_encoding(base)
        self.register_buffer('pe', pe)

    def _create_positional_encoding(self, base=10000.0):
        """
        Creates sinusoidal positional encodings.

        Returns:
            torch.Tensor: Positional encoding of shape (length, embed_dim)
        """
        pe = torch.zeros(self.length, self.embed_dim)
        position = torch.arange(0, self.length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() *
                            (-math.log(base) / self.embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, x):
        """
        Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)

        Returns:
            torch.Tensor: Input with positional encoding added
        """
        pe = self.pe[:x.size(1), :].to(x.device)
        return x + pe

class SpatioTemporalEncoding(nn.Module):
    def __init__(self, num_patches_x, num_patches_y, num_frames, embed_dim,
                 temporal_base=100.0, spatial_base=10000.0):
        """
        Creates spatial-temporal positional encodings by concatenating x, y, and t encodings.

        Args:
            num_patches_x: Number of patches in x (width) dimension
            num_patches_y: Number of patches in y (height) dimension
            num_frames: Number of frames (temporal dimension)
            embed_dim: Embedding dimension (will be split as: x_dim = embed_dim//3, y_dim = embed_dim//3, t_dim = remaining)
            temporal_base: Base frequency for temporal encoding (lower = slower change)
            spatial_base: Base frequency for spatial encoding
        """
        super().__init__()
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # Split embedding dimension among x, y, and t
        self.x_dim = embed_dim // 3
        self.y_dim = embed_dim // 3
        self.t_dim = embed_dim - self.x_dim - self.y_dim

        # Create positional encodings for each dimension with their respective sizes
        pe_x = self._create_positional_encoding(num_patches_x, self.x_dim, spatial_base)
        pe_y = self._create_positional_encoding(num_patches_y, self.y_dim, spatial_base)
        pe_t = self._create_positional_encoding(num_frames, self.t_dim, temporal_base)

        # Register as buffers
        self.register_buffer('pe_x', pe_x)
        self.register_buffer('pe_y', pe_y)
        self.register_buffer('pe_t', pe_t)

    def _create_positional_encoding(self, length, embed_dim, base=10000.0):
        """
        Creates sinusoidal positional encodings for a single dimension.

        Args:
            length: Number of positions
            embed_dim: Embedding dimension
            base: Base frequency for encoding

        Returns:
            torch.Tensor: Positional encoding of shape (length, embed_dim)
        """
        pe = torch.zeros(length, embed_dim)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                            (-math.log(base) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, x):
        """
        Add spatial-temporal positional encoding to input tensor by concatenating x, y, t encodings.

        Args:
            x: Input tensor of shape (B, T, N, P) where:
               B = batch size
               T = number of frames (temporal)
               N = number of patches (num_patches_x * num_patches_y)
               P = embedding dimension

        Returns:
            torch.Tensor: Input with positional encoding added, shape (B, T, N, P)
        """
        B, T, N, P = x.shape

        # Create spatial encoding grid
        # x positions: [0, 1, 2, ..., num_patches_x-1] repeated for each row
        # y positions: [0, 0, 0, ..., 1, 1, 1, ..., num_patches_y-1, ...]
        x_indices = torch.arange(self.num_patches_x, device=x.device).repeat(self.num_patches_y)[:N]
        y_indices = torch.arange(self.num_patches_y, device=x.device).repeat_interleave(self.num_patches_x)[:N]

        # Gather positional encodings for each spatial position
        pe_x_spatial = self.pe_x[x_indices].to(x.device)  # Shape: (N, x_dim)
        pe_y_spatial = self.pe_y[y_indices].to(x.device)  # Shape: (N, y_dim)

        # Concatenate x and y encodings: (N, x_dim + y_dim)
        pe_spatial = torch.cat([pe_x_spatial, pe_y_spatial], dim=-1)  # Shape: (N, x_dim + y_dim)

        # Expand to include batch and temporal dimensions: (N, x_dim + y_dim) -> (B, T, N, x_dim + y_dim)
        pe_spatial = pe_spatial.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

        # Temporal encoding: (T, t_dim) -> (B, T, 1, t_dim)
        pe_t = self.pe_t[:T, :].unsqueeze(0).unsqueeze(2).expand(B, -1, N, -1).to(x.device)

        # Concatenate spatial and temporal: (B, T, N, x_dim + y_dim + t_dim) = (B, T, N, P)
        pe_combined = torch.cat([pe_spatial, pe_t], dim=-1)

        # Add to input: (B, T, N, P) + (B, T, N, P) -> (B, T, N, P)
        return x + pe_combined

class RoPE:
    pass  # TODO: Implement Rotary Positional Encoding (RoPE)