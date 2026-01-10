import torch
import torch.nn as nn


class SpatialAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_patches_x, num_patches_y, kq_dim):
        """
        Spatial Attention Block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_patches_x: Number of patches in x (width) dimension
            num_patches_y: Number of patches in y (height) dimension
            kq_dim: Dimension of key and query projections
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y
        self.kq_dim = kq_dim

        self.q_proj = nn.Linear(embed_dim, kq_dim)
        self.k_proj = nn.Linear(embed_dim, kq_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass. Each token within each frame attends to all other tokens in the same frame.

        Args:
            x: Input tensor of shape (B, T, N, P) where:
               B = batch size
               T = number of frames
               N = number of patches (num_patches_x * num_patches_y)
               P = embedding dimension

        Returns:
            torch.Tensor: Output tensor of shape (B, T, N, P)
        """
        
        B, T, N, P = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape and permute to get multi-head format
        kq_head_dim = self.kq_dim // self.num_heads
        v_head_dim = self.embed_dim // self.num_heads

        q = q.reshape(B, T, N, self.num_heads, kq_head_dim).permute(0, 3, 1, 2, 4)
        k = k.reshape(B, T, N, self.num_heads, kq_head_dim).permute(0, 3, 1, 2, 4)
        v = v.reshape(B, T, N, self.num_heads, v_head_dim).permute(0, 3, 1, 2, 4)

        # Result: (B, num_heads, T, N, head_dim)

        # Compute attention scores: Q @ K^T
        scores = q @ k.transpose(-2, -1)
        scores = scores / (kq_head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)

        # Result: (B, num_heads, T, N, N)

        # Apply attention to values
        attn_output = attn_weights @ v

        # Result: (B, num_heads, T, N, head_dim)

        # Reshape back: (B, num_heads, T, N, head_dim) -> (B, T, N, embed_dim)
        attn_output = attn_output.permute(0, 2, 3, 1, 4)  # (B, T, N, num_heads, head_dim)
        attn_output = attn_output.reshape(B, T, N, self.embed_dim) # Merge heads

        return attn_output

class TemporalAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_patches_x, num_patches_y, kq_dim):
        """
        Temporal Attention Block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_patches_x: Number of patches in x (width) dimension
            num_patches_y: Number of patches in y (height) dimension
            kq_dim: Dimension of key and query projections
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y
        self.kq_dim = kq_dim

        self.q_proj = nn.Linear(embed_dim, kq_dim)
        self.k_proj = nn.Linear(embed_dim, kq_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass. Each patch attends across time (temporal tubes).

        Args:
            x: Input tensor of shape (B, T, N, P) where:
               B = batch size
               T = number of frames
               N = number of patches (num_patches_x * num_patches_y)
               P = embedding dimension

        Returns:
            torch.Tensor: Output tensor of shape (B, T, N, P)
        """
        B, T, N, P = x.shape

        # Reshape to tubes BEFORE projection: (B, T, N, P) -> (B, N, T, P)
        x = x.permute(0, 2, 1, 3)  # Now each N is a temporal sequence

        # Project Q, K, V
        q = self.q_proj(x)  # (B, N, T, kq_dim)
        k = self.k_proj(x)  # (B, N, T, kq_dim)
        v = self.v_proj(x)  # (B, N, T, embed_dim)

        # Reshape and permute to get multi-head format
        kq_head_dim = self.kq_dim // self.num_heads
        v_head_dim = self.embed_dim // self.num_heads

        q = q.reshape(B, N, T, self.num_heads, kq_head_dim).permute(0, 3, 1, 2, 4)
        k = k.reshape(B, N, T, self.num_heads, kq_head_dim).permute(0, 3, 1, 2, 4)
        v = v.reshape(B, N, T, self.num_heads, v_head_dim).permute(0, 3, 1, 2, 4)
        
        # Result: (B, num_heads, N, T, head_dim)

        # Compute attention scores: Q @ K^T
        # (B, num_heads, N, T, kq_head_dim) @ (B, num_heads, N, kq_head_dim, T)
        # -> (B, num_heads, N, T, T)
        scores = q @ k.transpose(-2, -1)

        # Scale by sqrt(kq_head_dim)
        scores = scores / (kq_head_dim ** 0.5)

        # Apply causal mask: prevent attending to future tokens
        # Create a lower triangular mask: [[1, 0, 0], [1, 1, 0], [1, 1, 1]] for T=3
        causal_mask = torch.tril(torch.ones(T, T, device=scores.device, dtype=torch.bool))
        # Set future positions to -inf so they become 0 after softmax
        scores = scores.masked_fill(~causal_mask, float('-inf'))

        # Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)  # (B, num_heads, N, T, T)

        # Apply attention to values: (B, num_heads, N, T, T) @ (B, num_heads, N, T, v_head_dim)
        # -> (B, num_heads, N, T, v_head_dim)
        attn_output = attn_weights @ v

        # Reshape back: (B, num_heads, N, T, head_dim) -> (B, T, N, embed_dim)
        attn_output = attn_output.permute(0, 3, 2, 1, 4)  # (B, T, N, num_heads, head_dim)
        attn_output = attn_output.reshape(B, T, N, self.embed_dim)  # Merge heads

        return attn_output
