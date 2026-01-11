import torch
import torch.nn as nn

from core.model.components.attention import SpatialAttentionBlock, TemporalAttentionBlock
from core.model.components.ffn import GeLU
from core.model.components.norm import RMSNorm

class STTransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, num_patches_x, num_patches_y):
        """
        A single Spatial-Temporal Transformer Block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_patches_x: Number of patches in x (width) dimension
            num_patches_y: Number of patches in y (height) dimension
        """

        super().__init__()

        self.spatial_attn = SpatialAttentionBlock(embed_dim, num_heads, num_patches_x, num_patches_y, embed_dim)
        self.temporal_attn = TemporalAttentionBlock(embed_dim, num_heads, num_patches_x, num_patches_y, embed_dim)
        self.ffn = GeLU(embed_dim, embed_dim * 2)

        self.norm_spatial = RMSNorm(embed_dim)
        self.norm_temporal = RMSNorm(embed_dim)
        self.norm_ffn = RMSNorm(embed_dim)

    def forward(self, x):
        """
        Forward pass through the Spatial-Temporal Transformer Block.

        Args:
            x: Input tensor of shape (B, T, N, P) where:
               B = batch size
               T = number of frames
               N = number of patches (num_patches_x * num_patches_y)
               P = embedding dimension
        Returns:
            torch.Tensor: Output tensor of shape (B, T, N, P)
        """

        # Pre-Norm: normalize before attention/FFN, then add residual
        x = x + self.spatial_attn(self.norm_spatial(x))
        x = x + self.temporal_attn(self.norm_temporal(x))
        x = x + self.ffn(self.norm_ffn(x))

        return x

class STTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_blocks, num_patches_x, num_patches_y, num_frames):
        """
        Spatial-Temporal Transformer.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_blocks: Number of transformer blocks
            num_patches_x: Number of patches in x (width) dimension
            num_patches_y: Number of patches in y (height) dimension
            num_frames: Number of frames (temporal dimension)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y
        self.num_frames = num_frames

        self.transformer_blocks = nn.ModuleList([
            STTransformerBlock(embed_dim, num_heads, num_patches_x, num_patches_y)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, T, N, P) where:
               B = batch size
               T = number of frames
               N = number of patches (num_patches_x * num_patches_y)
               P = embedding dimension

        Returns:
            torch.Tensor: Output tensor of shape (B, T, N, P)
        """

        for block in self.transformer_blocks:
            x = block(x)
            
        return x