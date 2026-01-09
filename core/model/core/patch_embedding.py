import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Convert image into patches and embed them.

    Args:
        img_size: Input image size (H, W)
        context_size: The number of timesteps in each batch/video
        patch_size: Size of each patch
        in_channels: Number of input channels
        embed_dim: Embedding dimension
    """
    def __init__(self, img_size=(128, 128), context_size=16, patch_size=8, in_channels=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Calculate number of patches
        self.num_patches_h = img_size[0] // patch_size
        self.num_patches_w = img_size[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            x: (B, T, N, P) where N is number of patches, P is pixels per patch
        """
        B, T, C, H, W = x.shape

        # Reshape into patches
        # (B, T, C, H, W) -> (B, T, C, num_patches_h, patch_size, num_patches_w, patch_size)
        x = x.reshape(B, T, C, self.num_patches_h, self.patch_size, self.num_patches_w, self.patch_size)

        # Rearrange to group patches together
        # (B, T, C, num_patches_h, patch_size, num_patches_w, patch_size) -> (B, T, num_patches_h, num_patches_w, C, patch_size, patch_size)
        x = x.permute(0, 1, 3, 5, 2, 4, 6)

        # Flatten patches
        # (B, T, num_patches_h, num_patches_w, C, patch_size, patch_size) -> (B, T, num_patches_h, num_patches_w, C * patch_size * patch_size)
        x = x.reshape(B, T, self.num_patches_h, self.num_patches_w, C * self.patch_size * self.patch_size)

        # Project patch pixels to embedding dimension
        x = self.proj(x)  # (B, T, num_patches_h, num_patches_w, embed_dim)

        # Flatten num_patches_h and num_patches_w
        x = x.reshape(B, T, self.num_patches, self.embed_dim)

        return x
