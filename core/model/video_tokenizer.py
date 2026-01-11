from torch import nn
from core.model.components.patch_embedding import PatchEmbedding
from core.model.components.positional_encoding import SpatioTemporalEncoding
from core.model.components.quantization import FSQ
from core.model.spatial_temporal_transformer import STTransformer

class VideoTokenizer(nn.Module):

    def __init__(self, img_size=(128, 128), patch_size=8, in_channels=3, num_frames=8, embed_dim=128, latent_dim=5):
        """
        Video Tokenizer that converts videos into patch embeddings.

        Args:
            img_size: Input image size (H, W). Default: (128, 128)
            patch_size: Size of each patch. Default: 8
            in_channels: Number of input channels. Default: 3
            embed_dim: Embedding dimension. Default: 128
        """
        super().__init__()

        # Store configuration parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_patches_x = img_size[1] // patch_size
        self.num_patches_y = img_size[0] // patch_size
        self.num_patches = self.num_patches_x * self.num_patches_y

        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.positional_encoding = SpatioTemporalEncoding(
            num_patches_x=img_size[1] // patch_size,
            num_patches_y=img_size[0] // patch_size,
            num_frames=num_frames,
            embed_dim=embed_dim
        )
        self.st_transformer_encoder = STTransformer(
            embed_dim=embed_dim,
            num_heads=4,
            num_blocks=4,
            num_patches_x=img_size[1] // patch_size,
            num_patches_y=img_size[0] // patch_size,
            num_frames=num_frames
        )
        self.st_transformer_decoder = STTransformer(
            embed_dim=embed_dim,
            num_heads=4,
            num_blocks=4,
            num_patches_x=img_size[1] // patch_size,
            num_patches_y=img_size[0] // patch_size,
            num_frames=num_frames
        )
        self.latent_head = nn.Linear(embed_dim, latent_dim)
        self.embedding_head = nn.Linear(latent_dim, embed_dim)
        self.FSQ = FSQ(latent_dim=latent_dim, num_bins=4)
        self.un_proj = nn.Linear(embed_dim, patch_size * patch_size * in_channels)

    def forward(self, x):
        """
        Forward pass to tokenize video input.

        Args:
            x: Input video tensor of shape (B, C, T, H, W) where:
               B = batch size
               T = number of channels
               C = number of frames
               H = height
               W = width

        Returns:
            torch.Tensor: Tokenized output of shape (B, T, N, P) where:
                          N = number of patches (num_patches_x * num_patches_y)
                          P = embedding dimension
        """

        # ENCODER

        B, T, C, H, W = x.shape

        # Patch Embedding
        x = self.patch_embedding(x)  # Shape: (B, T, N, P)

        # Add Positional Encoding
        x = self.positional_encoding(x)  # Shape: (B, T, N, P)

        # Spatial-Temporal Transformer
        x = self.st_transformer_encoder(x)  # Shape: (B, T, N, P)

        # Project to latent dimension
        x = self.latent_head(x)  # Shape: (B, T, N, latent_dim)

        # QUANTIZATION
        x = self.FSQ(x)  # Shape: (B, T, N, latent_dim)

        # DECODER

        # Project back to embedding dimension
        x = self.embedding_head(x)  # Shape: (B, T, N, P)

        # Spatial-Temporal Transformer
        x = self.st_transformer_decoder(x)  # Shape: (B, T, N, P)
        
        # Reshape back to matrices of patch tokens
        x = x.reshape(B, T, self.num_patches_y, self.num_patches_x, self.embed_dim)

        # Un-project to patch pixels
        x = self.un_proj(x)  # Shape: (B, T, num_patches_y, num_patches_x, patch_size * patch_size * in_channels)

        # Re-shape to original video dimensions
        x = x.reshape(B, T, self.num_patches_y, self.num_patches_x, self.in_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6)  # (B, T, C, num_patches_y, patch_size, num_patches_x, patch_size)
        x = x.reshape(B, T, C, H, W)  # (B, T, C, H, W)

        return x