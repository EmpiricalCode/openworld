import torch
from torch import nn

from torch.nn import Linear
from core.model.components.patch_embedding import PatchEmbedding
from core.model.components.positional_encoding import SpatioTemporalEncoding
from core.model.components.quantization import FSQ
from core.model.spatial_temporal_transformer import STTransformer, STTransformerAdaLN


class LatentActionEncoder(nn.Module):
    """Encoder that infers latent actions from video frames."""

    def __init__(self, img_size=(128, 128), patch_size=8, in_channels=3, num_frames=8, embed_dim=128, latent_dim_actions=2):
        """
        Initialize the Latent Action Encoder.

        Args:
            img_size: Input image size (H, W). Default: (128, 128)
            patch_size: Size of each patch. Default: 8
            in_channels: Number of input channels. Default: 3
            num_frames: Number of frames in input video. Default: 8
            embed_dim: Embedding dimension. Default: 128
            latent_dim_actions: Latent action dimension. Default: 2
        """
        super().__init__()

        self.patch_embedding = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        self.st_transformer = STTransformer(
            embed_dim=embed_dim,
            num_heads=8,
            num_blocks=4,
            num_patches_x=img_size[1] // patch_size,
            num_patches_y=img_size[0] // patch_size,
            num_frames=num_frames
        )
        self.positional_encoding = SpatioTemporalEncoding(
            num_patches_x=img_size[1] // patch_size,
            num_patches_y=img_size[0] // patch_size,
            num_frames=num_frames,
            embed_dim=embed_dim
        )
        self.latent_head = Linear(embed_dim, latent_dim_actions)
        self.fsq = FSQ(latent_dim=latent_dim_actions, num_bins=2)


    def forward(self, x):
        """
        Infer latent actions from video frames.

        Args:
            x: Input video tensor of shape (B, T, C, H, W) where:
               B = batch size
               T = number of frames
               C = number of channels
               H = height
               W = width

        Returns:
            Latent actions of shape (B, T-1, L) where:
            L = latent_dim
            Actions represent transitions from frame t-1 to frame t
        """
        x = self.patch_embedding(x) # (B, T, N, P)
        x = self.positional_encoding(x) # (B, T, N, P)
        x = self.st_transformer(x) # (B, T, N, P)
        x = x.mean(dim=2) # (B, T, P) - pool across patches first
        x = self.latent_head(x) # (B, T, L)
        x = self.fsq(x) # (B, T, L)
        x = x[:, 1:, :] # (B, T-1, L)

        return x


class LatentActionDecoder(nn.Module):
    """Decoder that predicts next frames given previous frames and latent actions."""

    def __init__(self, img_size=(128, 128), patch_size=8, in_channels=3, num_frames=8, embed_dim=128, latent_dim=5, latent_dim_actions=2, num_bins=4):
        """
        Initialize the Latent Action Decoder.

        Args:
            img_size: Input image size (H, W). Default: (128, 128)
            patch_size: Size of each patch. Default: 8
            in_channels: Number of input channels. Default: 3
            num_frames: Number of frames in input video. Default: 8
            embed_dim: Embedding dimension. Default: 128
            latent_dim: Latent action dimension. Default: 5
            num_bins: Number of FSQ bins for frame quantization. Default: 4
        """
        super().__init__()

        self.patch_embedding = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        self.st_transformer_1 = STTransformer(
            embed_dim=embed_dim,
            num_heads=4,
            num_blocks=4,
            num_patches_x=img_size[1] // patch_size,
            num_patches_y=img_size[0] // patch_size,
            num_frames=num_frames
        )
        self.positional_encoding = SpatioTemporalEncoding(
            num_patches_x=img_size[1] // patch_size,
            num_patches_y=img_size[0] // patch_size,
            num_frames=num_frames,
            embed_dim=embed_dim
        )
        self.st_transformer_2 = STTransformerAdaLN(
            embed_dim=embed_dim,
            num_heads=4,
            num_blocks=4,
            num_patches_x=img_size[1] // patch_size,
            num_patches_y=img_size[0] // patch_size,
            num_frames=num_frames,
            cond_dim=latent_dim_actions
        )

        self.num_patches_x = img_size[1] // patch_size
        self.num_patches_y = img_size[0] // patch_size
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        self.latent_head = Linear(embed_dim, latent_dim)
        self.fsq = FSQ(latent_dim=latent_dim, num_bins=num_bins)
        self.embed_head_frames = Linear(latent_dim, embed_dim)
        self.unembedding_head = Linear(embed_dim, patch_size * patch_size * in_channels)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

    def forward(self, x, actions):
        """
        Predict next frames given previous frames and latent actions.

        Args:
            x: Input video tensor of shape (B, T, C, H, W) where:
               B = batch size
               T = number of frames (typically T-1 from original video)
               C = number of channels
               H = height
               W = width
            actions: Latent actions of shape (B, T, L) where:
               L = latent_dim
               Each action conditions the prediction of the corresponding frame

        Returns:
            reconstructed: Reconstructed video of shape (B, T, C, H, W)
            patch_mask: Boolean mask of shape (B, T, N) where True = masked,
                        or None if not training.
        """
        B, T, _, _, _ = x.shape

        # Images to tokens
        x = self.patch_embedding(x)  # (B, T, N, P)
        x = self.positional_encoding(x)  # (B, T, N, P)

        # During training, mask 25-75% of patches in frames 1..T-1
        # Each frame position gets its own independently sampled mask ratio
        patch_mask = None
        N = x.shape[2]
        # Sample a separate mask ratio for each frame position in 1..T-1: (T-1,)
        frame_ratios = 0.25 + torch.rand(T - 1, device=x.device) * 0.5  # (T-1,)
        rand = torch.rand(B, T - 1, N, device=x.device)
        frame_mask = rand < frame_ratios.unsqueeze(0).unsqueeze(-1)  # (B, T-1, N)
        no_mask = torch.zeros(B, 1, N, dtype=torch.bool, device=x.device)
        patch_mask = torch.cat([no_mask, frame_mask], dim=1)  # (B, T, N)
        x = torch.where(patch_mask.unsqueeze(-1), self.mask_token.expand(B, T, N, -1), x)

        x = self.st_transformer_1(x)  # (B, T, N, P)
        x = self.latent_head(x)     # (B, T, N, L)
        x = self.fsq(x)             # (B, T, N, L)

        # Project back to embedding space
        x = self.embed_head_frames(x)  # (B, T, N, P)

        # PE + ST-Transformer with AdaLN conditioning
        # Actions passed directly — AdaLN.proj handles the projection
        x = self.positional_encoding(x)  # (B, T, N, P)
        x = self.st_transformer_2(x, actions)  # (B, T, N, P)

        # Project back to image space
        x = x.reshape(B, T, self.num_patches_y, self.num_patches_x, self.embed_dim) # (B, T, H//p, W//p, P)
        x = self.unembedding_head(x)  # (B, T, H//p, W//p, C*p*p)
        x = x.reshape(B, T, self.num_patches_y, self.num_patches_x, self.in_channels, self.patch_size, self.patch_size)  # (B, T, H//p, W//p, C, p, p)
        x = x.permute(0, 1, 4, 2, 5, 3, 6)  # (B, T, C, H//p, W//p, p, p)
        x = x.reshape(B, T, self.in_channels, self.img_size[0], self.img_size[1])  # (B, T, C, H, W)

        return x


class LatentActionModel(nn.Module):
    """
    Latent Action Model for unsupervised action discovery.

    Learns discrete latent actions by reconstructing video frames through
    an information bottleneck. The encoder infers actions from frame transitions,
    and the decoder predicts next frames conditioned on previous frames and actions.
    """

    def __init__(self, img_size=(128, 128), patch_size=8, in_channels=3, num_frames=8, embed_dim=128, latent_dim=5, latent_dim_actions=2, num_bins=4):
        """
        Initialize the Latent Action Model.

        Args:
            img_size: Input image size (H, W). Default: (128, 128)
            patch_size: Size of each patch. Default: 8
            in_channels: Number of input channels. Default: 3
            num_frames: Number of frames in input video. Default: 8
            embed_dim: Embedding dimension. Default: 128
            latent_dim: Latent action dimension. Default: 5
            num_bins: Number of FSQ bins for frame quantization. Default: 4
        """
        super().__init__()

        self.encoder = LatentActionEncoder(img_size, patch_size, in_channels, num_frames, embed_dim, latent_dim_actions)
        self.decoder = LatentActionDecoder(img_size, patch_size, in_channels, num_frames-1, embed_dim, latent_dim, latent_dim_actions, num_bins)

    def forward(self, x):
        """
        Forward pass for latent action model.
        Caller is responsible for padding x to num_frames frames.

        Args:
            x: Input video tensor of shape (B, T, C, H, W) where:
               B = batch size
               T = num_frames (caller-padded)
               C = number of channels
               H = height
               W = width

        Returns:
            reconstructed: Reconstructed video of shape (B, T-1, C, H, W)
            actions: Inferred latent actions of shape (B, T-1, L)
        """
        # Encoder: infer actions from all frames
        actions = self.encoder(x)  # (B, T-1, L)

        # Decoder: reconstruct frames from initial frames + actions
        # Use frames 0 to T-2 to predict frames 1 to T-1
        x_input = x[:, :-1, :, :, :]  # (B, T-1, C, H, W)
        reconstructed = self.decoder(x_input, actions)  # (B, T-1, C, H, W)

        return reconstructed, actions
