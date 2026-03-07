import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Linear
from core.model.components.positional_encoding import SpatioTemporalEncoding
from core.model.components.quantization import FSQ
from core.model.spatial_temporal_transformer import STTransformer, STTransformerAdaLN


class DynamicsModel(nn.Module):

    def __init__(self, num_patches_x, num_patches_y, in_channels=3, num_frames=8, embed_dim=128, latent_dim=5, latent_dim_actions=2, num_iterations=25, num_bins=4, mask_schedule='cosine', num_blocks=8, num_heads=8):
        """
        Dynamics Model - decoder-only MaskGIT transformer for video token prediction.
        Takes continuous latent tokens + action embeddings, predicts discrete token IDs.

        Args:
            num_patches_x: Number of patches in x dimension
            num_patches_y: Number of patches in y dimension
            patch_size: Size of each patch. Default: 8
            in_channels: Number of input channels. Default: 3
            num_frames: Number of frames to process. Default: 8
            embed_dim: Embedding dimension for transformer. Default: 128
            latent_dim: Dimension of the latent representation (codebook dimension). Default: 5
            latent_dim_actions: Dimension of the latent action representation. Default: 2
            num_iterations: Number of MaskGIT decoding iterations at inference. Default: 25
            num_bins: Number of bins per latent dimension for quantization. Default: 4
            mask_schedule: Masking schedule function ('linear', 'cosine'). Default: 'cosine'
        """
        super().__init__()

        # Store configuration parameters
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y
        self.num_patches = self.num_patches_x * self.num_patches_y
        self.num_iterations = num_iterations
        self.num_bins = num_bins
        self.codebook_size = num_bins ** latent_dim
        self.mask_schedule = mask_schedule

        # Learnable mask token (in latent space)
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, latent_dim) * 0.02)

        self.embed_head_frames = Linear(latent_dim, embed_dim)

        self.positional_encoding = SpatioTemporalEncoding(
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
            num_frames=num_frames,
            embed_dim=embed_dim
        )

        # ST-Transformer with AdaLN action conditioning at every block
        # Actions passed directly (no pre-embedding) — AdaLN.proj handles the projection
        self.st_transformer = STTransformerAdaLN(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
            num_frames=num_frames,
            cond_dim=latent_dim_actions
        )

        # Just used for converting latent to index and index to latent
        self.fsq = FSQ(latent_dim=latent_dim, num_bins=num_bins)

        # Prediction head with GeLU activation: latent_dim -> codebook logits
        self.prediction_head = Linear(embed_dim, self.codebook_size)

    def forward(self, x, a, lengths, targets=None, training=True):
        """
        Predict next latent state given context frames and actions using MaskGIT iterative decoding.
        x, a, and targets are pre-padded to the same T by the caller; lengths[b] gives the real frame
        count for sample b.

        Args:
            x: Latent tokens of shape (B, T, N, latent_dim). Zeros beyond lengths[b].
            a: Actions of shape (B, T, latent_dim). Zeros beyond lengths[b].
            lengths: Real frame count per sample, shape (B,).
            targets: Ground truth token IDs of shape (B, T, N). Required if training=True.
            training: If True, runs masked training. If False, runs iterative decoding.

        For training:
            - Random masking is applied across all frames.
            - Loss is only computed on frames t < lengths[b] (valid frames).

        For inference:
            - The frame at x[b, lengths[b]] is filled with the mask token and iteratively decoded.
            - Returns the generated frame for each sample.

        Returns:
            Training: (x_predict, loss) where x_predict is (B, T, N, codebook_size)
            Inference: Generated frame for each sample of shape (B, N, latent_dim)
        """

        B, T, N, _ = x.shape

        # Training mode: random masking
        if training:
            # Mask ratio sampled uniformly from [0.5, 1.0]
            mask_ratio = 0.5 + torch.rand(B, device=x.device) * 0.5  # (B,)

            # Random mask for all frames
            mask = torch.rand(B, T, N, device=x.device) < mask_ratio.reshape(B, 1, 1)  # (B, T, N)

            # Guarantee one unmasked anchor per (B, N) — random timestep, not always frame 0
            anchor_t = torch.randint(0, T, (B, N), device=x.device)  # (B, N)
            mask[
                torch.arange(B, device=x.device).unsqueeze(1),
                anchor_t,
                torch.arange(N, device=x.device).unsqueeze(0)
            ] = False

            mask_token = self.mask_token.expand(B, T, N, -1)  # (B, T, N, latent_dim)
            x_masked = torch.where(mask.unsqueeze(-1), mask_token, x)  # (B, T, N, latent_dim)

            x_embed_frames = self.embed_head_frames(x_masked)  # (B, T, N, embed_dim)

            x_pos = self.positional_encoding(x_embed_frames)  # (B, T, N, embed_dim)

            x_transformed = self.st_transformer(x_pos, a)  # (B, T, N, embed_dim)

            x_predict = self.prediction_head(x_transformed)  # (B, T, N, num_bins^latent_dim)

            # Restrict loss to valid (non-padded) frames
            t_idx = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
            valid = t_idx < lengths.unsqueeze(1)  # (B, T)
            loss_mask = mask & valid.unsqueeze(-1)  # (B, T, N)

            loss = F.cross_entropy(x_predict[loss_mask], targets[loss_mask])
            
            return x_predict, loss

        # Inference mode: iterative MaskGIT decoding
        else:
            batch_idx = torch.arange(B, device=x.device)

            # Place mask token at each sample's generation position (lengths[b])
            x[batch_idx, lengths] = self.mask_token.reshape(1, 1, self.latent_dim).expand(B, N, -1)

            # Keeping track of which patches have already been unmasked in the generated frame
            unmasked = torch.zeros(B, N, device=x.device)

            for i in range(1, 1+self.num_iterations):

                x_embed_frames = self.embed_head_frames(x)  # (B, T, N, embed_dim)

                x_pos = self.positional_encoding(x_embed_frames)  # (B, T, N, embed_dim)

                x_transformed = self.st_transformer(x_pos, a)  # (B, T, N, embed_dim)

                x_predict = self.prediction_head(x_transformed)  # (B, T, N, num_bins^latent_dim)
                x_probs = F.softmax(x_predict, dim=-1)  # (B, T, N, num_bins^latent_dim)

                # Sample predicted tokens from the distribution
                pred_indices = torch.multinomial(x_probs.reshape(B * T * N, -1), num_samples=1).reshape(B, T, N)  # (B, T, N)

                # Convert predicted indices back to latent vectors and get their probabilities
                pred_probs = x_probs.gather(-1, pred_indices.unsqueeze(-1)).squeeze(-1)  # (B, T, N)
                pred_tokens = self.fsq.index_to_latent(pred_indices)  # (B, T, N, latent_dim)

                # Get mask ratio for this iteration based on schedule
                if self.mask_schedule == 'cosine':
                    mask_ratio = np.cos(np.pi * i / (2 * self.num_iterations))  # Cosine decay
                elif self.mask_schedule == 'linear':
                    mask_ratio = 1 - i / self.num_iterations  # Linear decay
                else:
                    raise ValueError("Invalid mask_schedule. Choose 'cosine' or 'linear'.")

                # Operate on each sample's generation frame at lengths[b]
                new_frame_probs = pred_probs[batch_idx, lengths].clone()  # (B, N)
                new_frame_probs.masked_fill_(unmasked.bool(), 1.0)  # Already unmasked patches persist
                num_keep = N - int(mask_ratio * N)

                _, top_indices = torch.topk(new_frame_probs, num_keep, dim=-1)  # (B, num_keep)

                new_mask = torch.zeros(B, N, device=x.device)
                new_mask.scatter_(1, top_indices, 1.0)
                new_mask = new_mask * (1 - unmasked)  # exclude already unmasked

                new_frame_preds = pred_tokens[batch_idx, lengths]  # (B, N, latent_dim)
                current_frame = x[batch_idx, lengths]  # (B, N, latent_dim)
                x[batch_idx, lengths] = torch.where(new_mask.unsqueeze(-1).bool(), new_frame_preds, current_frame)

                unmasked = (unmasked + new_mask).clamp(max=1.0)

            # Return the generated frame for each sample
            return x[batch_idx, lengths]  # (B, N, latent_dim)
            