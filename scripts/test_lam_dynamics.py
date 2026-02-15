import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model.video_tokenizer import VideoTokenizer
from core.model.latent_action_model import LatentActionModel
from core.model.dynamics_model import DynamicsModel
from core.model.components.quantization import FSQ


def test(lam_checkpoint, dynamics_checkpoint, tokenizer_checkpoint='checkpoints/video_tokenizer_epoch_6.pt'):
    sequence_length = 16
    img_size = (128, 128)
    patch_size = 8
    in_channels = 3
    embed_dim = 96
    latent_dim = 5
    num_bins = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_patches_x = img_size[1] // patch_size
    num_patches_y = img_size[0] // patch_size
    T = sequence_length
    B = 2  # small batch for testing

    # Load frozen video tokenizer
    video_tokenizer = VideoTokenizer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_frames=sequence_length,
        embed_dim=128,
        latent_dim=latent_dim
    ).to(device)
    tok_ckpt = torch.load(tokenizer_checkpoint, map_location=device)
    video_tokenizer.load_state_dict(tok_ckpt['model_state_dict'], strict=False)
    video_tokenizer.eval()
    print(f"Loaded VideoTokenizer from {tokenizer_checkpoint}")

    # Load LAM and DynamicsModel
    lam = LatentActionModel(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_frames=sequence_length,
        embed_dim=embed_dim,
        latent_dim=latent_dim
    ).to(device)

    dynamics_model = DynamicsModel(
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
        in_channels=in_channels,
        num_frames=sequence_length,
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        num_bins=num_bins
    ).to(device)

    lam_ckpt = torch.load(lam_checkpoint, map_location=device)
    lam.load_state_dict(lam_ckpt['model_state_dict'])
    lam.eval()
    print(f"Loaded LAM from {lam_checkpoint}")
    print(f"  Checkpoint epoch: {lam_ckpt['epoch'] + 1}")
    print(f"  Saved LAM loss: {lam_ckpt['lam_loss']:.6f}")

    dyn_ckpt = torch.load(dynamics_checkpoint, map_location=device)
    dynamics_model.load_state_dict(dyn_ckpt['model_state_dict'])
    dynamics_model.eval()
    print(f"Loaded DynamicsModel from {dynamics_checkpoint}")
    print(f"  Checkpoint epoch: {dyn_ckpt['epoch'] + 1}")
    print(f"  Saved Dynamics loss: {dyn_ckpt['dynamics_loss']:.6f}")
    print()

    fsq = FSQ(latent_dim=latent_dim, num_bins=num_bins).to(device)

    # Random test batch
    videos = torch.rand(B, T, in_channels, *img_size, device=device)

    with torch.no_grad():
        # Tokenizer latents
        latents = video_tokenizer.encoder(videos)  # (B, T, N, latent_dim)
        print(f"Latents shape: {latents.shape}")

        # LAM forward
        reconstructed, actions = lam(videos)
        lam_loss = F.mse_loss(reconstructed, videos[:, 1:])
        print(f"Reconstructed shape: {reconstructed.shape}")
        print(f"Actions shape: {actions.shape}")
        print(f"LAM loss: {lam_loss.item():.6f}")

        # Dynamics forward
        null_action = torch.zeros(B, 1, latent_dim, device=device)
        a_shifted = torch.cat([null_action, actions], dim=1)
        lengths = torch.full((B,), T, dtype=torch.long, device=device)
        targets = fsq.latent_to_index(latents)

        _, dynamics_loss = dynamics_model(latents, a_shifted, lengths, targets=targets, training=True)
        print(f"Dynamics loss: {dynamics_loss.item():.6f}  (random baseline: {np.log(num_bins ** latent_dim):.3f})")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('lam_checkpoint', help='Path to LAM checkpoint, e.g. checkpoints/lam_epoch_1.pt')
    parser.add_argument('dynamics_checkpoint', help='Path to dynamics checkpoint, e.g. checkpoints/dynamics_epoch_1.pt')
    parser.add_argument('--tokenizer', default='checkpoints/video_tokenizer_epoch_6.pt')
    args = parser.parse_args()
    test(args.lam_checkpoint, args.dynamics_checkpoint, args.tokenizer)
