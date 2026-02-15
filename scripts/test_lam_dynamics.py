import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym
import vizdoom.gymnasium_wrapper
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model.video_tokenizer import VideoTokenizer
from core.model.latent_action_model import LatentActionModel
from core.model.dynamics_model import DynamicsModel
from core.model.components.quantization import FSQ


def test(lam_checkpoint, dynamics_checkpoint, tokenizer_checkpoint='checkpoints/video_tokenizer_epoch_6.pt',
         rollout_length=100):
    sequence_length = 16
    img_size = (128, 128)
    patch_size = 8
    in_channels = 3
    embed_dim = 96
    latent_dim = 5
    latent_dim_actions = 2
    num_bins = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_patches_x = img_size[1] // patch_size
    num_patches_y = img_size[0] // patch_size
    T = sequence_length

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

    # Load LAM
    lam = LatentActionModel(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_frames=sequence_length,
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        latent_dim_actions=latent_dim_actions
    ).to(device)
    lam_ckpt = torch.load(lam_checkpoint, map_location=device)
    lam.load_state_dict(lam_ckpt['model_state_dict'])
    lam.eval()
    print(f"Loaded LAM from {lam_checkpoint}")
    print(f"  Epoch: {lam_ckpt['epoch'] + 1}, LAM loss: {lam_ckpt['lam_loss']:.6f}")

    # Load DynamicsModel
    dynamics_model = DynamicsModel(
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
        in_channels=in_channels,
        num_frames=sequence_length,
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        latent_dim_actions=latent_dim_actions,
        num_bins=num_bins
    ).to(device)
    dyn_ckpt = torch.load(dynamics_checkpoint, map_location=device)
    dynamics_model.load_state_dict(dyn_ckpt['model_state_dict'])
    dynamics_model.eval()
    print(f"Loaded DynamicsModel from {dynamics_checkpoint}")
    print(f"  Epoch: {dyn_ckpt['epoch'] + 1}, Dynamics loss: {dyn_ckpt['dynamics_loss']:.6f}")
    print()

    fsq = FSQ(latent_dim=latent_dim, num_bins=num_bins).to(device)

    # Collect a rollout from the environment
    print(f"Collecting {rollout_length}-frame rollout from ViZDoom HealthGathering...")
    env = gym.make("VizdoomHealthGathering-v1", render_mode="rgb_array")
    observation, info = env.reset()

    all_frames = []
    for step in range(rollout_length):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        frame_resized = cv2.resize(frame, img_size, interpolation=cv2.INTER_AREA)
        all_frames.append(frame_resized)
        if terminated or truncated:
            observation, info = env.reset()
    env.close()

    all_frames = np.array(all_frames)  # (rollout_length, H, W, C)
    print(f"Collected {rollout_length} frames")

    # Sample a contiguous sequence
    start_idx = np.random.randint(0, rollout_length - sequence_length + 1)
    frames = all_frames[start_idx:start_idx + sequence_length]  # (T, H, W, C)
    print(f"Using frames {start_idx} to {start_idx + sequence_length - 1}")

    frames_normalized = frames.astype(np.float32) / 255.0
    videos = torch.from_numpy(frames_normalized).permute(0, 3, 1, 2).unsqueeze(0).to(device)  # (1, T, C, H, W)

    with torch.no_grad():
        # Tokenizer latents
        latents = video_tokenizer.encoder(videos)  # (1, T, N, latent_dim)

        # LAM forward: reconstructed is frames 1..T-1 (T-1 frames)
        reconstructed, actions = lam(videos)  # reconstructed: (1, T-1, C, H, W), actions: (1, T-1, latent_dim_actions)
        lam_loss = F.mse_loss(reconstructed, videos[:, 1:])
        print(f"LAM loss: {lam_loss.item():.6f}")

        # Dynamics inference: use frames 0..T-2 as context, generate frame T-1
        # lengths[b] = index of the frame to generate = T-1
        B = 1
        null_action = torch.zeros(B, 1, latent_dim_actions, device=device)
        a_shifted = torch.cat([null_action, actions], dim=1)  # (1, T, latent_dim_actions)
        lengths = torch.full((B,), T - 1, dtype=torch.long, device=device)

        generated_latent = dynamics_model(latents.clone(), a_shifted, lengths, training=False)  # (1, N, latent_dim)

        # Decode generated latent back to pixel space
        generated_frame = video_tokenizer.decoder(generated_latent.unsqueeze(1))  # (1, 1, C, H, W)
        print(f"Generated frame shape: {generated_frame.shape}")

        # Dynamics training loss for reference
        targets = fsq.latent_to_index(latents)
        lengths_train = torch.full((B,), T - 1, dtype=torch.long, device=device)
        _, dynamics_loss = dynamics_model(latents.clone(), a_shifted, lengths_train, targets=targets, training=True)
        print(f"Dynamics loss: {dynamics_loss.item():.6f}  (random baseline: {np.log(num_bins ** latent_dim):.3f})")

    # --- Visualize ---
    # Row 0: original frames t=2..16 (videos[0, 1:])
    # Row 1: LAM reconstructed frames t=2..16
    # Row 2: dynamics-predicted frame t=17 (last column only)
    orig = np.clip(videos[0, 1:].cpu().numpy().transpose(0, 2, 3, 1), 0, 1)      # (T-1, H, W, C)
    recon = np.clip(reconstructed[0].cpu().numpy().transpose(0, 2, 3, 1), 0, 1)  # (T-1, H, W, C)
    gen = np.clip(generated_frame[0, 0].cpu().numpy().transpose(1, 2, 0), 0, 1)  # (H, W, C)

    num_cols = orig.shape[0]  # T-1 = 15
    fig, axes = plt.subplots(3, num_cols, figsize=(num_cols * 2, 6))
    fig.suptitle('Original / LAM Reconstruction / Dynamics Prediction', fontsize=12)

    for t in range(num_cols):
        axes[0, t].imshow(orig[t])
        axes[0, t].axis('off')
        axes[0, t].set_title(f't={t+2}', fontsize=7)

        axes[1, t].imshow(recon[t])
        axes[1, t].axis('off')

        axes[2, t].axis('off')

    # Dynamics predicts frame T+1 (one beyond the context window) — show in last column
    axes[2, -1].imshow(gen)
    axes[2, -1].axis('off')
    axes[2, -1].set_title(f't={T} (pred)', fontsize=7)

    axes[0, 0].set_ylabel('Original', fontsize=8)
    axes[1, 0].set_ylabel('LAM recon', fontsize=8)
    axes[2, 0].set_ylabel('Dynamics pred', fontsize=8)

    plt.tight_layout()
    out_path = 'lam_reconstruction.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved reconstruction grid to {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lam', required=True, help='Path to LAM checkpoint, e.g. checkpoints/lam_epoch_1.pt')
    parser.add_argument('--dynamics', required=True, help='Path to dynamics checkpoint, e.g. checkpoints/dynamics_epoch_1.pt')
    parser.add_argument('--tokenizer', default='checkpoints/video_tokenizer_epoch_6.pt')
    parser.add_argument('--rollout-length', type=int, default=100)
    args = parser.parse_args()
    test(args.lam, args.dynamics, args.tokenizer, args.rollout_length)
