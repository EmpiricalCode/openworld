"""
Test dynamics model by sampling a 16-frame sequence from the dataset,
then autoregressively predicting each frame:
  frame 0           -> predict frame 1
  frames 0,1        -> predict frame 2
  ...
  frames 0..14      -> predict frame 15

Saves a grid: row 0 = ground truth, row 1 = dynamics predictions.
"""
import torch
import numpy as np
import sys
import os
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model.video_tokenizer import VideoTokenizer
from core.model.latent_action_model import LatentActionModel
from core.model.dynamics_model import DynamicsModel

# Empirical mapping from game action index to LAM token index
GAME_ACTION_TO_LAM_TOKEN = {0: 6, 1: 5, 2: 0, 3: 3}
ACTION_NAMES = {0: 'noop', 1: 'forward', 2: 'right', 3: 'left'}


def test(dynamics_checkpoint, tokenizer_checkpoint, lam_checkpoint,
         h5_path='data/vizdoom_healthgathering/vizdoom_healthgathering_dqn.h5',
         seed=13, out_path='dynamics_test.png'):

    sequence_length = 16
    img_size = (64, 64)
    patch_size = 4
    in_channels = 3
    tokenizer_embed_dim = 128
    tokenizer_latent_dim = 5
    tokenizer_num_bins = 4
    lam_embed_dim = 128
    lam_latent_dim_actions = 3
    dynamics_embed_dim = 264

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_patches_x = img_size[1] // patch_size
    num_patches_y = img_size[0] // patch_size

    # Load VideoTokenizer
    video_tokenizer = VideoTokenizer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_frames=sequence_length,
        embed_dim=tokenizer_embed_dim,
        latent_dim=tokenizer_latent_dim
    ).to(device)
    ckpt = torch.load(tokenizer_checkpoint, map_location=device)
    video_tokenizer.load_state_dict(ckpt['model_state_dict'])
    video_tokenizer.eval()
    print(f"Loaded VideoTokenizer from {tokenizer_checkpoint}")

    # Load LAM
    lam = LatentActionModel(
        img_size=img_size,
        patch_size=8,
        in_channels=in_channels,
        num_frames=sequence_length,
        embed_dim=lam_embed_dim,
        latent_dim=tokenizer_latent_dim,
        latent_dim_actions=lam_latent_dim_actions,
        num_bins=tokenizer_num_bins
    ).to(device)
    ckpt = torch.load(lam_checkpoint, map_location=device)
    lam.load_state_dict(ckpt['model_state_dict'])
    lam.eval()
    print(f"Loaded LAM from {lam_checkpoint}")

    # Load DynamicsModel
    dynamics_model = DynamicsModel(
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
        in_channels=in_channels,
        num_frames=sequence_length,
        embed_dim=dynamics_embed_dim,
        latent_dim=tokenizer_latent_dim,
        latent_dim_actions=lam_latent_dim_actions,
        num_bins=tokenizer_num_bins
    ).to(device)
    ckpt = torch.load(dynamics_checkpoint, map_location=device)
    dynamics_model.load_state_dict(ckpt['model_state_dict'])
    dynamics_model.eval()
    print(f"Loaded DynamicsModel from {dynamics_checkpoint}")
    print(f"  Epoch: {ckpt['epoch'] + 1}, Dynamics loss: {ckpt['dynamics_loss']:.6f}")
    print()

    # Load dataset and sample a sequence
    print(f"Loading dataset from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        frames_all  = f['frames'][:]   # (N, H, W, C) uint8
        actions_all = f['actions'][:]  # (N,) int
        dones_all   = f['dones'][:]

    total_frames = frames_all.shape[0]

    # Find valid sequence start indices (no episode boundary crossings)
    episode_ends = np.where(dones_all)[0]
    episode_starts = np.concatenate([[0], episode_ends[:-1] + 1]) if len(episode_ends) > 0 else [0]
    episode_ends = np.concatenate([episode_ends, [total_frames - 1]])

    valid_indices = []
    for start, end in zip(episode_starts, episode_ends):
        if end - start + 1 >= sequence_length:
            for i in range(start, end - sequence_length + 2):
                valid_indices.append(i)

    rng = np.random.default_rng(seed)
    start_idx = valid_indices[rng.integers(0, len(valid_indices))]
    frames = frames_all[start_idx:start_idx + sequence_length]  # (T, H, W, C)
    print(f"Sampled sequence starting at frame {start_idx}")

    # Normalize and build tensors at both resolutions
    frames_f = frames.astype(np.float32) / 255.0

    import cv2
    frames_64 = np.stack([cv2.resize(f, img_size, interpolation=cv2.INTER_AREA) for f in frames_f])
    videos = torch.from_numpy(frames_64).permute(0, 3, 1, 2).unsqueeze(0).to(device)  # (1, T, C, 64, 64)

    with torch.no_grad():
        # Get all ground-truth latents at once
        latents_gt = video_tokenizer.encoder(videos)  # (1, T, N, latent_dim)

        # Get actions from LAM
        _, actions = lam(videos)  # (1, T-1, latent_dim_actions)
        null_action = torch.zeros(1, 1, lam_latent_dim_actions, device=device)
        a_shifted = torch.cat([null_action, actions], dim=1)  # (1, T, latent_dim_actions)

        # Label each transition with the game action name from the dataset
        game_actions = actions_all[start_idx:start_idx + sequence_length - 1]  # (T-1,)
        action_labels = [ACTION_NAMES[int(a)] for a in game_actions]

        T = sequence_length
        N = latents_gt.shape[2]

        # Teacher-forced prediction: always use ground-truth context
        predicted_latents = []  # will hold T-1 predicted frames (indices 1..15)

        for gen_idx in range(1, T):
            # Use all ground-truth frames as context, zero out gen_idx onward
            x_ctx = latents_gt.clone()
            x_ctx[0, gen_idx:] = 0.0

            lengths = torch.tensor([gen_idx], dtype=torch.long, device=device)

            pred = dynamics_model(x_ctx, a_shifted, lengths, training=False)  # (1, N, latent_dim)
            predicted_latents.append(pred)

            print(f"  Generated frame {gen_idx}/{T-1}")

        # Decode all predicted latents to pixels
        predicted_pixels = []
        for pred_lat in predicted_latents:
            px = video_tokenizer.decoder(pred_lat.unsqueeze(1))  # (1, 1, C, H, W)
            predicted_pixels.append(px[0, 0].permute(1, 2, 0).cpu().numpy())

        # Decode ground truth frames 0..T-1 for comparison
        gt_pixels = []
        for t in range(T):
            px = video_tokenizer.decoder(latents_gt[:, t:t+1])  # (1, 1, C, H, W)
            gt_pixels.append(px[0, 0].permute(1, 2, 0).cpu().numpy())

    # Build comparison grid: row 0 = ground truth, row 1 = predictions
    num_cols = T
    fig, axes = plt.subplots(2, num_cols, figsize=(num_cols * 1.5, 3))
    fig.suptitle('Ground Truth (top) vs Dynamics Predictions (bottom)', fontsize=10)

    action_labels_full = ['seed'] + action_labels  # frame 0 has no action

    for t in range(num_cols):
        axes[0, t].imshow(np.clip(gt_pixels[t], 0, 1))
        axes[0, t].axis('off')
        axes[0, t].set_title(f't={t}\n{action_labels_full[t]}', fontsize=6)

        if t == 0:
            # No prediction for frame 0 — blank
            axes[1, t].axis('off')
        else:
            axes[1, t].imshow(np.clip(predicted_pixels[t - 1], 0, 1))
            axes[1, t].axis('off')

    axes[0, 0].set_ylabel('GT', fontsize=8)
    axes[1, 0].set_ylabel('Pred', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparison grid to {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dynamics', required=True, help='Path to dynamics checkpoint')
    parser.add_argument('--tokenizer', required=True, help='Path to video tokenizer checkpoint')
    parser.add_argument('--lam', required=True, help='Path to LAM checkpoint')
    parser.add_argument('--h5', default='data/vizdoom_healthgathering/vizdoom_healthgathering_dqn.h5')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--out', default='dynamics_test.png')
    args = parser.parse_args()
    test(args.dynamics, args.tokenizer, args.lam, args.h5, args.seed, args.out)
