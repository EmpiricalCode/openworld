"""
Evaluate dynamics model loss on a random 16-frame sequence from the dataset.
Uses the same loss computation as train_dynamics.py.
"""
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import h5py
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model.video_tokenizer import VideoTokenizer
from core.model.latent_action_model import LatentActionModel
from core.model.dynamics_model import DynamicsModel
from core.model.components.quantization import FSQ


def evaluate(dynamics_checkpoint, tokenizer_checkpoint, lam_checkpoint,
             h5_path='data/vizdoom_healthgathering/vizdoom_healthgathering_dqn.h5',
             seed=None, num_samples=10):

    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core', 'config', 'dynamics.json')
    with open(config_path) as f:
        cfg = json.load(f)

    sequence_length = cfg['sequence_length']
    img_size = tuple(cfg['img_size'])
    patch_size = cfg['patch_size']
    in_channels = cfg['in_channels']
    tokenizer_embed_dim = cfg['tokenizer_embed_dim']
    tokenizer_latent_dim = cfg['tokenizer_latent_dim']
    tokenizer_num_bins = cfg['tokenizer_num_bins']
    lam_embed_dim = cfg['lam_embed_dim']
    lam_latent_dim_actions = cfg['lam_latent_dim_actions']
    dynamics_embed_dim = cfg['dynamics_embed_dim']
    dynamics_num_blocks = cfg['num_blocks']
    dynamics_num_heads = cfg['num_heads']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_patches_x = img_size[1] // patch_size
    num_patches_y = img_size[0] // patch_size

    # Load models
    video_tokenizer = VideoTokenizer(
        img_size=img_size, patch_size=patch_size, in_channels=in_channels,
        num_frames=sequence_length, embed_dim=tokenizer_embed_dim, latent_dim=tokenizer_latent_dim
    ).to(device)
    ckpt = torch.load(tokenizer_checkpoint, map_location=device)
    video_tokenizer.load_state_dict(ckpt['model_state_dict'])
    video_tokenizer.eval()
    print(f"Loaded VideoTokenizer from {tokenizer_checkpoint}")

    lam = LatentActionModel(
        img_size=img_size, patch_size=8, in_channels=in_channels,
        num_frames=sequence_length, embed_dim=lam_embed_dim, latent_dim=tokenizer_latent_dim,
        latent_dim_actions=lam_latent_dim_actions, num_bins=tokenizer_num_bins
    ).to(device)
    ckpt = torch.load(lam_checkpoint, map_location=device)
    lam.load_state_dict(ckpt['model_state_dict'])
    lam.eval()
    print(f"Loaded LAM from {lam_checkpoint}")

    dynamics_model = DynamicsModel(
        num_patches_x=num_patches_x, num_patches_y=num_patches_y,
        in_channels=in_channels, num_frames=sequence_length,
        embed_dim=dynamics_embed_dim, latent_dim=tokenizer_latent_dim,
        latent_dim_actions=lam_latent_dim_actions, num_bins=tokenizer_num_bins,
        num_blocks=dynamics_num_blocks, num_heads=dynamics_num_heads
    ).to(device)
    ckpt = torch.load(dynamics_checkpoint, map_location=device)
    dynamics_model.load_state_dict(ckpt['model_state_dict'])
    dynamics_model.eval()
    print(f"Loaded DynamicsModel from {dynamics_checkpoint}")
    print(f"  Epoch: {ckpt['epoch'] + 1}, Train loss: {ckpt['dynamics_loss']:.6f}")

    fsq = FSQ(latent_dim=tokenizer_latent_dim, num_bins=tokenizer_num_bins).to(device)

    # Load dataset
    print(f"\nLoading dataset from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        frames_all = f['frames'][:]
        dones_all = f['dones'][:]

    total_frames = frames_all.shape[0]
    episode_ends = np.where(dones_all)[0]
    episode_starts = np.concatenate([[0], episode_ends[:-1] + 1]) if len(episode_ends) > 0 else [0]
    episode_ends = np.concatenate([episode_ends, [total_frames - 1]])

    valid_indices = []
    for start, end in zip(episode_starts, episode_ends):
        if end - start + 1 >= sequence_length:
            for i in range(start, end - sequence_length + 2):
                valid_indices.append(i)

    print(f"Total valid sequences: {len(valid_indices)}")

    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(valid_indices), size=num_samples, replace=False)

    T = sequence_length
    losses = []

    with torch.no_grad():
        for i, sample_idx in enumerate(sample_indices):
            start_idx = valid_indices[sample_idx]
            frames = frames_all[start_idx:start_idx + sequence_length]
            frames_f = frames.astype(np.float32) / 255.0
            videos = torch.from_numpy(frames_f).permute(0, 3, 1, 2).unsqueeze(0).to(device)  # (1, T, C, H, W)

            latents = video_tokenizer.encoder(videos)  # (1, T, N, latent_dim)
            _, actions = lam(videos)  # (1, T-1, latent_dim_actions)
            null_action = torch.zeros(1, 1, lam_latent_dim_actions, device=device)
            a_shifted = torch.cat([null_action, actions], dim=1)  # (1, T, latent_dim_actions)

            targets = fsq.latent_to_index(latents)  # (1, T, N)
            lengths = torch.full((1,), T, dtype=torch.long, device=device)

            x_predict, loss = dynamics_model(latents, a_shifted, lengths, targets=targets, training=True)

            losses.append(loss.item())
            print(f"  Sample {i+1}/{num_samples} (frame {start_idx}): loss = {loss.item():.6f}")

    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    print(f"\nResults over {num_samples} samples:")
    print(f"  Mean loss: {mean_loss:.6f}")
    print(f"  Std loss:  {std_loss:.6f}")
    print(f"  Min loss:  {np.min(losses):.6f}")
    print(f"  Max loss:  {np.max(losses):.6f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dynamics', required=True, help='Path to dynamics checkpoint')
    parser.add_argument('--tokenizer', required=True, help='Path to video tokenizer checkpoint')
    parser.add_argument('--lam', required=True, help='Path to LAM checkpoint')
    parser.add_argument('--h5', default='data/vizdoom_healthgathering/vizdoom_healthgathering_dqn.h5')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num-samples', type=int, default=10, help='Number of random sequences to evaluate')
    args = parser.parse_args()
    evaluate(args.dynamics, args.tokenizer, args.lam, args.h5, args.seed, args.num_samples)
