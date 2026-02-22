"""
Test LAM action consistency: for each true game action (0-3), collect the
distribution of LAM-predicted action tokens. Good consistency = each game
action maps to one dominant token.
"""
import torch
import numpy as np
import sys
import os
import h5py
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model.latent_action_model import LatentActionModel
from core.model.components.quantization import FSQ


def test_action_consistency(lam_checkpoint, h5_path='data/vizdoom_healthgathering/vizdoom_healthgathering_dqn.h5',
                            num_sequences=500, sequence_length=16):
    img_size = (64, 64)
    patch_size = 8
    in_channels = 3
    embed_dim = 128
    latent_dim = 5
    latent_dim_actions = 3
    num_bins = 2  # 2^3 = 8 action codes

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load LAM
    lam = LatentActionModel(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_frames=sequence_length,
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        latent_dim_actions=latent_dim_actions,
        num_bins=4
    ).to(device)
    ckpt = torch.load(lam_checkpoint, map_location=device)
    lam.load_state_dict(ckpt['model_state_dict'])
    lam.eval()
    print(f"Loaded LAM from {lam_checkpoint}")
    sup_loss_str = f", Supervised loss: {ckpt['supervised_loss']:.6f}" if 'supervised_loss' in ckpt else ""
    print(f"  Epoch: {ckpt['epoch'] + 1}, LAM loss: {ckpt['loss']:.6f}{sup_loss_str}")

    # FSQ to convert continuous action latents to discrete tokens
    fsq_actions = FSQ(latent_dim=latent_dim_actions, num_bins=num_bins).to(device)

    # Load dataset
    print(f"\nLoading dataset from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        frames = f['frames'][:]    # (N, H, W, C) uint8
        actions = f['actions'][:]  # (N,) int
        dones = f['dones'][:]

    total_frames = frames.shape[0]
    print(f"Total frames: {total_frames}, unique game actions: {np.unique(actions)}")

    # Find valid sequence start indices (no episode boundary crossings)
    episode_ends = np.where(dones)[0]
    episode_starts = np.concatenate([[0], episode_ends[:-1] + 1]) if len(episode_ends) > 0 else [0]
    episode_ends = np.concatenate([episode_ends, [total_frames - 1]])

    valid_indices = []
    for start, end in zip(episode_starts, episode_ends):
        episode_length = end - start + 1
        if episode_length >= sequence_length:
            for i in range(start, end - sequence_length + 2):
                valid_indices.append(i)

    print(f"Valid sequences: {len(valid_indices)}")

    # Sample sequences
    rng = np.random.default_rng(42)
    sampled = rng.choice(len(valid_indices), size=min(num_sequences, len(valid_indices)), replace=False)

    # For each game action, count how many times each LAM token was predicted
    # game_action -> {lam_token: count}
    action_token_counts = defaultdict(lambda: defaultdict(int))
    total_transitions = 0

    with torch.no_grad():
        for idx in sampled:
            start_idx = valid_indices[idx]
            seq_frames = frames[start_idx:start_idx + sequence_length]  # (T, H, W, C)
            seq_actions = actions[start_idx:start_idx + sequence_length]  # (T,) game actions

            # Normalize and convert to tensor
            seq_frames_f = seq_frames.astype(np.float32) / 255.0
            videos = torch.from_numpy(seq_frames_f).permute(0, 3, 1, 2).unsqueeze(0).to(device)  # (1, T, C, H, W)

            # LAM forward: actions shape (1, T-1, latent_dim_actions)
            _, lam_actions_continuous = lam(videos)

            # Quantize to get discrete tokens: (1, T-1, latent_dim_actions) -> (1, T-1) token indices
            lam_tokens = fsq_actions.latent_to_index(lam_actions_continuous)  # (1, T-1)
            lam_tokens = lam_tokens[0].cpu().numpy()  # (T-1,)

            # seq_actions[t] is the game action taken at step t, which caused the transition t->t+1
            # So lam_tokens[t] should correspond to seq_actions[t] for t in 0..T-2
            for t in range(sequence_length - 1):
                game_action = seq_actions[t]
                lam_token = lam_tokens[t]
                action_token_counts[game_action][lam_token] += 1
                total_transitions += 1

    # Print results
    print(f"\nTotal transitions analyzed: {total_transitions}")
    print(f"\nLAM token distribution per game action:")
    num_tokens = num_bins ** latent_dim_actions  # 8
    header = f"{'Game Action':<15}" + "".join(f"{'Token '+str(i):>10}" for i in range(num_tokens)) + f"{'Dominant':>10} {'Purity':>10}"
    print(header)
    print("-" * (15 + 10 * num_tokens + 20))

    for game_action in sorted(action_token_counts.keys()):
        token_dist = action_token_counts[game_action]
        total = sum(token_dist.values())
        counts = [token_dist.get(t, 0) for t in range(num_tokens)]
        dominant_token = np.argmax(counts)
        purity = counts[dominant_token] / total if total > 0 else 0.0
        pcts = "".join(f"{c/total*100:>9.1f}%" for c in counts)
        print(f"{game_action:<15} {pcts} {dominant_token:>10} {purity*100:>9.1f}%")

    # Summary
    print(f"\nSummary:")
    action_names = {0: 'noop', 1: 'move_forward', 2: 'turn_right', 3: 'turn_left'}
    dominant_map = {}
    for game_action in sorted(action_token_counts.keys()):
        token_dist = action_token_counts[game_action]
        total = sum(token_dist.values())
        counts = [token_dist.get(t, 0) for t in range(num_tokens)]
        dominant_token = int(np.argmax(counts))
        purity = counts[dominant_token] / total
        name = action_names.get(game_action, str(game_action))
        dominant_map[game_action] = dominant_token
        print(f"  Game action {game_action} ({name}) -> LAM token {dominant_token} ({purity*100:.1f}% of the time)")

    # Check for collisions (two game actions mapping to same token)
    tokens_used = list(dominant_map.values())
    if len(set(tokens_used)) < len(tokens_used):
        print(f"\n  WARNING: Multiple game actions map to the same dominant token — action collapse detected")
    else:
        print(f"\n  All game actions map to distinct dominant tokens — good separation!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lam', required=True, help='Path to LAM checkpoint')
    parser.add_argument('--data', default='data/vizdoom_healthgathering/vizdoom_healthgathering_dqn.h5')
    parser.add_argument('--num-sequences', type=int, default=500)
    args = parser.parse_args()
    test_action_consistency(args.lam, args.h5, args.num_sequences)
