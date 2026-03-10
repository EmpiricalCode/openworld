"""
Precompute tokenizer latents and LAM actions for dynamics training.
Saves to an h5 file with datasets: latents, actions.
Run this once before training dynamics model.
"""
import torch
import numpy as np
import sys
import os
import h5py
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.model.video_tokenizer import VideoTokenizer
from core.model.latent_action_model import LatentActionModel
from core.model.components.quantization import FSQ


def precompute(h5_path, tokenizer_ckpt, lam_ckpt, out_path):
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'core', 'config', 'dynamics.json')
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load raw frames and find valid sequences
    print(f"Loading dataset from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        frames = f['frames'][:]
        dones = f['dones'][:]

    total_frames = frames.shape[0]

    episode_ends = np.where(dones)[0]
    episode_starts = np.concatenate([[0], episode_ends[:-1] + 1]) if len(episode_ends) > 0 else [0]
    episode_ends = np.concatenate([episode_ends, [total_frames - 1]])

    valid_indices = []
    for start, end in zip(episode_starts, episode_ends):
        episode_length = end - start + 1
        if episode_length >= sequence_length:
            for i in range(start, end - sequence_length + 2):
                valid_indices.append(i)

    num_sequences = len(valid_indices)
    print(f"Total frames: {total_frames}, valid sequences: {num_sequences}")

    # Load models
    print("Loading VideoTokenizer...")
    video_tokenizer = VideoTokenizer(
        img_size=img_size, patch_size=patch_size, in_channels=in_channels,
        num_frames=sequence_length, embed_dim=tokenizer_embed_dim, latent_dim=tokenizer_latent_dim
    ).to(device)
    ckpt = torch.load(tokenizer_ckpt, map_location=device)
    video_tokenizer.load_state_dict(ckpt['model_state_dict'])
    video_tokenizer.eval()
    del ckpt

    fsq = FSQ(latent_dim=tokenizer_latent_dim, num_bins=tokenizer_num_bins).to(device)
    num_patches = video_tokenizer.encoder.patch_embedding.num_patches

    print("Loading LAM...")
    lam = LatentActionModel(
        img_size=img_size, patch_size=8, in_channels=in_channels,
        num_frames=sequence_length, embed_dim=lam_embed_dim, latent_dim=tokenizer_latent_dim,
        latent_dim_actions=lam_latent_dim_actions, num_bins=tokenizer_num_bins
    ).to(device)
    ckpt = torch.load(lam_ckpt, map_location=device)
    lam.load_state_dict(ckpt['model_state_dict'])
    lam.eval()
    del ckpt

    # Create output h5 file
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    h5_out = h5py.File(out_path, 'w')
    h5_out.create_dataset('latents', shape=(num_sequences, sequence_length, num_patches, tokenizer_latent_dim), dtype='float32')
    h5_out.create_dataset('actions', shape=(num_sequences, sequence_length, lam_latent_dim_actions), dtype='float32')
    h5_out.create_dataset('targets', shape=(num_sequences, sequence_length, num_patches), dtype='int64')

    print("Computing latents and actions...")
    batch_size = 32
    use_amp = device.type == 'cuda'
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        for i in range(0, num_sequences, batch_size):
            batch_end = min(i + batch_size, num_sequences)
            batch_indices = valid_indices[i:batch_end]
            B = len(batch_indices)

            batch_frames = np.stack([frames[idx:idx + sequence_length] for idx in batch_indices])
            batch_frames = batch_frames.astype(np.float32) / 255.0
            videos = torch.from_numpy(batch_frames).permute(0, 1, 4, 2, 3).to(device)

            latents = video_tokenizer.encoder(videos)
            targets = fsq.latent_to_index(latents)
            _, actions = lam(videos)
            null_action = torch.zeros(B, 1, lam_latent_dim_actions, device=device)
            a_shifted = torch.cat([null_action, actions], dim=1)

            h5_out['latents'][i:batch_end] = latents.float().cpu().numpy()
            h5_out['targets'][i:batch_end] = targets.cpu().numpy()
            h5_out['actions'][i:batch_end] = a_shifted.float().cpu().numpy()

            if (i // batch_size) % 50 == 0:
                print(f"  {batch_end}/{num_sequences}")

    h5_out.close()
    del video_tokenizer, lam
    torch.cuda.empty_cache()

    file_size_mb = os.path.getsize(out_path) / (1024**2)
    print(f"\nDone! Saved to {out_path} ({file_size_mb:.1f} MB)")
    print(f"  latents: ({num_sequences}, {sequence_length}, {num_patches}, {tokenizer_latent_dim})")
    print(f"  actions: ({num_sequences}, {sequence_length}, {lam_latent_dim_actions})")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to raw frames h5 file')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to video tokenizer checkpoint')
    parser.add_argument('--lam', type=str, required=True, help='Path to LAM checkpoint')
    parser.add_argument('--out', type=str, required=True, help='Output path for precomputed h5')
    args = parser.parse_args()
    precompute(args.data, args.tokenizer, args.lam, args.out)
