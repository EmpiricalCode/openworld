"""
Diagnostic: train dynamics with sequence_length=2 to test if the model
is just copying frame 0 or actually using actions.

With only 2 frames, the model sees frame 0 + action and must predict frame 1.
If it ignores actions and copies frame 0, loss will plateau at the average
inter-frame distance. If it uses actions, loss should go lower.
"""
import torch
import torch.nn.functional as F
import h5py
import numpy as np
import sys
import os
import time
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model.video_tokenizer import VideoTokenizer
from core.model.latent_action_model import LatentActionModel
from core.model.dynamics_model import DynamicsModel
from core.model.components.quantization import FSQ


class VizdoomDataset(Dataset):
    def __init__(self, h5_path, sequence_length=2):
        self.sequence_length = sequence_length

        print(f"Loading dataset from {h5_path}...")
        with h5py.File(h5_path, 'r') as f:
            self.frames = f['frames'][:]
            self.dones = f['dones'][:]

        self.total_frames = self.frames.shape[0]

        episode_ends = np.where(self.dones)[0]
        episode_starts = np.concatenate([[0], episode_ends[:-1] + 1]) if len(episode_ends) > 0 else [0]
        episode_ends = np.concatenate([episode_ends, [self.total_frames - 1]])

        self.valid_indices = []
        for start, end in zip(episode_starts, episode_ends):
            episode_length = end - start + 1
            if episode_length >= sequence_length:
                for i in range(start, end - sequence_length + 2):
                    self.valid_indices.append(i)

        print(f"Dataset loaded: {self.total_frames} total frames")
        print(f"Total valid sequences: {len(self.valid_indices)}")
        print()

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        frames = self.frames[start_idx:start_idx + self.sequence_length]
        frames = frames.astype(np.float32) / 255.0
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        return frames


def train():
    batch_size = 128
    num_epochs = 5
    learning_rate = 1e-4
    sequence_length = 2  # KEY: only 2 frames
    img_size = (64, 64)
    patch_size = 4
    in_channels = 3

    tokenizer_embed_dim = 128
    tokenizer_latent_dim = 5
    tokenizer_num_bins = 4
    lam_embed_dim = 128
    lam_latent_dim_actions = 3
    dynamics_embed_dim = 192

    # We need the full-length models for encoding, but dynamics uses num_frames=2
    full_sequence_length = 16

    tokenizer_checkpoint = 'checkpoints/video_tokenizer_epoch_6.pt'
    lam_checkpoint = 'checkpoints/lam_epoch_3.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"DIAGNOSTIC MODE: sequence_length={sequence_length}")
    print()

    dataset = VizdoomDataset(
        h5_path='data/vizdoom_healthgathering/vizdoom_healthgathering_dqn.h5',
        sequence_length=full_sequence_length  # Load full sequences for LAM
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    num_patches_x = img_size[1] // patch_size
    num_patches_y = img_size[0] // patch_size

    # Frozen video tokenizer (full length for encoding)
    video_tokenizer = VideoTokenizer(
        img_size=img_size, patch_size=patch_size, in_channels=in_channels,
        num_frames=full_sequence_length, embed_dim=tokenizer_embed_dim,
        latent_dim=tokenizer_latent_dim
    ).to(device)
    checkpoint = torch.load(tokenizer_checkpoint, map_location=device)
    video_tokenizer.load_state_dict(checkpoint['model_state_dict'])
    for p in video_tokenizer.parameters():
        p.requires_grad = False
    video_tokenizer.eval()
    print(f"Loaded frozen VideoTokenizer from {tokenizer_checkpoint}")

    # Frozen LAM (full length for action extraction)
    lam = LatentActionModel(
        img_size=img_size, patch_size=8, in_channels=in_channels,
        num_frames=full_sequence_length, embed_dim=lam_embed_dim,
        latent_dim=tokenizer_latent_dim, latent_dim_actions=lam_latent_dim_actions,
        num_bins=tokenizer_num_bins
    ).to(device)
    checkpoint = torch.load(lam_checkpoint, map_location=device)
    lam.load_state_dict(checkpoint['model_state_dict'])
    for p in lam.parameters():
        p.requires_grad = False
    lam.eval()
    print(f"Loaded frozen LAM from {lam_checkpoint}")

    # Dynamics model with num_frames=2
    dynamics_model = DynamicsModel(
        num_patches_x=num_patches_x, num_patches_y=num_patches_y,
        in_channels=in_channels, num_frames=sequence_length,
        embed_dim=dynamics_embed_dim, latent_dim=tokenizer_latent_dim,
        latent_dim_actions=lam_latent_dim_actions, num_bins=tokenizer_num_bins
    ).to(device)

    fsq = FSQ(latent_dim=tokenizer_latent_dim, num_bins=tokenizer_num_bins).to(device)
    dynamics_optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=learning_rate)

    print(f"DynamicsModel parameters: {sum(p.numel() for p in dynamics_model.parameters()):,}")
    print(f"Starting diagnostic training...")
    print()

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(num_epochs):
        dynamics_model.train()
        total_loss = 0

        for batch_idx, videos in enumerate(dataloader):
            batch_start_time = time.time()
            B = videos.shape[0]
            videos = videos.to(device)  # (B, 16, C, H, W)

            with torch.no_grad():
                # Encode all 16 frames
                latents_full = video_tokenizer.encoder(videos)  # (B, 16, N, latent_dim)

                # Get actions from LAM (needs full 16 frames)
                _, actions_full = lam(videos)  # (B, 15, latent_dim_actions)

                # Slice to first 2 frames
                latents = latents_full[:, :sequence_length]  # (B, 2, N, latent_dim)
                targets = fsq.latent_to_index(latents)  # (B, 2, N)

                # Action for frame 1 = actions_full[:, 0]
                null_action = torch.zeros(B, 1, lam_latent_dim_actions, device=device)
                action_0 = actions_full[:, 0:1]  # (B, 1, latent_dim_actions)
                a_shifted = torch.cat([null_action, action_0], dim=1)  # (B, 2, latent_dim_actions)

            lengths = torch.full((B,), sequence_length, dtype=torch.long, device=device)

            _, dynamics_loss = dynamics_model(latents, a_shifted, lengths, targets=targets, training=True)

            dynamics_optimizer.zero_grad()
            dynamics_loss.backward()
            torch.nn.utils.clip_grad_norm_(dynamics_model.parameters(), max_norm=1.0)
            dynamics_optimizer.step()

            batch_time = time.time() - batch_start_time
            total_loss += dynamics_loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], "
                      f"Loss: {dynamics_loss.item():.6f}, Time: {batch_time:.3f}s")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Avg Loss: {avg_loss:.6f}")
        print()


if __name__ == '__main__':
    train()
