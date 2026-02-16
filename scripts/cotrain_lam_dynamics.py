import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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
    def __init__(self, h5_path, sequence_length=16):
        """
        Dataset for loading ViZDoom video sequences.
        Respects episode boundaries - sequences will not cross episodes.

        Args:
            h5_path: Path to the h5 file
            sequence_length: Number of frames per sequence
        """
        self.sequence_length = sequence_length

        print(f"Loading dataset from {h5_path}...")
        with h5py.File(h5_path, 'r') as f:
            self.frames = f['frames'][:]
            self.actions = f['actions'][:]
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
        print(f"Frame shape: {self.frames.shape[1:]}")
        print(f"Number of episodes: {len(episode_starts)}")
        print(f"Creating sequences of length {sequence_length}")
        print(f"Total valid sequences: {len(self.valid_indices)}")
        print(f"Memory usage: {self.frames.nbytes / (1024**2):.2f} MB")
        print()

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        frames = self.frames[start_idx:start_idx + self.sequence_length]  # (T, H, W, C)
        frames = frames.astype(np.float32) / 255.0
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W)
        return frames


def train():
    # Shared
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4
    sequence_length = 16
    img_size = (128, 128)
    patch_size = 8
    in_channels = 3

    # VideoTokenizer (frozen) — must match checkpoint
    tokenizer_embed_dim = 128
    tokenizer_latent_dim = 5   # FSQ latent dim for frame tokens
    tokenizer_num_bins = 4     # FSQ bins for frame tokens → 4^5 = 1024 codes

    # LatentActionModel
    lam_embed_dim = 128
    lam_latent_dim_actions = 2  # FSQ latent dim for action tokens
    lam_num_bins = 2            # FSQ bins for action tokens → 2^2 = 4 codes

    # DynamicsModel
    dynamics_embed_dim = 128

    tokenizer_checkpoint = 'checkpoints/video_tokenizer_epoch_6.pt'

    # DDP setup
    ddp = 'LOCAL_RANK' in os.environ
    if ddp:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        is_main = local_rank == 0
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_main = True

    if is_main:
        print(f"Using device: {device}, DDP: {ddp}")

    # Dataset and dataloader
    dataset = VizdoomDataset(
        h5_path='data/vizdoom_healthgathering/vizdoom_healthgathering_dqn.h5',
        sequence_length=sequence_length
    )
    sampler = DistributedSampler(dataset) if ddp else None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=0)

    num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
    num_patches_x = img_size[1] // patch_size
    num_patches_y = img_size[0] // patch_size

    # Frozen pre-trained video tokenizer
    video_tokenizer = VideoTokenizer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_frames=sequence_length,
        embed_dim=tokenizer_embed_dim,
        latent_dim=tokenizer_latent_dim
    ).to(device)
    checkpoint = torch.load(tokenizer_checkpoint, map_location=device)
    video_tokenizer.load_state_dict(checkpoint['model_state_dict'])
    for p in video_tokenizer.parameters():
        p.requires_grad = False
    video_tokenizer.eval()
    if is_main:
        print(f"Loaded frozen VideoTokenizer from {tokenizer_checkpoint}")

    # LAM
    lam = LatentActionModel(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_frames=sequence_length,
        embed_dim=lam_embed_dim,
        latent_dim=tokenizer_latent_dim,
        latent_dim_actions=lam_latent_dim_actions
    ).to(device)

    # Dynamics model
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

    if ddp:
        lam = DDP(lam, device_ids=[local_rank])
        dynamics_model = DDP(dynamics_model, device_ids=[local_rank])

    # FSQ for converting tokenizer latents to target indices (no trainable params)
    fsq = FSQ(latent_dim=tokenizer_latent_dim, num_bins=tokenizer_num_bins).to(device)

    lam_optimizer = torch.optim.Adam(lam.parameters(), lr=learning_rate)
    dynamics_optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=learning_rate)

    if is_main:
        print(f"LAM parameters: {sum(p.numel() for p in lam.parameters()):,}")
        print(f"DynamicsModel parameters: {sum(p.numel() for p in dynamics_model.parameters()):,}")
        print(f"Starting training...")
        print()

    os.makedirs('checkpoints', exist_ok=True)

    T = sequence_length

    for epoch in range(num_epochs):
        lam.train()
        dynamics_model.train()

        total_lam_loss = 0
        total_dynamics_loss = 0

        if ddp:
            sampler.set_epoch(epoch)

        for batch_idx, videos in enumerate(dataloader):
            batch_start_time = time.time()
            B = videos.shape[0]
            videos = videos.to(device)  # (B, T, C, H, W)

            # Step A: get latents from frozen tokenizer
            with torch.no_grad():
                latents = video_tokenizer.encoder(videos)  # (B, T, N, latent_dim)

            # Step B: LAM forward + loss
            reconstructed, actions = lam(videos)  # reconstructed: (B, T-1, C, H, W), actions: (B, T-1, latent_dim)
            lam_loss = F.mse_loss(reconstructed, videos[:, 1:])

            # Step C: prepare dynamics inputs
            null_action = torch.zeros(B, 1, lam_latent_dim_actions, device=device)
            a_shifted = torch.cat([null_action, actions], dim=1)  # (B, T, latent_dim)
            a_shifted = a_shifted.detach()  # stopgrad: don't let dynamics loss affect LAM

            lengths = torch.full((B,), T, dtype=torch.long, device=device)
            targets = fsq.latent_to_index(latents)  # (B, T, N)

            # Step D: dynamics forward + loss
            _, dynamics_loss = dynamics_model(latents, a_shifted, lengths, targets=targets, training=True)

            # Step E: backward passes
            lam_optimizer.zero_grad()
            lam_loss.backward()
            lam_optimizer.step()

            dynamics_optimizer.zero_grad()
            dynamics_loss.backward()
            dynamics_optimizer.step()

            batch_time = time.time() - batch_start_time
            total_lam_loss += lam_loss.item()
            total_dynamics_loss += dynamics_loss.item()

            if is_main:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], "
                      f"LAM Loss: {lam_loss.item():.6f}, Dynamics Loss: {dynamics_loss.item():.6f}, "
                      f"Time: {batch_time:.3f}s")

        avg_lam_loss = total_lam_loss / len(dataloader)
        avg_dynamics_loss = total_dynamics_loss / len(dataloader)

        if is_main:
            print(f"Epoch [{epoch+1}/{num_epochs}] Avg LAM Loss: {avg_lam_loss:.6f}, Avg Dynamics Loss: {avg_dynamics_loss:.6f}")
            print()

            # Unwrap DDP before saving
            lam_state = lam.module.state_dict() if ddp else lam.state_dict()
            dynamics_state = dynamics_model.module.state_dict() if ddp else dynamics_model.state_dict()

            torch.save({
                'epoch': epoch,
                'model_state_dict': lam_state,
                'optimizer_state_dict': lam_optimizer.state_dict(),
                'lam_loss': avg_lam_loss,
            }, f'checkpoints/lam_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': dynamics_state,
                'optimizer_state_dict': dynamics_optimizer.state_dict(),
                'dynamics_loss': avg_dynamics_loss,
            }, f'checkpoints/dynamics_epoch_{epoch+1}.pt')
            print(f"Checkpoints saved: lam_epoch_{epoch+1}.pt, dynamics_epoch_{epoch+1}.pt")
            print()


if __name__ == '__main__':
    train()
    if dist.is_initialized():
        dist.destroy_process_group()
