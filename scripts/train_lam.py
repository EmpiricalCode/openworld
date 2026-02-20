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

from core.model.latent_action_model import LatentActionModel


class VizdoomDataset(Dataset):
    def __init__(self, h5_path, sequence_length=16):
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

        self.num_labeled = len(self.valid_indices)

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
        frames = self.frames[start_idx:start_idx + self.sequence_length]
        frames = frames.astype(np.float32) / 255.0
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W)
        game_actions = torch.from_numpy(self.actions[start_idx:start_idx + self.sequence_length].astype(np.int64))  # (T,)
        return frames, game_actions


def train(resume=None):
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4
    sequence_length = 16
    img_size = (64, 64)
    patch_size = 8
    in_channels = 3

    lam_embed_dim = 128
    lam_latent_dim_actions = 3
    tokenizer_latent_dim = 5
    tokenizer_num_bins = 4

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

    dataset = VizdoomDataset(
        h5_path='/datasets/health-gathering/vizdoom_healthgathering_dqn.h5',
        sequence_length=sequence_length
    )
    sampler = DistributedSampler(dataset) if ddp else None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=0)

    lam = LatentActionModel(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_frames=sequence_length,
        embed_dim=lam_embed_dim,
        latent_dim=tokenizer_latent_dim,
        latent_dim_actions=lam_latent_dim_actions,
        num_bins=tokenizer_num_bins
    ).to(device)

    if ddp:
        lam = DDP(lam, device_ids=[local_rank])

    # Small classifier: maps action latents -> game action logits
    action_classifier = torch.nn.Linear(lam_latent_dim_actions, 4).to(device)

    optimizer = torch.optim.Adam(list(lam.parameters()) + list(action_classifier.parameters()), lr=learning_rate)

    start_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        (lam.module if ddp else lam).load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        if is_main:
            print(f"Resumed from {resume} (epoch {ckpt['epoch']+1})")

    if is_main:
        print(f"LAM parameters: {sum(p.numel() for p in lam.parameters()):,}")
        print(f"Starting training...")
        print()

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        lam.train()
        total_loss = 0
        total_recon_loss = 0
        total_supervised_loss = 0

        if ddp:
            sampler.set_epoch(epoch)

        for batch_idx, (videos, game_actions) in enumerate(dataloader):
            batch_start_time = time.time()
            B = videos.shape[0]
            videos = videos.to(device)
            game_actions = game_actions.to(device)

            # LAM forward
            reconstructed, actions = lam(videos)  # (B, T-1, C, H, W), (B, T-1, latent_dim_actions)
            recon_loss = F.mse_loss(reconstructed, videos[:, 1:])

            # Supervised loss on all samples
            logits = action_classifier(actions)          # (B, T-1, 4)
            labels = game_actions[:, :-1]                # (B, T-1)
            supervised_loss = F.cross_entropy(logits.reshape(-1, 4), labels.reshape(-1))

            loss = recon_loss + supervised_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time = time.time() - batch_start_time
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_supervised_loss += supervised_loss.item()

            if is_main:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], "
                      f"Loss: {loss.item():.6f}, Recon: {recon_loss.item():.6f}, "
                      f"Supervised: {supervised_loss.item():.6f}, Time: {batch_time:.3f}s")

        if is_main:
            avg_loss = total_loss / len(dataloader)
            avg_recon = total_recon_loss / len(dataloader)
            avg_sup = total_supervised_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}] Avg Loss: {avg_loss:.6f}, Recon: {avg_recon:.6f}, Supervised: {avg_sup:.6f}")
            print()

            lam_state = lam.module.state_dict() if ddp else lam.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': lam_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'recon_loss': avg_recon,
                'supervised_loss': avg_sup,
            }, f'checkpoints/lam_epoch_{epoch+1}.pt')
            print(f"Checkpoint saved: lam_epoch_{epoch+1}.pt")
            print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    train(resume=args.resume)
    if dist.is_initialized():
        dist.destroy_process_group()
