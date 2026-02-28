import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import h5py
import numpy as np
import sys
import os
import time
from torch.utils.data import Dataset, DataLoader
import modal

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model.video_tokenizer import VideoTokenizer

class VizdoomDataset(Dataset):
    def __init__(self, h5_path, sequence_length=8):
        """
        Dataset for loading ViZDoom video sequences.
        Respects episode boundaries - sequences will not cross episodes.

        Args:
            h5_path: Path to the h5 file
            sequence_length: Number of frames per sequence
        """
        self.sequence_length = sequence_length

        # Load entire dataset into memory once
        print(f"Loading dataset from {h5_path}...")
        with h5py.File(h5_path, 'r') as f:
            # Load all frames into RAM
            self.frames = f['frames'][:]  # Load entire array
            self.dones = f['dones'][:]    # Load episode boundaries

        self.total_frames = self.frames.shape[0]

        # Find episode boundaries
        episode_ends = np.where(self.dones)[0]
        episode_starts = np.concatenate([[0], episode_ends[:-1] + 1]) if len(episode_ends) > 0 else [0]
        episode_ends = np.concatenate([episode_ends, [self.total_frames - 1]])

        # Create list of valid sequence start indices
        self.valid_indices = []
        for start, end in zip(episode_starts, episode_ends):
            episode_length = end - start + 1
            if episode_length >= sequence_length:
                # Add all valid starting positions within this episode
                for i in range(start, end - sequence_length + 2):
                    self.valid_indices.append(i)

        print(f"Dataset loaded: {self.total_frames} total frames")
        print(f"Frame shape: {self.frames.shape[1:]}")
        print(f"Number of episodes: {len(episode_starts)}")
        print(f"Creating sequences of length {sequence_length}")
        print(f"Total valid sequences (respecting episode boundaries): {len(self.valid_indices)}")
        print(f"Memory usage: {self.frames.nbytes / (1024**2):.2f} MB")
        print()

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Returns a sequence of frames that does not cross episode boundaries.

        Returns:
            torch.Tensor: Video sequence of shape (T, C, H, W)
        """
        start_idx = self.valid_indices[idx]
        frames = self.frames[start_idx:start_idx + self.sequence_length]

        # Convert to float and normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0

        # Convert to tensor and permute to (T, C, H, W)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)

        return frames

def train(h5_path='data/vizdoom_healthgathering/vizdoom_healthgathering_dqn.h5', resume=None, volume_name=None):
    if volume_name:
        vol = modal.Volume.from_name(volume_name, create_if_missing=True)

    # Hyperparameters
    batch_size = 64
    num_epochs = 10
    learning_rate = 1.4e-4
    sequence_length = 16

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

    # Create dataset and dataloader
    dataset = VizdoomDataset(
        h5_path=h5_path,
        sequence_length=sequence_length
    )
    sampler = DistributedSampler(dataset) if ddp else None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=0)

    # Initialize model
    model = VideoTokenizer(
        img_size=(64, 64),
        patch_size=4,
        in_channels=3,
        num_frames=sequence_length,
        embed_dim=128
    ).to(device)

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        (model.module if ddp else model).load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        if is_main:
            print(f"Resumed from {resume} (epoch {ckpt['epoch']+1})")

    if is_main:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Starting training...")
        print()

    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0

        if ddp:
            sampler.set_epoch(epoch)

        for batch_idx, videos in enumerate(dataloader):
            batch_start_time = time.time()

            videos = videos.to(device)

            # Forward pass
            reconstructed = model(videos)
            loss = criterion(reconstructed, videos)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time = time.time() - batch_start_time
            total_loss += loss.item()

            if is_main:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.6f}, Time: {batch_time:.3f}s")

        if is_main:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.6f}")
            print()

            # Save checkpoint
            model_state = model.module.state_dict() if ddp else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoints/video_tokenizer_epoch_{epoch+1}.pt')
            print(f"Checkpoint saved: checkpoints/video_tokenizer_epoch_{epoch+1}.pt")
            if volume_name:
                with vol.batch_upload(force=True) as batch:
                    batch.put_file(f'checkpoints/video_tokenizer_epoch_{epoch+1}.pt', f'video_tokenizer_epoch_{epoch+1}.pt')
                print(f"Uploaded to volume: {volume_name}")
            print()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/vizdoom_healthgathering/vizdoom_healthgathering_dqn.h5')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--volume', type=str, default=None, help='Modal volume name to upload checkpoints to')
    args = parser.parse_args()
    train(h5_path=args.data, resume=args.resume, volume_name=args.volume)
    if dist.is_initialized():
        dist.destroy_process_group()
