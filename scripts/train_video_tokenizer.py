import torch
import torch.nn as nn
import h5py
import numpy as np
import sys
import os
import time
from torch.utils.data import Dataset, DataLoader

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
        print(f"Frame shape: {self.frames.shape[1:]}")  # (128, 128, 3)
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
                         T = sequence_length
                         C = 3 (RGB)
                         H = 128
                         W = 128
        """
        # Get the actual frame index from valid indices
        start_idx = self.valid_indices[idx]

        # Get sequence of frames from in-memory array
        frames = self.frames[start_idx:start_idx + self.sequence_length]  # (T, H, W, C)

        # Convert to float and normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0

        # Convert to tensor and permute to (T, C, H, W)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)

        return frames

def train(h5_path='data/vizdoom_healthgathering/vizdoom_healthgathering_dqn.h5'):
    # Hyperparameters
    batch_size = 32  # Increased from 4 for better GPU utilization
    num_epochs = 10
    learning_rate = 1e-4
    sequence_length = 16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataset and dataloader
    dataset = VizdoomDataset(
        h5_path=h5_path,
        sequence_length=sequence_length
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # Initialize model
    model = VideoTokenizer(
        img_size=(64, 64),
        patch_size=4,
        in_channels=3,
        num_frames=sequence_length,
        embed_dim=128
    ).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Starting training...")
    print()

    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

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

            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.6f}, Time: {batch_time:.3f}s")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.6f}")
        print()

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'checkpoints/video_tokenizer_epoch_{epoch+1}.pt')
        print(f"Checkpoint saved: checkpoints/video_tokenizer_epoch_{epoch+1}.pt")
        print()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/vizdoom_healthgathering/vizdoom_healthgathering_dqn.h5')
    args = parser.parse_args()
    train(h5_path=args.data)
