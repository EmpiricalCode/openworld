import torch
import torch.nn as nn
import h5py
import numpy as np
import sys
import os
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model.video_tokenizer import VideoTokenizer

class LunarLanderDataset(Dataset):
    def __init__(self, h5_path, sequence_length=8):
        """
        Dataset for loading Lunar Lander video sequences.

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

        self.total_frames = self.frames.shape[0]
        print(f"Dataset loaded: {self.total_frames} total frames")
        print(f"Frame shape: {self.frames.shape[1:]}")  # (128, 128, 3)
        print(f"Creating sequences of length {sequence_length}")
        print(f"Total sequences: {self.__len__()}")
        print(f"Memory usage: {self.frames.nbytes / (1024**2):.2f} MB")
        print()

    def __len__(self):
        # Number of sequences we can create
        # We need sequence_length consecutive frames for each sequence
        return self.total_frames - self.sequence_length + 1

    def __getitem__(self, idx):
        """
        Returns a sequence of frames.

        Returns:
            torch.Tensor: Video sequence of shape (T, C, H, W)
                         T = sequence_length
                         C = 3 (RGB)
                         H = 128
                         W = 128
        """
        # Get sequence of frames from in-memory array
        frames = self.frames[idx:idx + self.sequence_length]  # (T, H, W, C)

        # Convert to float and normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0

        # Convert to tensor and permute to (T, C, H, W)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)

        return frames

def train():
    # Hyperparameters
    batch_size = 16  # Increased from 4 for better GPU utilization
    num_epochs = 10
    learning_rate = 1e-4
    sequence_length = 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataset and dataloader
    dataset = LunarLanderDataset(
        h5_path='data/lunar_lander/lunar_lander_10k_steps.h5',
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
        img_size=(128, 128),
        patch_size=8,
        in_channels=3,
        num_frames=sequence_length,
        embed_dim=128,
        latent_dim=5
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
            videos = videos.to(device)

            # Forward pass
            reconstructed = model(videos)
            loss = criterion(reconstructed, videos)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.6f}")
        print()

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoints/video_tokenizer_epoch_{epoch+1}.pt')
            print(f"Checkpoint saved: checkpoints/video_tokenizer_epoch_{epoch+1}.pt")
            print()

if __name__ == '__main__':
    train()
