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
    def __init__(self, h5_path, sequence_length=16, label_fraction=0.1):
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

        # Per-frame label mask: True means this frame has a labelled action
        # Stratified by action class so each action is equally represented
        self.label_mask = np.zeros(self.total_frames, dtype=bool)
        unique_actions = np.unique(self.actions)
        num_per_action = max(1, int(label_fraction * self.total_frames / len(unique_actions)))
        for a in unique_actions:
            indices = np.where(self.actions == a)[0]
            n_select = min(num_per_action, len(indices))
            selected = np.random.choice(indices, size=n_select, replace=False)
            self.label_mask[selected] = True
        num_labeled = self.label_mask.sum()

        print(f"Dataset loaded: {self.total_frames} total frames")
        print(f"Frame shape: {self.frames.shape[1:]}")
        print(f"Number of episodes: {len(episode_starts)}")
        print(f"Creating sequences of length {sequence_length}")
        print(f"Total valid sequences: {len(self.valid_indices)}")
        print(f"Label fraction: {label_fraction:.1%} ({num_labeled}/{self.total_frames} frames labelled)")
        per_action_counts = {int(a): int(self.label_mask[self.actions == a].sum()) for a in unique_actions}
        print(f"Labels per action: {per_action_counts}")
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
        mask = torch.from_numpy(self.label_mask[start_idx:start_idx + self.sequence_length])  # (T,)
        return frames, game_actions, mask


def train(resume=None, h5_path='/datasets/health-gathering/vizdoom_healthgathering_dqn.h5', num_actions=4, num_epochs=10, sup_weight=1.0, label_fraction=1.0, seed=None):
    # Set seed for reproducible weight initialization
    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print(f"Seed: {seed}")

    batch_size = 32
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
        h5_path=h5_path,
        sequence_length=sequence_length,
        label_fraction=label_fraction
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
    action_classifier = torch.nn.Linear(lam_latent_dim_actions, num_actions).to(device)

    optimizer = torch.optim.Adam(list(lam.parameters()) + list(action_classifier.parameters()), lr=learning_rate)

    start_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        (lam.module if ddp else lam).load_state_dict(ckpt['model_state_dict'])
        if 'classifier_state_dict' in ckpt:
            action_classifier.load_state_dict(ckpt['classifier_state_dict'])
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

        for batch_idx, (videos, game_actions, mask) in enumerate(dataloader):
            batch_start_time = time.time()
            B = videos.shape[0]
            videos = videos.to(device)
            game_actions = game_actions.to(device)
            mask = mask[:, 1:].to(device)  # (B, T-1) — align with actions

            # LAM forward
            reconstructed, actions = lam(videos)  # (B, T-1, C, H, W), (B, T-1, latent_dim_actions)
            recon_loss = F.mse_loss(reconstructed, videos[:, 1:])

            # Supervised loss on labelled samples only
            logits = action_classifier(actions)          # (B, T-1, num_actions)
            labels = game_actions[:, 1:]                 # (B, T-1)
            flat_mask = mask.reshape(-1)
            if flat_mask.any():
                supervised_loss = F.cross_entropy(logits.reshape(-1, num_actions)[flat_mask], labels.reshape(-1)[flat_mask])
            else:
                supervised_loss = torch.tensor(0.0, device=device)

            loss = recon_loss + sup_weight * supervised_loss

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
                'classifier_state_dict': action_classifier.state_dict(),
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
    parser.add_argument('--data', type=str, default='/datasets/health-gathering/vizdoom_healthgathering_dqn.h5')
    parser.add_argument('--num-actions', type=int, default=3, help='Number of game actions (4 for HealthGathering, 3 for TakeCover)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--sup-weight', type=float, default=1.0)
    parser.add_argument('--label-fraction', type=float, default=0.1, help='Fraction of frames with labelled actions (0.0-1.0)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (auto-generated if not set)')
    args = parser.parse_args()
    train(resume=args.resume, h5_path=args.data, num_actions=args.num_actions, num_epochs=args.epochs, sup_weight=args.sup_weight, label_fraction=args.label_fraction, seed=args.seed)
    if dist.is_initialized():
        dist.destroy_process_group()
