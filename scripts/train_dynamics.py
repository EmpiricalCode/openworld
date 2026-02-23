import math
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
import modal

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


def train(resume=None, h5_path='data/vizdoom_healthgathering/vizdoom_healthgathering_dqn.h5', tokenizer_ckpt=None, lam_ckpt=None, num_epochs=10, volume_name=None):
   
    vol = modal.Volume.from_name(volume_name, create_if_missing=True)
   
    # Shared
    batch_size = 32
    learning_rate = 1e-4
    sequence_length = 16
    img_size = (64, 64)
    patch_size = 4
    in_channels = 3

    # VideoTokenizer (frozen) — must match checkpoint
    tokenizer_embed_dim = 128
    tokenizer_latent_dim = 5   # FSQ latent dim for frame tokens
    tokenizer_num_bins = 4     # FSQ bins for frame tokens → 4^5 = 1024 codes

    # LatentActionModel (frozen) — must match checkpoint
    lam_embed_dim = 128
    lam_latent_dim_actions = 3  # FSQ latent dim for action tokens

    # DynamicsModel
    dynamics_embed_dim = 216

    tokenizer_checkpoint = tokenizer_ckpt
    lam_checkpoint = lam_ckpt

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
        h5_path=h5_path,
        sequence_length=sequence_length
    )
    sampler = DistributedSampler(dataset) if ddp else None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=0)

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

    # Frozen pre-trained LAM
    lam = LatentActionModel(
        img_size=img_size,
        patch_size=8,
        in_channels=in_channels,
        num_frames=sequence_length,
        embed_dim=lam_embed_dim,
        latent_dim=tokenizer_latent_dim,
        latent_dim_actions=lam_latent_dim_actions,
        num_bins=tokenizer_num_bins
    ).to(device)
    checkpoint = torch.load(lam_checkpoint, map_location=device)
    lam.load_state_dict(checkpoint['model_state_dict'])
    for p in lam.parameters():
        p.requires_grad = False
    lam.eval()
    if is_main:
        print(f"Loaded frozen LAM from {lam_checkpoint}")

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
        dynamics_model = DDP(dynamics_model, device_ids=[local_rank])

    # FSQ for converting tokenizer latents to target indices (no trainable params)
    fsq = FSQ(latent_dim=tokenizer_latent_dim, num_bins=tokenizer_num_bins).to(device)

    # Separate params: no weight decay on biases and norm weights
    decay_params = []
    no_decay_params = []
    for name, param in dynamics_model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith('.bias'):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    dynamics_optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': 0.01},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=learning_rate)

    # Warmup + cosine LR schedule
    warmup_steps = 500
    total_steps = num_epochs * len(dataloader)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        min_lr_ratio = 0.1  # floor at 10% of peak LR
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(dynamics_optimizer, lr_lambda)

    # Mixed precision (bf16 — no scaler needed, same dynamic range as fp32)
    use_amp = device.type == 'cuda'

    start_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        (dynamics_model.module if ddp else dynamics_model).load_state_dict(ckpt['model_state_dict'])
        dynamics_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        if is_main:
            print(f"Resumed from {resume} (epoch {ckpt['epoch']+1})")

    if is_main:
        print(f"DynamicsModel parameters: {sum(p.numel() for p in dynamics_model.parameters()):,}")
        print(f"Starting training...")
        print()

    os.makedirs('checkpoints', exist_ok=True)

    T = sequence_length

    for epoch in range(start_epoch, num_epochs):
        dynamics_model.train()

        total_dynamics_loss = 0

        if ddp:
            sampler.set_epoch(epoch)

        for batch_idx, videos in enumerate(dataloader):
            batch_start_time = time.time()
            B = videos.shape[0]
            videos = videos.to(device)  # (B, T, C, H, W)

            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                # Step A: get latents from frozen tokenizer
                latents = video_tokenizer.encoder(videos)  # (B, T, N, latent_dim)

                # Step B: get action tokens from frozen LAM
                _, actions = lam(videos)  # (B, T-1, latent_dim_actions)

                # Step C: prepare dynamics inputs — shift actions right by 1
                null_action = torch.zeros(B, 1, lam_latent_dim_actions, device=device)
                a_shifted = torch.cat([null_action, actions], dim=1)  # (B, T, latent_dim_actions)

                targets = fsq.latent_to_index(latents)  # (B, T, N)

            lengths = torch.full((B,), T, dtype=torch.long, device=device)

            # Step D: dynamics forward + loss
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                _, dynamics_loss = dynamics_model(latents, a_shifted, lengths, targets=targets, training=True)

            # Step E: backward
            dynamics_optimizer.zero_grad()
            dynamics_loss.backward()
            torch.nn.utils.clip_grad_norm_(dynamics_model.parameters(), max_norm=1.0)
            dynamics_optimizer.step()
            scheduler.step()

            batch_time = time.time() - batch_start_time
            total_dynamics_loss += dynamics_loss.item()

            if is_main:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], "
                      f"Dynamics Loss: {dynamics_loss.item():.6f}, Time: {batch_time:.3f}s")

        avg_dynamics_loss = total_dynamics_loss / len(dataloader)

        if is_main:
            print(f"Epoch [{epoch+1}/{num_epochs}] Avg Dynamics Loss: {avg_dynamics_loss:.6f}")
            print()

            dynamics_state = dynamics_model.module.state_dict() if ddp else dynamics_model.state_dict()

            torch.save({
                'epoch': epoch,
                'model_state_dict': dynamics_state,
                'optimizer_state_dict': dynamics_optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'dynamics_loss': avg_dynamics_loss,
            }, f'checkpoints/dynamics_epoch_{epoch+1}.pt')
            print(f"Checkpoint saved: dynamics_epoch_{epoch+1}.pt")
            if volume_name:
                vol.put_file(f'checkpoints/dynamics_epoch_{epoch+1}.pt', f'/dynamics_epoch_{epoch+1}.pt')
                print(f"Uploaded to volume: {volume_name}")
            print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--data', type=str, default='data/vizdoom_healthgathering/vizdoom_healthgathering_dqn.h5')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to video tokenizer checkpoint')
    parser.add_argument('--lam', type=str, required=True, help='Path to LAM checkpoint')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--volume', type=str, default=None, help='Modal volume name to upload checkpoints to')
    args = parser.parse_args()
    train(resume=args.resume, h5_path=args.data, tokenizer_ckpt=args.tokenizer, lam_ckpt=args.lam, num_epochs=args.epochs, volume_name=args.volume)
    if dist.is_initialized():
        dist.destroy_process_group()
