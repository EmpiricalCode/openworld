import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import h5py
import sys
import os
import time
from torch.utils.data import Dataset, DataLoader
import modal
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model.dynamics_model import DynamicsModel


class PrecomputedDataset(Dataset):
    def __init__(self, h5_path):
        """Load precomputed latents and actions from h5 file."""
        print(f"Loading precomputed dataset from {h5_path}...")
        with h5py.File(h5_path, 'r') as f:
            self.all_latents = torch.from_numpy(f['latents'][:])
            self.all_actions = torch.from_numpy(f['actions'][:])
            self.all_targets = torch.from_numpy(f['targets'][:])

        total_mb = (self.all_latents.nbytes + self.all_actions.nbytes + self.all_targets.nbytes) / (1024**2)
        print(f"Loaded {len(self)} sequences ({total_mb:.1f} MB)")

    def __len__(self):
        return self.all_latents.shape[0]

    def __getitem__(self, idx):
        return self.all_latents[idx], self.all_actions[idx], self.all_targets[idx]


def train(resume=None, h5_path=None, num_epochs=10, volume_name=None):
   
    if volume_name:
        vol = modal.Volume.from_name(volume_name, create_if_missing=True)

    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core', 'config', 'dynamics.json')
    with open(config_path) as f:
        cfg = json.load(f)

    batch_size = cfg['batch_size']
    learning_rate = cfg['learning_rate']
    sequence_length = cfg['sequence_length']
    img_size = tuple(cfg['img_size'])
    patch_size = cfg['patch_size']
    in_channels = cfg['in_channels']
    tokenizer_latent_dim = cfg['tokenizer_latent_dim']
    tokenizer_num_bins = cfg['tokenizer_num_bins']
    lam_latent_dim_actions = cfg['lam_latent_dim_actions']
    dynamics_embed_dim = cfg['dynamics_embed_dim']
    dynamics_num_blocks = cfg['num_blocks']
    dynamics_num_heads = cfg['num_heads']

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

    num_patches_x = img_size[1] // patch_size
    num_patches_y = img_size[0] // patch_size

    dataset = PrecomputedDataset(h5_path)

    sampler = DistributedSampler(dataset) if ddp else None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=0)

    # Dynamics model
    dynamics_model = DynamicsModel(
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
        in_channels=in_channels,
        num_frames=sequence_length,
        embed_dim=dynamics_embed_dim,
        latent_dim=tokenizer_latent_dim,
        latent_dim_actions=lam_latent_dim_actions,
        num_bins=tokenizer_num_bins,
        num_blocks=dynamics_num_blocks,
        num_heads=dynamics_num_heads
    ).to(device)

    if ddp:
        dynamics_model = DDP(dynamics_model, device_ids=[local_rank])

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

    # Warmup-Stable-Decay (WSD) schedule with 20% cooldown
    warmup_steps = 500
    total_steps = num_epochs * len(dataloader)
    cooldown_steps = int(0.2 * total_steps)
    stable_steps = total_steps - warmup_steps - cooldown_steps
    min_lr_ratio = 0.1
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        elif step < warmup_steps + stable_steps:
            return 1.0
        else:
            progress = (step - warmup_steps - stable_steps) / max(1, cooldown_steps)
            return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(dynamics_optimizer, lr_lambda)

    # Mixed precision (bf16 — no scaler needed, same dynamic range as fp32)
    use_amp = device.type == 'cuda'

    start_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        (dynamics_model.module if ddp else dynamics_model).load_state_dict(ckpt['model_state_dict'])
        dynamics_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        # Fast-forward scheduler to match resumed epoch (fresh schedule, no stale state)
        steps_to_skip = start_epoch * len(dataloader)
        for _ in range(steps_to_skip):
            scheduler.step()
        if is_main:
            print(f"Resumed from {resume} (epoch {ckpt['epoch']+1}), scheduler fast-forwarded {steps_to_skip} steps")

    if is_main:
        print(f"DynamicsModel parameters: {sum(p.numel() for p in dynamics_model.parameters()):,}")
        print(f"Starting training...")
        print()

    os.makedirs('checkpoints', exist_ok=True)

    T = sequence_length

    for epoch in range(start_epoch, num_epochs):
        dynamics_model.train()

        total_dynamics_loss = 0
        window_batch_time = 0

        if ddp:
            sampler.set_epoch(epoch)

        for batch_idx, (latents, a_shifted, targets) in enumerate(dataloader):
            batch_start_time = time.time()
            B = latents.shape[0]
            latents = latents.to(device)       # (B, T, N, latent_dim)
            a_shifted = a_shifted.to(device)   # (B, T, latent_dim_actions)
            targets = targets.to(device)       # (B, T, N)

            lengths = torch.full((B,), T, dtype=torch.long, device=device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                _, dynamics_loss = dynamics_model(latents, a_shifted, lengths, targets=targets, training=True)

            dynamics_optimizer.zero_grad()
            dynamics_loss.backward()
            torch.nn.utils.clip_grad_norm_(dynamics_model.parameters(), max_norm=1.0)
            dynamics_optimizer.step()
            scheduler.step()

            batch_time = time.time() - batch_start_time
            total_dynamics_loss += dynamics_loss.item()
            window_batch_time += batch_time

            if is_main and (batch_idx + 1) % 100 == 0:
                avg_batch_time = window_batch_time / 100
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], "
                      f"Dynamics Loss: {dynamics_loss.item():.6f}, Avg Time: {avg_batch_time:.3f}s/batch")
                window_batch_time = 0

        avg_dynamics_loss = total_dynamics_loss / len(dataloader)

        if is_main:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch+1}/{num_epochs}] Avg Dynamics Loss: {avg_dynamics_loss:.6f}, LR: {current_lr:.6e} ({current_lr/learning_rate:.2%} of peak)")
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
                with vol.batch_upload(force=True) as batch:
                    batch.put_file(f'checkpoints/dynamics_epoch_{epoch+1}.pt', f'dynamics_epoch_{epoch+1}.pt')
                print(f"Uploaded to volume: {volume_name}")
            print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--data', type=str, required=True, help='Path to precomputed latents h5 file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--volume', type=str, default=None, help='Modal volume name to upload checkpoints to')
    args = parser.parse_args()
    train(resume=args.resume, h5_path=args.data, num_epochs=args.epochs, volume_name=args.volume)
    if dist.is_initialized():
        dist.destroy_process_group()
