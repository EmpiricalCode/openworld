"""
Generate videos by rolling out the dynamics model with a fixed action.
Saves one video per action (0=noop, 1=forward, 2=right, 3=left).

Videos are saved to a Modal volume (default: outputs/inference/).
"""
import torch
import numpy as np
import sys
import os
import h5py
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model.video_tokenizer import VideoTokenizer
from core.model.latent_action_model import LatentActionModel
from core.model.dynamics_model import DynamicsModel


ACTION_NAMES = {0: 'noop', 1: 'forward', 2: 'right', 3: 'left'}

# Empirical mapping from game action index to LAM token index
GAME_ACTION_TO_LAM_TOKEN = {0: 6, 1: 5, 2: 0, 3: 3}


def lam_token_for_action(game_action, lam, device):
    """
    Returns the FSQ latent vector for a given game action using the LAM's quantizer.
    """
    lam_token_idx = GAME_ACTION_TO_LAM_TOKEN[game_action]
    num_codes = lam.action_quantizer.num_bins ** lam.action_quantizer.latent_dim
    all_indices = torch.arange(num_codes, device=device)
    all_latents = lam.action_quantizer.index_to_latent(all_indices.unsqueeze(0))  # (1, num_codes, latent_dim)
    return all_latents[0, lam_token_idx]  # (latent_dim_actions,)


def generate_video(dynamics_model, video_tokenizer, seed_latents, action_latent,
                   num_steps, device):
    """
    Autoregressively generate num_steps frames given seed latents and a fixed action.

    Args:
        seed_latents: (1, T_seed, N, latent_dim) — context frames to seed generation
        action_latent: (latent_dim_actions,) — fixed action to repeat

    Returns:
        frames: list of (H, W, C) uint8 numpy arrays
    """
    N = seed_latents.shape[2]

    # Decode seed frame to pixels for the output video
    with torch.no_grad():
        seed_pixels = video_tokenizer.decoder(seed_latents[:, :1])  # (1, 1, C, H, W)
    frames = []
    frame = seed_pixels[0, 0].permute(1, 2, 0).cpu().numpy()
    frame = (frame.clip(0, 1) * 255).astype(np.uint8)
    frames.append(frame)

    num_frames = dynamics_model.num_frames
    lam_latent_dim_actions = action_latent.shape[0]
    fixed_action = action_latent.reshape(1, -1).to(device)  # (1, L)

    latent_dim = seed_latents.shape[-1]
    x_input = torch.zeros(1, num_frames, N, latent_dim, device=device)
    x_input[0, 0] = seed_latents[0, 0]  # seed frame at position 0

    gen_idx = 1

    for step in range(num_steps):
        # Build action tensor: null at 0, fixed_action at positions 1..gen_idx
        actions = torch.zeros(1, num_frames, lam_latent_dim_actions, device=device)
        actions[0, 1:gen_idx + 1] = fixed_action

        lengths = torch.tensor([gen_idx], dtype=torch.long, device=device)

        with torch.no_grad():
            next_latent = dynamics_model(x_input.clone(), actions, lengths, training=False)  # (1, N, latent_dim)

        # Place generated frame into x_input
        x_input[0, gen_idx] = next_latent[0]

        # Decode to pixel frame
        with torch.no_grad():
            next_pixel = video_tokenizer.decoder(next_latent.unsqueeze(1))  # (1, 1, C, H, W)
        frame = next_pixel[0, 0].permute(1, 2, 0).cpu().numpy()
        frame = (frame.clip(0, 1) * 255).astype(np.uint8)
        frames.append(frame)

        if (step + 1) % 50 == 0:
            print(f"    Step {step + 1}/{num_steps}")

        # Advance gen_idx; once window is full, slide everything left
        gen_idx += 1
        if gen_idx >= num_frames:
            x_input[0, :-1] = x_input[0, 1:].clone()
            x_input[0, -1] = 0
            gen_idx = num_frames - 1

    return frames


def save_video(frames, path, fps=10):
    H, W, C = frames[0].shape
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  Saved {len(frames)} frames to {path}")


def main(dynamics_checkpoint, tokenizer_checkpoint, lam_checkpoint,
         h5_path='data/vizdoom_healthgathering/vizdoom_healthgathering_dqn.h5',
         num_steps=300, output_dir='outputs/inference', seed=42, volume_name=None):

    # Config — must match training
    sequence_length = 16
    img_size = (64, 64)
    patch_size = 4
    in_channels = 3
    tokenizer_embed_dim = 128
    tokenizer_latent_dim = 5
    tokenizer_num_bins = 4
    lam_embed_dim = 128
    lam_latent_dim_actions = 3
    dynamics_embed_dim = 192

    num_patches_x = img_size[1] // patch_size
    num_patches_y = img_size[0] // patch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)

    # Load VideoTokenizer
    video_tokenizer = VideoTokenizer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_frames=sequence_length,
        embed_dim=tokenizer_embed_dim,
        latent_dim=tokenizer_latent_dim
    ).to(device)
    ckpt = torch.load(tokenizer_checkpoint, map_location=device)
    video_tokenizer.load_state_dict(ckpt['model_state_dict'])
    video_tokenizer.eval()
    print(f"Loaded VideoTokenizer from {tokenizer_checkpoint}")

    # Load LAM
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
    ckpt = torch.load(lam_checkpoint, map_location=device)
    lam.load_state_dict(ckpt['model_state_dict'])
    lam.eval()
    print(f"Loaded LAM from {lam_checkpoint}")

    # Load DynamicsModel
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
    ckpt = torch.load(dynamics_checkpoint, map_location=device)
    dynamics_model.load_state_dict(ckpt['model_state_dict'])
    dynamics_model.eval()
    print(f"Loaded DynamicsModel from {dynamics_checkpoint}")
    print(f"  Epoch: {ckpt['epoch'] + 1}, Dynamics loss: {ckpt['dynamics_loss']:.6f}")
    print()

    # Load a seed sequence from the dataset
    print(f"Loading seed frames from {h5_path}...")
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, 'r') as f:
        total_frames = f['frames'].shape[0]
        start_idx = rng.integers(0, total_frames - sequence_length)
        seed_frames = f['frames'][start_idx:start_idx + sequence_length]  # (T, H, W, C) uint8

    print(f"Seed sequence starting at frame {start_idx}")
    seed_frames_f = seed_frames.astype(np.float32) / 255.0
    seed_tensor = torch.from_numpy(seed_frames_f).permute(0, 3, 1, 2).unsqueeze(0).to(device)  # (1, T, C, H, W)

    with torch.no_grad():
        seed_latents = video_tokenizer.encoder(seed_tensor)  # (1, T, N, latent_dim)

    print(f"Seed latents shape: {seed_latents.shape}")

    # Generate one video per action
    for game_action in range(4):
        action_name = ACTION_NAMES[game_action]
        print(f"\nGenerating video for action {game_action} ({action_name})...")

        action_latent = lam_token_for_action(game_action, lam, device)

        video_frames = generate_video(
            dynamics_model=dynamics_model,
            video_tokenizer=video_tokenizer,
            seed_latents=seed_latents.clone(),
            action_latent=action_latent,
            num_steps=num_steps,
            device=device
        )

        out_path = os.path.join(output_dir, f'action_{game_action}_{action_name}.mp4')
        save_video(video_frames, out_path, fps=30)

    # Upload to Modal volume if requested
    if volume_name:
        import modal
        vol = modal.Volume.from_name(volume_name, create_if_missing=True)
        for game_action in range(4):
            action_name = ACTION_NAMES[game_action]
            local_path = os.path.join(output_dir, f'action_{game_action}_{action_name}.mp4')
            remote_path = f'inference/action_{game_action}_{action_name}.mp4'
            with open(local_path, 'rb') as f:
                vol.write_file(remote_path, f)
            print(f"  Uploaded to volume: {remote_path}")
        print(f"\nDownload with: modal volume get {volume_name} inference/ ./")

    print(f"\nDone. Videos saved to {output_dir}/")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dynamics', required=True, help='Path to dynamics model checkpoint')
    parser.add_argument('--tokenizer', required=True, help='Path to video tokenizer checkpoint')
    parser.add_argument('--lam', required=True, help='Path to LAM checkpoint')
    parser.add_argument('--h5', default='data/vizdoom_healthgathering/vizdoom_healthgathering_dqn.h5')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='outputs/inference')
    parser.add_argument('--volume', type=str, default=None, help='Modal volume name to upload videos to')
    args = parser.parse_args()

    main(
        dynamics_checkpoint=args.dynamics,
        tokenizer_checkpoint=args.tokenizer,
        lam_checkpoint=args.lam,
        h5_path=args.h5,
        num_steps=args.steps,
        output_dir=args.output_dir,
        seed=args.seed,
        volume_name=args.volume,
    )
