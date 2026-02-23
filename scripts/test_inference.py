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
from core.model.dynamics_model import DynamicsModel
from core.model.components.quantization import FSQ


ACTION_NAMES = {0: 'noop', 1: 'left', 2: 'right'}

# Empirical mapping from game action index to LAM token index (TakeCover)
GAME_ACTION_TO_LAM_TOKEN = {0: 3, 1: 1, 2: 2}


def action_index_to_latent(game_action, fsq, device):
    """
    Returns the FSQ latent vector for a given game action.
    Uses a standalone FSQ — no LAM model needed.
    """
    lam_token_idx = GAME_ACTION_TO_LAM_TOKEN[game_action]
    num_codes = fsq.num_bins ** fsq.latent_dim
    all_indices = torch.arange(num_codes, device=device)
    all_latents = fsq.index_to_latent(all_indices.unsqueeze(0))  # (1, num_codes, latent_dim)
    return all_latents[0, lam_token_idx]  # (latent_dim_actions,)


def generate_video(dynamics_model, video_tokenizer, seed_latents, action_latent,
                   num_steps, device):
    """
    Autoregressively generate num_steps frames given seed latents and a fixed action.

    Uses all 16 seed frames as initial context (matching training distribution where
    the model always sees real latents at every position). Each generated frame replaces
    the last context frame, sliding the window forward.

    Args:
        seed_latents: (1, T_seed, N, latent_dim) — full window of context frames
        action_latent: (latent_dim_actions,) — fixed action to repeat

    Returns:
        frames: list of (H, W, C) uint8 numpy arrays
    """
    num_frames = dynamics_model.num_frames  # 16
    N = seed_latents.shape[2]
    latent_dim = seed_latents.shape[-1]
    lam_latent_dim_actions = action_latent.shape[0]
    fixed_action = action_latent.reshape(1, -1).to(device)  # (1, L)

    # Decode seed frames (first 15) using full context window
    frames = []
    with torch.no_grad():
        decoded = video_tokenizer.decoder(seed_latents)  # (1, 16, C, H, W)
        for t in range(num_frames - 1):
            frame = decoded[0, t].permute(1, 2, 0).cpu().numpy()
            frame = (frame.clip(0, 1) * 255).astype(np.uint8)
            frames.append(frame)

    # Start with full context window
    x_input = seed_latents.clone()  # (1, 16, N, latent_dim)

    for step in range(num_steps):
        # Actions: null at position 0, fixed_action at all other positions
        actions = torch.zeros(1, num_frames, lam_latent_dim_actions, device=device)
        actions[0, 1:] = fixed_action

        # Generate next frame at position num_frames-1 (last slot)
        # Use lengths = num_frames - 1 so the model generates at that index
        lengths = torch.tensor([num_frames - 1], dtype=torch.long, device=device)

        with torch.no_grad():
            next_latent = dynamics_model(x_input.clone(), actions, lengths, training=False)  # (1, N, latent_dim)

        # Slide window: drop oldest frame, append generated frame
        x_input[0, :-1] = x_input[0, 1:].clone()
        x_input[0, -1] = next_latent[0]

        # Decode full 16-frame context window, take last frame
        with torch.no_grad():
            decoded_window = video_tokenizer.decoder(x_input)  # (1, 16, C, H, W)
        frame = decoded_window[0, -1].permute(1, 2, 0).cpu().numpy()
        frame = (frame.clip(0, 1) * 255).astype(np.uint8)
        frames.append(frame)

        if (step + 1) % 50 == 0:
            print(f"    Step {step + 1}/{num_steps}")

    return frames


def save_video(frames, path, fps=10):
    H, W, C = frames[0].shape
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  Saved {len(frames)} frames to {path}")


def main(dynamics_checkpoint, tokenizer_checkpoint,
         h5_path='data/vizdoom_healthgathering/vizdoom_healthgathering_dqn.h5',
         num_steps=100, output_dir='outputs/inference', seed=42, volume_name=None):

    # Config — must match training
    sequence_length = 16
    img_size = (64, 64)
    patch_size = 4
    in_channels = 3
    tokenizer_embed_dim = 128
    tokenizer_latent_dim = 5
    tokenizer_num_bins = 4
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

    # Standalone FSQ for action tokens (no learned params, just needs matching config)
    action_fsq = FSQ(latent_dim=lam_latent_dim_actions, num_bins=tokenizer_num_bins).to(device)

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
    num_seed = sequence_length - 1  # 15 seed frames, 16th slot is for generation
    with h5py.File(h5_path, 'r') as f:
        total_frames = f['frames'].shape[0]
        start_idx = rng.integers(0, total_frames - num_seed)
        seed_frames = f['frames'][start_idx:start_idx + num_seed]  # (15, H, W, C) uint8

    print(f"Seed sequence: frames {start_idx} to {start_idx + num_seed - 1} ({num_seed} frames)")
    seed_frames_f = seed_frames.astype(np.float32) / 255.0
    seed_tensor = torch.from_numpy(seed_frames_f).permute(0, 3, 1, 2).unsqueeze(0).to(device)  # (1, 15, C, H, W)

    with torch.no_grad():
        seed_latents_15 = video_tokenizer.encoder(seed_tensor)  # (1, 15, N, latent_dim)

    # Pad to full 16-frame window (last slot will be overwritten by dynamics model)
    N = seed_latents_15.shape[2]
    latent_dim = seed_latents_15.shape[3]
    seed_latents = torch.zeros(1, sequence_length, N, latent_dim, device=device)
    seed_latents[:, :num_seed] = seed_latents_15

    print(f"Seed latents shape: {seed_latents.shape} (15 real + 1 placeholder)")

    # Generate one video per action
    for game_action in ACTION_NAMES:
        action_name = ACTION_NAMES[game_action]
        print(f"\nGenerating video for action {game_action} ({action_name})...")

        action_latent = action_index_to_latent(game_action, action_fsq, device)

        video_frames = generate_video(
            dynamics_model=dynamics_model,
            video_tokenizer=video_tokenizer,
            seed_latents=seed_latents.clone(),
            action_latent=action_latent,
            num_steps=num_steps,
            device=device
        )

        out_path = os.path.join(output_dir, f'action_{game_action}_{action_name}.mp4')
        save_video(video_frames, out_path, fps=10)

    # Upload to Modal volume if requested
    if volume_name:
        import modal
        vol = modal.Volume.from_name(volume_name, create_if_missing=True)
        with vol.batch_upload() as batch:
            for game_action in ACTION_NAMES:
                action_name = ACTION_NAMES[game_action]
                local_path = os.path.join(output_dir, f'action_{game_action}_{action_name}.mp4')
                remote_path = f'/inference/action_{game_action}_{action_name}.mp4'
                batch.put_file(local_path, remote_path)
                print(f"  Uploaded to volume: {remote_path}")
        print(f"\nDownload with: modal volume get {volume_name} inference/ ./")

    print(f"\nDone. Videos saved to {output_dir}/")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dynamics', required=True, help='Path to dynamics model checkpoint')
    parser.add_argument('--tokenizer', required=True, help='Path to video tokenizer checkpoint')
    parser.add_argument('--h5', default='data/vizdoom_healthgathering/vizdoom_healthgathering_dqn.h5')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='outputs/inference')
    parser.add_argument('--volume', type=str, default=None, help='Modal volume name to upload videos to')
    args = parser.parse_args()

    main(
        dynamics_checkpoint=args.dynamics,
        tokenizer_checkpoint=args.tokenizer,
        h5_path=args.h5,
        num_steps=args.steps,
        output_dir=args.output_dir,
        seed=args.seed,
        volume_name=args.volume,
    )
