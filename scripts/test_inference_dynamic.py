"""
Generate a single video with changing actions over time.
Action schedule: 50 right, 15 left, 20 noop, 50 left.
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
GAME_ACTION_TO_LAM_TOKEN = {0: 3, 1: 1, 2: 2}

# Action schedule: (action_id, num_frames)
ACTION_SCHEDULE = [
    (2, 50),   # right
    (1, 50),   # left
    (0, 50),   # noop
    (1, 50),   # left
]


def action_index_to_latent(game_action, fsq, device):
    lam_token_idx = GAME_ACTION_TO_LAM_TOKEN[game_action]
    num_codes = fsq.num_bins ** fsq.latent_dim
    all_indices = torch.arange(num_codes, device=device)
    all_latents = fsq.index_to_latent(all_indices.unsqueeze(0))
    return all_latents[0, lam_token_idx]


def generate_video_dynamic(dynamics_model, video_tokenizer, seed_latents,
                           action_schedule, action_fsq, device):
    """
    Autoregressively generate frames with a changing action schedule.

    Args:
        seed_latents: (1, 16, N, latent_dim)
        action_schedule: list of (game_action_id, num_frames)
    """
    num_frames = dynamics_model.num_frames  # 16
    N = seed_latents.shape[2]
    latent_dim = seed_latents.shape[-1]
    lam_latent_dim_actions = action_fsq.latent_dim

    # Pre-compute action latents
    action_latents = {}
    for action_id, _ in action_schedule:
        if action_id not in action_latents:
            action_latents[action_id] = action_index_to_latent(action_id, action_fsq, device)

    # Build flat list of per-step actions
    step_actions = []
    for action_id, count in action_schedule:
        step_actions.extend([action_id] * count)
    total_steps = len(step_actions)

    # Decode seed frames (first 15)
    frames = []
    with torch.no_grad():
        decoded = video_tokenizer.decoder(seed_latents)
        for t in range(num_frames - 1):
            frame = decoded[0, t].permute(1, 2, 0).cpu().numpy()
            frame = (frame.clip(0, 1) * 255).astype(np.uint8)
            frames.append(frame)

    x_input = seed_latents.clone()

    # Print schedule
    print(f"  Action schedule ({total_steps} steps):")
    cumulative = 0
    for action_id, count in action_schedule:
        print(f"    Steps {cumulative}-{cumulative+count-1}: {ACTION_NAMES[action_id]}")
        cumulative += count

    for step in range(total_steps):
        action_id = step_actions[step]
        fixed_action = action_latents[action_id].reshape(1, -1)

        actions = torch.zeros(1, num_frames, lam_latent_dim_actions, device=device)
        actions[0, 1:] = fixed_action

        lengths = torch.tensor([num_frames - 1], dtype=torch.long, device=device)

        with torch.no_grad():
            next_latent = dynamics_model(x_input.clone(), actions, lengths, training=False)

        x_input[0, :-1] = x_input[0, 1:].clone()
        x_input[0, -1] = next_latent[0]

        with torch.no_grad():
            decoded_window = video_tokenizer.decoder(x_input)
        frame = decoded_window[0, -1].permute(1, 2, 0).cpu().numpy()
        frame = (frame.clip(0, 1) * 255).astype(np.uint8)
        frames.append(frame)

        if (step + 1) % 25 == 0:
            print(f"    Step {step + 1}/{total_steps} ({ACTION_NAMES[action_id]})")

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
         output_dir='outputs/inference', seed=42, volume_name=None):

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
        img_size=img_size, patch_size=patch_size, in_channels=in_channels,
        num_frames=sequence_length, embed_dim=tokenizer_embed_dim,
        latent_dim=tokenizer_latent_dim
    ).to(device)
    ckpt = torch.load(tokenizer_checkpoint, map_location=device)
    video_tokenizer.load_state_dict(ckpt['model_state_dict'])
    video_tokenizer.eval()
    print(f"Loaded VideoTokenizer from {tokenizer_checkpoint}")

    action_fsq = FSQ(latent_dim=lam_latent_dim_actions, num_bins=tokenizer_num_bins).to(device)

    # Load DynamicsModel
    dynamics_model = DynamicsModel(
        num_patches_x=num_patches_x, num_patches_y=num_patches_y,
        in_channels=in_channels, num_frames=sequence_length,
        embed_dim=dynamics_embed_dim, latent_dim=tokenizer_latent_dim,
        latent_dim_actions=lam_latent_dim_actions, num_bins=tokenizer_num_bins
    ).to(device)
    ckpt = torch.load(dynamics_checkpoint, map_location=device)
    dynamics_model.load_state_dict(ckpt['model_state_dict'])
    dynamics_model.eval()
    print(f"Loaded DynamicsModel from {dynamics_checkpoint}")
    print(f"  Epoch: {ckpt['epoch'] + 1}, Dynamics loss: {ckpt['dynamics_loss']:.6f}")
    print()

    # Load seed sequence
    print(f"Loading seed frames from {h5_path}...")
    rng = np.random.default_rng(seed)
    num_seed = sequence_length - 1
    with h5py.File(h5_path, 'r') as f:
        total_frames = f['frames'].shape[0]
        start_idx = rng.integers(0, total_frames - num_seed)
        seed_frames = f['frames'][start_idx:start_idx + num_seed]

    print(f"Seed sequence: frames {start_idx} to {start_idx + num_seed - 1}")
    seed_frames_f = seed_frames.astype(np.float32) / 255.0
    seed_tensor = torch.from_numpy(seed_frames_f).permute(0, 3, 1, 2).unsqueeze(0).to(device)

    with torch.no_grad():
        seed_latents_15 = video_tokenizer.encoder(seed_tensor)

    N = seed_latents_15.shape[2]
    latent_dim = seed_latents_15.shape[3]
    seed_latents = torch.zeros(1, sequence_length, N, latent_dim, device=device)
    seed_latents[:, :num_seed] = seed_latents_15

    # Generate
    print("\nGenerating dynamic action video...")
    video_frames = generate_video_dynamic(
        dynamics_model=dynamics_model,
        video_tokenizer=video_tokenizer,
        seed_latents=seed_latents,
        action_schedule=ACTION_SCHEDULE,
        action_fsq=action_fsq,
        device=device
    )

    out_path = os.path.join(output_dir, 'dynamic_actions.mp4')
    save_video(video_frames, out_path, fps=10)

    if volume_name:
        import modal
        vol = modal.Volume.from_name(volume_name, create_if_missing=True)
        with vol.batch_upload() as batch:
            batch.put_file(out_path, '/inference/dynamic_actions.mp4')
        print(f"Uploaded to volume: {volume_name}")

    print(f"\nDone. Video saved to {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dynamics', required=True, help='Path to dynamics model checkpoint')
    parser.add_argument('--tokenizer', required=True, help='Path to video tokenizer checkpoint')
    parser.add_argument('--h5', default='data/vizdoom_healthgathering/vizdoom_healthgathering_dqn.h5')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='outputs/inference')
    parser.add_argument('--volume', type=str, default=None, help='Modal volume name to upload video to')
    args = parser.parse_args()

    main(
        dynamics_checkpoint=args.dynamics,
        tokenizer_checkpoint=args.tokenizer,
        h5_path=args.h5,
        output_dir=args.output_dir,
        seed=args.seed,
        volume_name=args.volume,
    )
