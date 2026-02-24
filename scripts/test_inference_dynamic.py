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
    (1, 10),   # left
    (0, 30),   # noop
    (2, 30),   # right
    (0, 30),   # noop
]


def action_index_to_latent(game_action, fsq, device):
    lam_token_idx = GAME_ACTION_TO_LAM_TOKEN[game_action]
    idx = torch.tensor([[lam_token_idx]], device=device)
    return fsq.index_to_latent(idx)[0, 0]


def generate_video_dynamic(dynamics_model, video_tokenizer, seed_latent,
                           action_schedule, action_fsq, device):
    """
    Autoregressively generate frames with a changing action schedule.
    Starts from a single seed frame, grows context up to 16, then slides.

    Args:
        seed_latent: (1, 1, N, latent_dim) — single seed frame latent
        action_schedule: list of (game_action_id, num_frames)
    """
    num_frames = dynamics_model.num_frames  # 16
    N = seed_latent.shape[2]
    latent_dim = seed_latent.shape[-1]
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

    # Growing lists of context latents and actions
    context_latents = [seed_latent[0, 0]]  # list of (N, latent_dim)
    context_actions = [torch.zeros(lam_latent_dim_actions, device=device)]  # null for seed

    # Decode seed frame
    frames = []
    with torch.no_grad():
        x_pad = torch.zeros(1, num_frames, N, latent_dim, device=device)
        x_pad[0, 0] = context_latents[0]
        decoded = video_tokenizer.decoder(x_pad)
        frame = decoded[0, 0].permute(1, 2, 0).cpu().numpy()
        frame = (frame.clip(0, 1) * 255).astype(np.uint8)
        frames.append(frame)

    # Print schedule
    print(f"  Action schedule ({total_steps} steps):")
    cumulative = 0
    for action_id, count in action_schedule:
        print(f"    Steps {cumulative}-{cumulative+count-1}: {ACTION_NAMES[action_id]}")
        cumulative += count

    for step in range(total_steps):
        action_id = step_actions[step]
        current_action = action_latents[action_id]
        ctx_len = len(context_latents)

        # Build 16-frame input: context frames packed at the front
        x_input = torch.zeros(1, num_frames, N, latent_dim, device=device)
        for i, lat in enumerate(context_latents):
            x_input[0, i] = lat

        # Build 16-frame action tensor: context actions + current at prediction slot
        actions = torch.zeros(1, num_frames, lam_latent_dim_actions, device=device)
        for i, act in enumerate(context_actions):
            actions[0, i] = act
        actions[0, ctx_len] = current_action

        # Predict at index ctx_len (next frame after context)
        lengths = torch.tensor([ctx_len], dtype=torch.long, device=device)

        with torch.no_grad():
            next_latent = dynamics_model(x_input, actions, lengths, training=False)

        # Append to context
        context_latents.append(next_latent[0])
        context_actions.append(current_action)

        # If over 16, slide
        if len(context_latents) > num_frames:
            context_latents = context_latents[1:]
            context_actions = context_actions[1:]

        # Decode with current context, take the last frame
        ctx_len_now = len(context_latents)
        x_decode = torch.zeros(1, num_frames, N, latent_dim, device=device)
        for i, lat in enumerate(context_latents):
            x_decode[0, i] = lat

        with torch.no_grad():
            decoded_window = video_tokenizer.decoder(x_decode)
        frame = decoded_window[0, ctx_len_now - 1].permute(1, 2, 0).cpu().numpy()
        frame = (frame.clip(0, 1) * 255).astype(np.uint8)
        frames.append(frame)

        if (step + 1) % 25 == 0:
            print(f"    Step {step + 1}/{total_steps} ({ACTION_NAMES[action_id]}, ctx={ctx_len_now})")

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
    dynamics_embed_dim = 216

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

    # Load single seed frame
    print(f"Loading seed frame from {h5_path}...")
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, 'r') as f:
        total_frames = f['frames'].shape[0]
        start_idx = rng.integers(0, total_frames)
        seed_frame = f['frames'][start_idx:start_idx + 1]  # (1, H, W, C)

    print(f"Seed frame: index {start_idx}")
    seed_frame_f = seed_frame.astype(np.float32) / 255.0
    seed_tensor = torch.from_numpy(seed_frame_f).permute(0, 3, 1, 2).unsqueeze(0).to(device)  # (1, 1, C, H, W)

    with torch.no_grad():
        seed_latent = video_tokenizer.encoder(seed_tensor)  # (1, 1, N, latent_dim)

    print(f"Seed latent shape: {seed_latent.shape}")

    # Generate
    print("\nGenerating dynamic action video...")
    video_frames = generate_video_dynamic(
        dynamics_model=dynamics_model,
        video_tokenizer=video_tokenizer,
        seed_latent=seed_latent,
        action_schedule=ACTION_SCHEDULE,
        action_fsq=action_fsq,
        device=device,
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
