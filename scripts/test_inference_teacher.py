"""
Teacher-forced inference: slide a 16-frame window of ground truth pixels,
encode each window through tokenizer + LAM, predict the 16th frame via MaskGIT,
decode it, and append to the output video.

Produces side-by-side video: ground truth (left) vs predicted (right).
First 15 frames are ground truth only (no prediction yet).
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


def decode_frame(video_tokenizer, latent):
    """Decode a single latent frame to uint8 numpy. latent: (1, 1, N, latent_dim)"""
    px = video_tokenizer.decoder(latent)  # (1, 1, C, H, W)
    frame = px[0, 0].permute(1, 2, 0).cpu().numpy()
    return (frame.clip(0, 1) * 255).astype(np.uint8)


def save_video(frames, path, fps=10):
    H, W, C = frames[0].shape
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  Saved {len(frames)} frames to {path}")


def save_side_by_side(gt_frames, pred_frames, path, fps=10):
    H, W, C = gt_frames[0].shape
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W * 2, H))
    for gt, pred in zip(gt_frames, pred_frames):
        combined = np.concatenate([gt, pred], axis=1)  # (H, 2W, C)
        writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  Saved {len(gt_frames)} side-by-side frames to {path}")


def main(dynamics_checkpoint, tokenizer_checkpoint, lam_checkpoint,
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
    lam_embed_dim = 128
    lam_latent_dim_actions = 3
    dynamics_embed_dim = 264

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

    # Load contiguous ground truth frames
    # Need 15 context + num_steps frames to predict = 15 + num_steps total
    total_needed = num_steps + (sequence_length - 1)
    print(f"Loading {total_needed} contiguous frames from {h5_path}...")
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, 'r') as f:
        total_frames = f['frames'].shape[0]
        start_idx = rng.integers(0, total_frames - total_needed)
        raw_frames = f['frames'][start_idx:start_idx + total_needed]  # (T_total, H, W, C) uint8

    print(f"Sequence: frames {start_idx} to {start_idx + total_needed - 1}")

    frames_f = raw_frames.astype(np.float32) / 255.0
    frames_tensor = torch.from_numpy(frames_f).permute(0, 3, 1, 2).to(device)  # (T_total, C, H, W)

    gt_frames = []
    pred_frames = []

    # First 15 frames: ground truth only (no prediction possible yet)
    print("Decoding first 15 ground truth frames...")
    with torch.no_grad():
        for t in range(sequence_length - 1):
            frame = (frames_f[t] * 255).astype(np.uint8)  # (H, W, C) — already in HWC
            gt_frames.append(frame)
            pred_frames.append(frame)  # mirror GT for side-by-side alignment

    # Sliding window: for each step, take 16 GT frames, encode, predict 16th, decode
    print(f"Running teacher-forced generation for {num_steps} steps...")
    with torch.no_grad():
        for step in range(num_steps):
            # Window of 16 ground truth pixel frames
            t = step + (sequence_length - 1)  # index of the frame we're predicting
            window_start = step
            window = frames_tensor[window_start:window_start + sequence_length].unsqueeze(0)  # (1, 16, C, H, W)

            # Encode through tokenizer to get latents for all 16 frames
            latents = video_tokenizer.encoder(window)  # (1, 16, N, latent_dim)

            # Get actions from LAM for this window
            _, actions_raw = lam(window)  # (1, 15, latent_dim_actions)

            # Build action tensor: null at position 0, then LAM actions at 1..15
            actions = torch.zeros(1, sequence_length, lam_latent_dim_actions, device=device)
            actions[:, 1:] = actions_raw

            # Predict the 16th frame (index 15) via MaskGIT
            lengths = torch.tensor([sequence_length - 1], dtype=torch.long, device=device)
            pred_latent = dynamics_model(latents, actions, lengths, training=False)  # (1, N, latent_dim)

            # Decode predicted frame
            pred_frame = decode_frame(video_tokenizer, pred_latent.unsqueeze(1))
            pred_frames.append(pred_frame)

            # Ground truth frame at position t
            gt_frame = (frames_f[t] * 255).astype(np.uint8)
            gt_frames.append(gt_frame)

            if (step + 1) % 50 == 0:
                print(f"    Step {step + 1}/{num_steps}")

    print(f"Total frames: {len(gt_frames)} ({sequence_length - 1} context + {num_steps} predicted)")

    # Save videos
    save_side_by_side(gt_frames, pred_frames,
                      os.path.join(output_dir, 'teacher_forced_comparison.mp4'), fps=10)
    save_video(gt_frames, os.path.join(output_dir, 'teacher_forced_gt.mp4'), fps=10)
    save_video(pred_frames, os.path.join(output_dir, 'teacher_forced_pred.mp4'), fps=10)

    # Upload to Modal volume if requested
    if volume_name:
        import modal
        vol = modal.Volume.from_name(volume_name, create_if_missing=True)
        with vol.batch_upload() as batch:
            for fname in ['teacher_forced_comparison.mp4', 'teacher_forced_gt.mp4', 'teacher_forced_pred.mp4']:
                local_path = os.path.join(output_dir, fname)
                remote_path = f'/inference/{fname}'
                batch.put_file(local_path, remote_path)
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
    parser.add_argument('--steps', type=int, default=100)
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
