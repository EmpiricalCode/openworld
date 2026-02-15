import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import gymnasium as gym
import vizdoom.gymnasium_wrapper  # Register ViZDoom environments
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model.video_tokenizer import VideoTokenizer

def visualize_reconstruction(checkpoint_path='checkpoints/video_tokenizer_epoch_10.pt',
                            sequence_length=8,
                            rollout_length=100):
    """
    Generate a longer rollout from the environment, sample random frames from it, reconstruct them, and display original vs reconstructed frames.

    Args:
        checkpoint_path: Path to the model checkpoint
        sequence_length: Number of consecutive frames to reconstruct (for the tokenizer)
        rollout_length: Total number of frames to collect before sampling
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = VideoTokenizer(
        img_size=(128, 128),
        patch_size=8,
        in_channels=3,
        num_frames=sequence_length,
        embed_dim=128
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
    print(f"Checkpoint loss: {checkpoint['loss']:.6f}")

    # Generate a longer rollout from the environment
    print(f"\nGenerating {rollout_length}-frame rollout from ViZDoom HealthGathering environment...")
    env = gym.make("VizdoomHealthGathering-v1", render_mode="rgb_array")
    observation, info = env.reset()

    all_frames = []
    for step in range(rollout_length):
        # Take random action
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        # Get frame and resize to 128x128
        frame = env.render()
        frame_resized = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
        all_frames.append(frame_resized)

        # Reset if episode ends
        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    all_frames = np.array(all_frames)  # (rollout_length, H, W, C)
    print(f"Generated {rollout_length} frames from rollout")

    # Sample random starting point for the sequence
    max_start_idx = rollout_length - sequence_length
    start_idx = np.random.randint(0, max_start_idx + 1)
    print(f"Sampling frames {start_idx} to {start_idx + sequence_length - 1} from rollout")

    frames = all_frames[start_idx:start_idx + sequence_length]  # (sequence_length, H, W, C)

    # Preprocess: normalize to [0, 1] and convert to tensor
    frames_normalized = frames.astype(np.float32) / 255.0
    frames_tensor = torch.from_numpy(frames_normalized).permute(0, 3, 1, 2)  # (T, C, H, W)
    frames_tensor = frames_tensor.unsqueeze(0).to(device)  # (1, T, C, H, W)

    # Reconstruct
    print("Reconstructing...")
    with torch.no_grad():
        reconstructed = model(frames_tensor)

    # Move back to CPU and denormalize
    original = frames_tensor.cpu().squeeze(0).permute(0, 2, 3, 1).numpy()  # (T, H, W, C)
    reconstructed = reconstructed.cpu().squeeze(0).permute(0, 2, 3, 1).numpy()  # (T, H, W, C)

    # Clip reconstructed to [0, 1]
    reconstructed = np.clip(reconstructed, 0, 1)

    # Visualize
    print("\nDisplaying results...")
    fig, axes = plt.subplots(2, sequence_length, figsize=(20, 5))
    fig.suptitle('Original (top) vs Reconstructed (bottom)', fontsize=16)

    for t in range(sequence_length):
        # Original
        axes[0, t].imshow(original[t])
        axes[0, t].axis('off')
        axes[0, t].set_title(f'Frame {t}')

        # Reconstructed
        axes[1, t].imshow(reconstructed[t])
        axes[1, t].axis('off')

    plt.tight_layout()
    plt.savefig('reconstruction_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to reconstruction_comparison.png")
    plt.show()

    # Calculate MSE
    mse = np.mean((original - reconstructed) ** 2)
    print(f"\nMSE: {mse:.6f}")

if __name__ == '__main__':
    visualize_reconstruction()
