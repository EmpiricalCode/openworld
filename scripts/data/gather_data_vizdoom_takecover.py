"""
Gather data from ViZDoom TakeCover with biased random exploration.

TakeCover: Agent must dodge fireballs by moving left/right.
3 actions: 0=NOOP, 1=MOVE_LEFT, 2=MOVE_RIGHT

Each episode picks a random movement bias (left-heavy, right-heavy, or uniform)
to ensure diverse trajectories without needing a trained agent.
"""
import gymnasium as gym
import vizdoom.gymnasium_wrapper
import cv2
import numpy as np
import h5py
from pathlib import Path

SAVE_DIR = Path("data/vizdoom_takecover")


def collect_data(num_episodes=500, output_file="vizdoom_takecover.h5"):
    env = gym.make("VizdoomTakeCover-v1", render_mode="rgb_array", frame_skip=8)

    all_frames = []
    all_actions = []
    all_rewards = []
    all_dones = []

    total_steps = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        bias = np.random.choice(['left', 'right', 'noop', 'random'])
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            if bias == 'left':
                action = np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])
            elif bias == 'right':
                action = np.random.choice([0, 1, 2], p=[0.2, 0.2, 0.6])
            elif bias == 'noop':
                action = np.random.choice([0, 1, 2], p=[0.6, 0.2, 0.2])
            else:
                action = np.random.randint(3)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            frame_rgb = env.render()
            frame_resized = cv2.resize(frame_rgb, (64, 64), interpolation=cv2.INTER_AREA)

            all_frames.append(frame_resized)
            all_actions.append(action)
            all_rewards.append(reward)
            all_dones.append(done)

        print(f"Episode {ep + 1}/{num_episodes} | Bias: {bias:>6} | "
              f"Reward: {episode_reward:.1f} | Steps: {episode_steps} | "
              f"Total: {total_steps}")

    env.close()

    # Save
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAVE_DIR / output_file

    with h5py.File(out_path, "w") as f:
        f.create_dataset("frames", data=np.array(all_frames), compression="lzf")
        f.create_dataset("actions", data=np.array(all_actions), compression="lzf")
        f.create_dataset("rewards", data=np.array(all_rewards), compression="lzf")
        f.create_dataset("dones", data=np.array(all_dones), compression="lzf")

    print(f"\nData saved to {out_path}")
    print(f"Frames: {np.array(all_frames).shape}")
    print(f"Total steps: {total_steps}")
    print(f"File size: {out_path.stat().st_size / (1024**2):.2f} MB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--output', type=str, default='vizdoom_takecover.h5')
    args = parser.parse_args()

    collect_data(num_episodes=args.episodes, output_file=args.output)
