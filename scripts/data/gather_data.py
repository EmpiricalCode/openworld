import gymnasium as gym
import pygame
import cv2
import numpy as np
import h5py
from pathlib import Path

# Create data directory
data_dir = Path("data/lunar_lander")
data_dir.mkdir(parents=True, exist_ok=True)

env = gym.make("LunarLander-v3", render_mode="rgb_array")
observation, info = env.reset()

# Initialize pygame for display
pygame.init()
screen = pygame.display.set_mode((128, 128))
pygame.display.set_caption("Lunar Lander 128x128")

# Storage for frames and actions
frames = []
actions = []
rewards = []
dones = []

num_steps = 10000
running = True

for step in range(num_steps):
    if not running:
        break

    # Check for quit event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Get frame and resize to 128x128
    frame = env.render()
    frame_resized = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)

    # Store data
    frames.append(frame_resized)
    actions.append(action)
    rewards.append(reward)
    dones.append(done)

    # Convert to pygame surface and display
    frame_transposed = np.transpose(frame_resized, (1, 0, 2))
    surface = pygame.surfarray.make_surface(frame_transposed)
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    if done:
        observation, info = env.reset()

    # Print progress
    if (step + 1) % 1000 == 0:
        print(f"Collected {step + 1}/{num_steps} steps")

pygame.quit()
env.close()

# Save data to HDF5 with LZF compression
print("Saving data...")
output_file = data_dir / "lunar_lander_10k_steps.h5"

with h5py.File(output_file, "w") as f:
    f.create_dataset("frames", data=np.array(frames), compression="lzf")
    f.create_dataset("actions", data=np.array(actions), compression="lzf")
    f.create_dataset("rewards", data=np.array(rewards), compression="lzf")
    f.create_dataset("dones", data=np.array(dones), compression="lzf")

print(f"Data saved to {output_file}")
print(f"Frames shape: {np.array(frames).shape}")
print(f"File size: {output_file.stat().st_size / (1024**2):.2f} MB")
