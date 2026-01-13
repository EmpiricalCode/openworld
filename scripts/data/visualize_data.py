import h5py
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def visualize_data(data_file, fps=30, start_step=0):
    """
    Visualize collected Lunar Lander data using matplotlib.

    Args:
        data_file: Path to HDF5 file
        fps: Playback frames per second
        start_step: Step to start from
    """
    # Load data
    print(f"Loading data from {data_file}...")
    with h5py.File(data_file, "r") as f:
        frames = f["frames"][:]
        actions = f["actions"][:]
        rewards = f["rewards"][:]
        dones = f["dones"][:]

    print(f"Loaded {len(frames)} frames")
    print(f"Frame shape: {frames[0].shape}")

    # Action names
    action_names = {0: "NOOP", 1: "LEFT", 2: "MAIN", 3: "RIGHT"}

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')

    # Initial frame
    im = ax.imshow(frames[start_step])

    # Text for info
    info_text = ax.text(5, 15, '', fontsize=8, color='white',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    step_counter = [start_step]  # Use list to modify in nested function

    def update(frame_idx):
        step = step_counter[0]

        # Update image
        im.set_array(frames[step])

        # Update info text
        if step < len(actions):
            action = actions[step]
            # Check if action is discrete (int) or continuous (array)
            if isinstance(action, (int, np.integer)):
                action_str = action_names[action]
            else:
                # Continuous action (e.g., Car Racing: [steering, gas, brake])
                action_str = f"[{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]"
            info = f"Step: {step}/{len(frames)-1}\nAction: {action_str}\nReward: {rewards[step]:.2f}"
        else:
            info = f"Step: {step}/{len(frames)-1} (final)"

        info_text.set_text(info)

        # Advance to next frame
        step_counter[0] = (step + 1) % len(frames)

        return [im, info_text]

    print("\nStarting animation...")
    print("Close the window to stop.")

    # Create animation
    interval = 1000 / fps  # milliseconds per frame
    anim = FuncAnimation(fig, update, frames=len(frames),
                        interval=interval, blit=True, repeat=True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Default data file
    default_file = Path("data/vizdoom_healthgathering/vizdoom_healthgathering_10k_steps.h5")

    if len(sys.argv) > 1:
        data_file = Path(sys.argv[1])
    else:
        data_file = default_file

    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        sys.exit(1)

    visualize_data(data_file)
