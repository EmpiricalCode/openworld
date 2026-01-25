import h5py
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def visualize_data(data_file, fps=30, start_step=0):
    """
    Visualize collected ViZDoom HealthGathering data using matplotlib.

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

    # Action names for ViZDoom HealthGathering
    action_names = {0: "NOOP", 1: "FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}

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
    # Default to ViZDoom data files
    default_files = [
        Path("data/vizdoom_healthgathering/vizdoom_healthgathering_dqn.h5"),
        Path("data/vizdoom_healthgathering/vizdoom_healthgathering_random.h5"),
    ]

    if len(sys.argv) > 1:
        data_file = Path(sys.argv[1])
    else:
        # Try to find an existing file
        data_file = None
        for f in default_files:
            if f.exists():
                data_file = f
                break

        if data_file is None:
            print(f"Error: No data files found. Checked:")
            for f in default_files:
                print(f"  - {f}")
            sys.exit(1)

    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        sys.exit(1)

    print(f"Using data file: {data_file}\n")
    visualize_data(data_file)
