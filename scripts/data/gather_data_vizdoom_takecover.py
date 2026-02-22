"""
Gather data from ViZDoom TakeCover with PPO (stable-baselines3).

TakeCover: Agent must dodge fireballs by moving left/right.
3 actions: NOOP, MOVE_LEFT, MOVE_RIGHT
"""
import gymnasium as gym
import vizdoom.gymnasium_wrapper
import pygame
import cv2
import numpy as np
import h5py
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


class EpisodeLogCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_count = 0

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
                r = info["episode"]["r"]
                l = info["episode"]["l"]
                print(f"  Episode {self.episode_count} | Reward: {r:.1f} | Length: {l}")
        return True


LOG_DIR = "data/vizdoom_takecover/tb_logs"
SAVE_DIR = Path("data/vizdoom_takecover")


def make_env():
    return Monitor(gym.make("VizdoomTakeCover-v1", frame_skip=2))


def train_ppo(total_timesteps=250000):
    """Train PPO agent on TakeCover"""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env])

    model = PPO(
        "MultiInputPolicy",
        env,
        tensorboard_log=None,
        verbose=1,
        learning_rate=1e-5,
        n_steps=8192,
        clip_range=0.1,
        gamma=0.95,
        gae_lambda=0.9,
    )

    model.learn(total_timesteps=total_timesteps, callback=EpisodeLogCallback())
    model.save(SAVE_DIR / "ppo_takecover")
    env.close()
    print(f"Model saved to {SAVE_DIR / 'ppo_takecover'}")
    return model


def collect_data(model=None, num_steps=10000, visualize=True):
    """Collect data with trained agent or random policy"""
    env = gym.make("VizdoomTakeCover-v1", render_mode="rgb_array")

    if model:
        print(f"Collecting {num_steps} steps with TRAINED PPO agent")
    else:
        print(f"Collecting {num_steps} steps with RANDOM agent")

    if visualize:
        pygame.init()
        display_size = 480
        screen = pygame.display.set_mode((display_size, display_size))
        pygame.display.set_caption("ViZDoom TakeCover - Data Collection")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 36)

    all_frames = []
    all_actions = []
    all_rewards = []
    all_dones = []

    num_steps_collected = 0
    episode_num = 0
    episode_reward = 0

    obs, _ = env.reset()

    while num_steps_collected < num_steps:
        if visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    env.close()
                    return

        if model:
            action, _ = model.predict(obs, deterministic=False)
            action = int(action)
        else:
            action = env.action_space.sample()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        frame_rgb = env.render()
        frame_resized = cv2.resize(frame_rgb, (64, 64), interpolation=cv2.INTER_AREA)

        all_frames.append(frame_resized)
        all_actions.append(action)
        all_rewards.append(reward)
        all_dones.append(done)

        if visualize:
            frame_display = cv2.resize(frame_resized, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
            frame_transposed = np.transpose(frame_display, (1, 0, 2))
            surface = pygame.surfarray.make_surface(frame_transposed)
            screen.blit(surface, (0, 0))

            agent_type = "PPO" if model else "Random"
            info_text = font.render(f'{agent_type} - Steps: {num_steps_collected}/{num_steps} | Reward: {episode_reward:.1f}',
                                   True, (255, 255, 0))
            text_bg = pygame.Surface((info_text.get_width() + 10, info_text.get_height() + 10))
            text_bg.fill((0, 0, 0))
            text_bg.set_alpha(180)
            screen.blit(text_bg, (5, 5))
            screen.blit(info_text, (10, 10))

            pygame.display.flip()
            clock.tick(35)

        obs = next_obs
        num_steps_collected += 1

        if done:
            episode_num += 1
            print(f"Episode {episode_num} - Reward: {episode_reward:.2f} - Steps: {num_steps_collected}")
            episode_reward = 0
            obs, _ = env.reset()

        if num_steps_collected % 1000 == 0:
            print(f"Progress: {num_steps_collected}/{num_steps} steps")

    if visualize:
        pygame.quit()
    env.close()

    # Save data
    print("\nSaving data...")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    filename = "vizdoom_takecover_ppo.h5" if model else "vizdoom_takecover_random.h5"
    output_file = SAVE_DIR / filename

    with h5py.File(output_file, "w") as f:
        f.create_dataset("frames", data=np.array(all_frames), compression="lzf")
        f.create_dataset("actions", data=np.array(all_actions), compression="lzf")
        f.create_dataset("rewards", data=np.array(all_rewards), compression="lzf")
        f.create_dataset("dones", data=np.array(all_dones), compression="lzf")

    print(f"Data saved to {output_file}")
    print(f"Frames shape: {np.array(all_frames).shape}")
    print(f"File size: {output_file.stat().st_size / (1024**2):.2f} MB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train PPO before collecting data')
    parser.add_argument('--load-agent', type=str, default=None, help='Path to saved PPO model (without .zip)')
    parser.add_argument('--num-steps', type=int, default=10000, help='Number of steps to collect')
    parser.add_argument('--train-steps', type=int, default=10000, help='PPO training timesteps')
    args = parser.parse_args()

    if args.load_agent:
        print(f"Loading PPO from {args.load_agent}")
        model = PPO.load(args.load_agent)
        collect_data(model, num_steps=args.num_steps, visualize=True)
    elif args.train:
        model = train_ppo(total_timesteps=args.train_steps)
        collect_data(model, num_steps=args.num_steps, visualize=True)
    else:
        collect_data(None, num_steps=args.num_steps, visualize=True)
