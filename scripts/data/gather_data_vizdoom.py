"""
Gather data from ViZDoom HealthGathering with optional DQN training.
"""
import gymnasium as gym
import vizdoom.gymnasium_wrapper
import pygame
import cv2
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from pathlib import Path
from collections import deque


class DQN(nn.Module):
    """Deep Q-Network with health input"""
    def __init__(self, input_shape, n_actions, use_health=True):
        super().__init__()
        self.use_health = use_health

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.conv(dummy).shape[1]

        # Add health dimension if using it
        fc_input_size = conv_out + 1 if use_health else conv_out

        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x, health=None):
        conv_out = self.conv(x)
        if self.use_health and health is not None:
            # Concatenate health to conv features
            combined = torch.cat([conv_out, health], dim=1)
            return self.fc(combined)
        return self.fc(conv_out)


class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, health=None, next_health=None):
        self.buffer.append((state, action, reward, next_state, done, health, next_health))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, healths, next_healths = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones),
                np.array(healths) if healths[0] is not None else None,
                np.array(next_healths) if next_healths[0] is not None else None)

    def __len__(self):
        return len(self.buffer)


def preprocess_observation(obs):
    """Preprocess observation - extract frame and health"""
    if isinstance(obs, dict):
        frame = obs['screen']
        health = obs['gamevariables'][0] if 'gamevariables' in obs else 100.0
    else:
        frame = obs
        health = 100.0

    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    elif len(frame.shape) == 3:
        frame = frame.squeeze(-1)

    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0

    # Normalize health to 0-1 range (assuming max health ~100)
    health_normalized = health / 100.0

    return frame, health_normalized


def train_dqn(max_steps=250000, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Train DQN agent on HealthGathering (based on Aitchison 2019 paper)"""
    print(f"Training DQN on device: {device}")
    print(f"Training for {max_steps} environment steps")

    # No rendering during training for better performance
    env = gym.make("VizdoomHealthGathering-v1")
    n_actions = env.action_space.n
    frame_stack_size = 4

    # Networks
    policy_net = DQN((frame_stack_size, 84, 84), n_actions).to(device)
    target_net = DQN((frame_stack_size, 84, 84), n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Hyperparameters from Aitchison 2019 paper
    optimizer = optim.RMSprop(policy_net.parameters(), lr=5e-5)
    replay_buffer = ReplayBuffer(10000)

    batch_size = 32
    gamma = 1.0  # No discounting (as per paper)
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = max_steps * 0.2  # Decay over first 20% of training
    target_update = 1000
    train_start = 1000
    update_frequency = 1  # Update every step (can change to 4 for speed)

    total_steps = 0
    episode_num = 0
    episode_rewards = []
    best_avg_reward = -float('inf')

    print("Starting training...")

    while total_steps < max_steps:
        episode_start_time = time.time()
        obs, _ = env.reset()
        frame, health = preprocess_observation(obs)
        frame_stack = deque([frame] * frame_stack_size, maxlen=frame_stack_size)
        state = np.array(frame_stack)

        episode_reward = 0
        episode_steps = 0
        done = False

        # Track alternating turn actions (to penalize 2-3-2-3 oscillation)
        prev_action = None
        prev_prev_action = None
        alternation_count = 0

        while not done:
            # Epsilon-greedy
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-total_steps / epsilon_decay)

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    health_tensor = torch.FloatTensor([[health]]).to(device)
                    q_values = policy_net(state_tensor, health_tensor)
                    action = q_values.argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_frame, next_health = preprocess_observation(next_obs)

            # Reward shaping: +100 for health gain, -1 for existing, +0.5 for moving forward
            health_delta = next_health - health
            if health_delta > 0:
                # Gained health (collected health pack)
                reward = 100.0
            else:
                # Alive penalty to encourage collecting health packs
                reward = -1.0

            # Bonus for moving forward (action 1 is typically forward)
            if action == 1:
                reward += 0.25

            # Penalty for alternating between left (2) and right (3) repeatedly
            # Detect pattern: 2-3-2 or 3-2-3 and onwards
            if action in [2, 3]:  # Current action is a turn
                # Check if we're alternating between 2 and 3
                if prev_action is not None and prev_prev_action is not None:
                    # Detect alternation: current matches prev_prev, and prev is opposite
                    opposite_turn = 3 if action == 2 else 2
                    if prev_action == opposite_turn and prev_prev_action == action:
                        # We're in an alternation pattern!
                        reward -= 5.0
                        alternation_count += 1
                    else:
                        # Pattern broken, reset counter
                        alternation_count = 0

                # Update action history
                prev_prev_action = prev_action
                prev_action = action
            else:
                # Non-turn action breaks the pattern
                prev_prev_action = None
                prev_action = None
                alternation_count = 0

            frame_stack.append(next_frame)
            next_state = np.array(frame_stack)

            replay_buffer.push(state, action, reward, next_state, done, health, next_health)
            health = next_health

            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            # Training
            if len(replay_buffer) >= train_start and len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones, healths, next_healths = replay_buffer.sample(batch_size)

                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)
                healths = torch.FloatTensor(healths).unsqueeze(1).to(device)
                next_healths = torch.FloatTensor(next_healths).unsqueeze(1).to(device)

                q_values = policy_net(states, healths).gather(1, actions.unsqueeze(1))

                with torch.no_grad():
                    next_q_values = target_net(next_states, next_healths).max(1)[0]
                    target_q_values = rewards + (1 - dones) * gamma * next_q_values

                loss = nn.MSELoss()(q_values.squeeze(), target_q_values)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
                optimizer.step()

            if total_steps % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Break if we've reached max steps
            if total_steps >= max_steps:
                done = True

        episode_num += 1
        episode_rewards.append(episode_reward)
        episode_time = time.time() - episode_start_time

        # Print every episode
        print(f"Episode {episode_num} | Reward: {episode_reward:.2f} | "
              f"Steps: {episode_steps} | Time: {episode_time:.2f}s | "
              f"Total Steps: {total_steps}/{max_steps} | Epsilon: {epsilon:.3f}")

        if episode_num % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"  --> Avg Reward (last 10): {avg_reward:.2f}")

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                # Save model in data directory
                save_dir = Path("data/vizdoom_healthgathering")
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / 'dqn_healthgathering.pth'
                torch.save(policy_net.state_dict(), save_path)
                print(f"  --> New best model saved to {save_path}! Avg reward: {avg_reward:.2f}")

    env.close()
    print(f"Training complete! Best avg reward: {best_avg_reward:.2f}")
    return policy_net


def collect_data_with_agent(agent=None, num_steps=10000, visualize=True):
    """Collect data with trained agent or random policy"""
    env = gym.make("VizdoomHealthGathering-v1", render_mode="rgb_array")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if agent:
        agent.eval()
        frame_stack_size = 4
        print(f"Collecting {num_steps} steps with TRAINED agent")
    else:
        print(f"Collecting {num_steps} steps with RANDOM agent")

    if visualize:
        pygame.init()
        display_size = 480
        screen = pygame.display.set_mode((display_size, display_size))
        pygame.display.set_caption("ViZDoom HealthGathering - Data Collection")
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
    health = None

    if agent:
        frame, health = preprocess_observation(obs)
        frame_stack = deque([frame] * frame_stack_size, maxlen=frame_stack_size)

    while num_steps_collected < num_steps:
        if visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    env.close()
                    return

        if agent:
            state = np.array(frame_stack)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                health_tensor = torch.FloatTensor([[health]]).to(device)
                q_values = agent(state_tensor, health_tensor)
                action = q_values.argmax().item()
        else:
            action = env.action_space.sample()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        # Get and store frame
        frame_rgb = env.render()
        frame_resized = cv2.resize(frame_rgb, (128, 128), interpolation=cv2.INTER_AREA)

        all_frames.append(frame_resized)
        all_actions.append(action)
        all_rewards.append(reward)
        all_dones.append(done)

        if visualize:
            frame_display = cv2.resize(frame_resized, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
            frame_transposed = np.transpose(frame_display, (1, 0, 2))
            surface = pygame.surfarray.make_surface(frame_transposed)
            screen.blit(surface, (0, 0))

            agent_type = "Trained DQN" if agent else "Random"
            info_text = font.render(f'{agent_type} - Steps: {num_steps_collected}/{num_steps} | Reward: {episode_reward:.1f}',
                                   True, (255, 255, 0))
            text_bg = pygame.Surface((info_text.get_width() + 10, info_text.get_height() + 10))
            text_bg.fill((0, 0, 0))
            text_bg.set_alpha(180)
            screen.blit(text_bg, (5, 5))
            screen.blit(info_text, (10, 10))

            pygame.display.flip()
            clock.tick(35)

        if agent:
            next_frame, health = preprocess_observation(next_obs)
            frame_stack.append(next_frame)

        num_steps_collected += 1

        if done:
            episode_num += 1
            print(f"Episode {episode_num} - Reward: {episode_reward:.2f} - Steps: {num_steps_collected}")
            episode_reward = 0
            obs, _ = env.reset()

            if agent:
                frame, health = preprocess_observation(obs)
                frame_stack = deque([frame] * frame_stack_size, maxlen=frame_stack_size)

        if num_steps_collected % 1000 == 0:
            print(f"Progress: {num_steps_collected}/{num_steps} steps")

    if visualize:
        pygame.quit()
    env.close()

    # Save data
    print("\nSaving data...")
    data_dir = Path("data/vizdoom_healthgathering")
    data_dir.mkdir(parents=True, exist_ok=True)

    filename = "vizdoom_healthgathering_dqn.h5" if agent else "vizdoom_healthgathering_random.h5"
    output_file = data_dir / filename

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
    parser.add_argument('--train', action='store_true', help='Train DQN before collecting data')
    parser.add_argument('--load-agent', type=str, default=None, help='Path to saved agent model to use for data collection')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if args.load_agent:
        # Load saved agent and collect data
        print(f"Loading agent from {args.load_agent}")
        env = gym.make("VizdoomHealthGathering-v1")
        n_actions = env.action_space.n
        agent = DQN((4, 84, 84), n_actions).to(device)
        agent.load_state_dict(torch.load(args.load_agent, map_location=device))
        agent.eval()
        env.close()
        collect_data_with_agent(agent, num_steps=100000, visualize=True)
    elif args.train:
        # Train and use trained agent (250k steps as per Aitchison 2019)
        agent = train_dqn(max_steps=250000, device=device)
        collect_data_with_agent(agent, num_steps=100000, visualize=True)
    else:
        # Use random agent
        collect_data_with_agent(None, num_steps=100000, visualize=True)
