"""
Manual control test for ViZDoom HealthGathering.
Use keyboard to play and see rewards in real-time.

Controls:
- W: Move forward
- A: Turn left
- D: Turn right
- ESC: Quit
"""
import gymnasium as gym
import vizdoom.gymnasium_wrapper
import pygame
import cv2
import numpy as np

def main():
    # Create environment
    env = gym.make("VizdoomHealthGathering-v1", render_mode="rgb_array")

    # Initialize pygame
    pygame.init()
    display_size = 640
    screen = pygame.display.set_mode((display_size, display_size))
    pygame.display.set_caption("ViZDoom HealthGathering - Manual Control")
    clock = pygame.time.Clock()
    font_large = pygame.font.Font(None, 48)
    font_small = pygame.font.Font(None, 32)

    # Action mapping
    # action_space usually: [NOOP, FORWARD, TURN_LEFT, TURN_RIGHT]
    ACTION_NOOP = 0
    ACTION_FORWARD = 1
    ACTION_TURN_LEFT = 2
    ACTION_TURN_RIGHT = 3

    obs, _ = env.reset()
    health = obs['gamevariables'][0]

    episode_reward = 0
    episode_shaped_reward = 0
    total_steps = 0
    health_packs_collected = 0

    running = True
    while running:
        action = ACTION_NOOP

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Get keyboard state
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = ACTION_FORWARD
        elif keys[pygame.K_a]:
            action = ACTION_TURN_LEFT
        elif keys[pygame.K_d]:
            action = ACTION_TURN_RIGHT

        # Step environment
        old_health = health
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        health = obs['gamevariables'][0]

        # Calculate shaped reward (same as training in gather_data_vizdoom.py)
        health_delta = health - old_health
        if health_delta > 0:
            shaped_reward = 100.0
            health_packs_collected += 1
        else:
            shaped_reward = -1.0

        # Bonus for moving forward (action 1)
        if action == ACTION_FORWARD:
            shaped_reward += 0.25

        episode_reward += reward
        episode_shaped_reward += shaped_reward
        total_steps += 1

        # Render
        frame = env.render()
        frame_display = cv2.resize(frame, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
        frame_transposed = np.transpose(frame_display, (1, 0, 2))
        surface = pygame.surfarray.make_surface(frame_transposed)
        screen.blit(surface, (0, 0))

        # Display info overlay
        y_offset = 10

        # Health bar
        health_text = font_large.render(f'Health: {health:.0f}', True, (0, 255, 0) if health > 50 else (255, 0, 0))
        text_bg = pygame.Surface((health_text.get_width() + 20, health_text.get_height() + 10))
        text_bg.fill((0, 0, 0))
        text_bg.set_alpha(200)
        screen.blit(text_bg, (10, y_offset))
        screen.blit(health_text, (20, y_offset + 5))
        y_offset += health_text.get_height() + 20

        # Stats
        stats = [
            f'Steps: {total_steps}',
            f'Health Packs: {health_packs_collected}',
            f'Step Reward: {shaped_reward:+.1f}',  # Current step reward
            f'Total Shaped: {episode_shaped_reward:.1f}',
            f'Total Env: {episode_reward:.1f}',
            f'Action: {["NOOP", "FORWARD", "LEFT", "RIGHT"][action]}'
        ]

        for stat in stats:
            stat_text = font_small.render(stat, True, (255, 255, 255))
            text_bg = pygame.Surface((stat_text.get_width() + 20, stat_text.get_height() + 8))
            text_bg.fill((0, 0, 0))
            text_bg.set_alpha(180)
            screen.blit(text_bg, (10, y_offset))
            screen.blit(stat_text, (20, y_offset + 4))
            y_offset += stat_text.get_height() + 12

        # Controls help
        y_offset += 10
        controls = ['W: Forward', 'A: Turn Left', 'D: Turn Right', 'ESC: Quit']
        for control in controls:
            control_text = font_small.render(control, True, (200, 200, 200))
            text_bg = pygame.Surface((control_text.get_width() + 20, control_text.get_height() + 8))
            text_bg.fill((0, 0, 0))
            text_bg.set_alpha(150)
            screen.blit(text_bg, (10, y_offset))
            screen.blit(control_text, (20, y_offset + 4))
            y_offset += control_text.get_height() + 8

        pygame.display.flip()
        clock.tick(35)  # 35 FPS

        # Reset if done
        if done:
            print(f"\n=== Episode Ended ===")
            print(f"Steps: {total_steps}")
            print(f"Health Packs Collected: {health_packs_collected}")
            print(f"Environment Reward: {episode_reward:.1f}")
            print(f"Shaped Reward: {episode_shaped_reward:.1f}")
            print(f"Final Health: {health:.1f}")
            print("Press any key to restart or ESC to quit...")

            # Wait for keypress
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            waiting = False
                        else:
                            waiting = False

            if running:
                obs, _ = env.reset()
                health = obs['gamevariables'][0]
                episode_reward = 0
                episode_shaped_reward = 0
                total_steps = 0
                health_packs_collected = 0

    pygame.quit()
    env.close()
    print("Thanks for playing!")

if __name__ == "__main__":
    main()
