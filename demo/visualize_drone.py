"""
Visual demonstration of the drone environment using Pygame.

This script loads a trained model and visualizes the drone navigating
through the environment with wind forces.
"""

import os
import sys
import pygame
import numpy as np
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.drone_env import DroneWindEnv
from stable_baselines3 import PPO


# Pygame constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 30

# Color definitions
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
ORANGE = (255, 165, 0)


class DroneVisualizer:
    """Pygame-based visualizer for the drone environment."""
    
    def __init__(self, env: DroneWindEnv, model: Optional[PPO] = None):
        """
        Initialize the visualizer.
        
        Args:
            env: DroneWindEnv instance
            model: Optional trained PPO model (if None, uses random actions)
        """
        self.env = env
        self.model = model
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Drone RL - Visual Demonstration")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # World to screen scaling
        # Environment is [0, 1] x [0, 1], we'll use most of the screen
        self.world_margin = 50
        self.world_width = WINDOW_WIDTH - 2 * self.world_margin
        self.world_height = WINDOW_HEIGHT - 2 * self.world_margin
        
    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates [0,1] to screen coordinates."""
        screen_x = int(self.world_margin + x * self.world_width)
        # Flip y-axis (world y=0 is bottom, screen y=0 is top)
        screen_y = int(WINDOW_HEIGHT - self.world_margin - y * self.world_height)
        return screen_x, screen_y
    
    def draw_drone(self, x: float, y: float, vx: float, vy: float):
        """Draw the drone as a circle with velocity vector."""
        screen_x, screen_y = self.world_to_screen(x, y)
        
        # Draw drone body (circle)
        drone_radius = 15
        pygame.draw.circle(self.screen, CYAN, (screen_x, screen_y), drone_radius)
        pygame.draw.circle(self.screen, BLUE, (screen_x, screen_y), drone_radius, 2)
        
        # Draw velocity vector
        if abs(vx) > 0.01 or abs(vy) > 0.01:
            # Scale velocity for visualization
            scale = 30
            end_x = screen_x + int(vx * scale)
            end_y = screen_y - int(vy * scale)  # Flip y for screen
            pygame.draw.line(self.screen, YELLOW, (screen_x, screen_y), (end_x, end_y), 3)
            # Draw arrowhead
            if abs(vx) > 0.01 or abs(vy) > 0.01:
                angle = np.arctan2(-vy, vx)  # Negative vy because screen y is flipped
                arrow_size = 8
                arrow_x1 = end_x - arrow_size * np.cos(angle - np.pi / 6)
                arrow_y1 = end_y - arrow_size * np.sin(angle - np.pi / 6)
                arrow_x2 = end_x - arrow_size * np.cos(angle + np.pi / 6)
                arrow_y2 = end_y - arrow_size * np.sin(angle + np.pi / 6)
                pygame.draw.line(self.screen, YELLOW, (end_x, end_y), (int(arrow_x1), int(arrow_y1)), 2)
                pygame.draw.line(self.screen, YELLOW, (end_x, end_y), (int(arrow_x2), int(arrow_y2)), 2)
    
    def draw_wind(self, wind_x: float, wind_y: float):
        """Draw wind arrows indicating direction."""
        # Draw fewer, clearer wind arrows
        grid_size = 6
        for i in range(grid_size):
            for j in range(grid_size):
                x = (i + 0.5) / grid_size
                y = (j + 0.5) / grid_size
                screen_x, screen_y = self.world_to_screen(x, y)
                
                # Draw wind arrow
                if abs(wind_x) > 0.01 or abs(wind_y) > 0.01:
                    scale = 25
                    end_x = screen_x + int(wind_x * scale)
                    end_y = screen_y - int(wind_y * scale)  # Flip y
                    
                    # Color based on wind strength
                    wind_strength = abs(wind_x) + abs(wind_y)
                    if wind_strength < 1.0:
                        color = GREEN
                    elif wind_strength < 1.5:
                        color = YELLOW
                    else:
                        color = ORANGE
                    
                    # Draw arrow line
                    pygame.draw.line(self.screen, color, (screen_x, screen_y), (end_x, end_y), 3)
                    
                    # Draw arrowhead
                    if abs(wind_x) > 0.01 or abs(wind_y) > 0.01:
                        angle = np.arctan2(-wind_y, wind_x)  # Negative y because screen y is flipped
                        arrow_size = 10
                        arrow_x1 = end_x - arrow_size * np.cos(angle - np.pi / 6)
                        arrow_y1 = end_y - arrow_size * np.sin(angle - np.pi / 6)
                        arrow_x2 = end_x - arrow_size * np.cos(angle + np.pi / 6)
                        arrow_y2 = end_y - arrow_size * np.sin(angle + np.pi / 6)
                        pygame.draw.polygon(self.screen, color, [
                            (end_x, end_y),
                            (int(arrow_x1), int(arrow_y1)),
                            (int(arrow_x2), int(arrow_y2))
                        ])
    
    def draw_boundaries(self):
        """Draw the world boundaries."""
        # Top boundary
        top_left = self.world_to_screen(0, 1)
        top_right = self.world_to_screen(1, 1)
        pygame.draw.line(self.screen, RED, top_left, top_right, 3)
        
        # Bottom boundary
        bot_left = self.world_to_screen(0, 0)
        bot_right = self.world_to_screen(1, 0)
        pygame.draw.line(self.screen, RED, bot_left, bot_right, 3)
        
        # Left boundary
        pygame.draw.line(self.screen, RED, top_left, bot_left, 3)
        
        # Right boundary
        pygame.draw.line(self.screen, RED, top_right, bot_right, 3)
    
    def draw_target_zone(self, target_spawned: bool = True):
        """Draw the target zone (box) that the drone needs to reach."""
        from env.drone_env import TARGET_X_MIN, TARGET_X_MAX, TARGET_Y_MIN, TARGET_Y_MAX, TARGET_SPAWN_DELAY
        
        # Only draw if target has spawned
        if not target_spawned:
            return
        
        # Get screen coordinates for target zone corners
        top_left = self.world_to_screen(TARGET_X_MIN, TARGET_Y_MAX)
        top_right = self.world_to_screen(TARGET_X_MAX, TARGET_Y_MAX)
        bot_left = self.world_to_screen(TARGET_X_MIN, TARGET_Y_MIN)
        bot_right = self.world_to_screen(TARGET_X_MAX, TARGET_Y_MIN)
        
        # Draw target zone as a semi-transparent box
        # Create a surface for transparency
        target_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        target_surface.set_alpha(100)  # Semi-transparent
        
        # Draw filled rectangle
        rect = pygame.Rect(
            top_left[0], top_left[1],
            top_right[0] - top_left[0],
            bot_left[1] - top_left[1]
        )
        pygame.draw.rect(target_surface, MAGENTA, rect)
        self.screen.blit(target_surface, (0, 0))
        
        # Draw border
        pygame.draw.line(self.screen, MAGENTA, top_left, top_right, 3)
        pygame.draw.line(self.screen, MAGENTA, top_right, bot_right, 3)
        pygame.draw.line(self.screen, MAGENTA, bot_right, bot_left, 3)
        pygame.draw.line(self.screen, MAGENTA, bot_left, top_left, 3)
        
        # Draw label
        label_x = (top_left[0] + top_right[0]) // 2
        label_y = (top_left[1] + bot_left[1]) // 2
        text = self.small_font.render("TARGET", True, WHITE)
        text_rect = text.get_rect(center=(label_x, label_y))
        self.screen.blit(text, text_rect)
    
    def draw_info(self, step: int, reward: float, action: Optional[int] = None, in_target: bool = False):
        """Draw information text."""
        y_offset = 10
        
        # Step count
        text = self.font.render(f"Step: {step}", True, WHITE)
        self.screen.blit(text, (10, y_offset))
        y_offset += 30
        
        # Reward
        text = self.font.render(f"Reward: {reward:.2f}", True, WHITE)
        self.screen.blit(text, (10, y_offset))
        y_offset += 30
        
        # In target zone status
        target_color = GREEN if in_target else GRAY
        target_text = "IN TARGET ZONE!" if in_target else "Not in target"
        text = self.font.render(target_text, True, target_color)
        self.screen.blit(text, (10, y_offset))
        y_offset += 30
        
        # Position
        text = self.small_font.render(f"Position: ({self.env.x:.2f}, {self.env.y:.2f})", True, WHITE)
        self.screen.blit(text, (10, y_offset))
        y_offset += 25
        
        # Velocity
        text = self.small_font.render(f"Velocity: ({self.env.vx:.2f}, {self.env.vy:.2f})", True, WHITE)
        self.screen.blit(text, (10, y_offset))
        y_offset += 25
        
        # Wind
        text = self.small_font.render(f"Wind: ({self.env.wind_x:.2f}, {self.env.wind_y:.2f})", True, GREEN)
        self.screen.blit(text, (10, y_offset))
        y_offset += 25
        
        # Action
        if action is not None:
            action_names = ["No thrust", "Up", "Down", "Left", "Right"]
            text = self.small_font.render(f"Action: {action_names[action]}", True, YELLOW)
            self.screen.blit(text, (10, y_offset))
            y_offset += 25
        
        # Model info
        if self.model is not None:
            text = self.small_font.render("Mode: AI Agent (Liquid NN)", True, CYAN)
        else:
            text = self.small_font.render("Mode: Random Actions", True, GRAY)
        self.screen.blit(text, (10, y_offset))
    
    def run(self, max_steps: int = 500, speed: float = 1.0):
        """
        Run the visualization.
        
        Args:
            max_steps: Maximum number of steps to run
            speed: Speed multiplier (1.0 = normal, higher = faster)
        """
        obs, info = self.env.reset()
        done = False
        truncated = False
        step_count = 0
        action = None
        
        running = True
        paused = False
        
        while running and step_count < max_steps:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        # Reset
                        obs, info = self.env.reset()
                        done = False
                        truncated = False
                        step_count = 0
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            if not paused and not done and not truncated:
                # Get action
                if self.model is not None:
                    action, _ = self.model.predict(obs, deterministic=True)
                else:
                    action = self.env.action_space.sample()
                
                # Step environment
                obs, reward, done, truncated, info = self.env.step(action)
                step_count += 1
                in_target = info.get("in_target", False)
                target_spawned = info.get("target_spawned", False)
            
            # Draw everything
            self.screen.fill(BLACK)
            
            # Draw boundaries
            self.draw_boundaries()
            
            # Draw target zone (only if spawned)
            target_spawned_current = info.get("target_spawned", self.env.step_count >= 50) if not paused else False
            self.draw_target_zone(target_spawned=target_spawned_current)
            
            # Draw wind arrows
            self.draw_wind(self.env.wind_x, self.env.wind_y)
            
            # Draw drone
            self.draw_drone(self.env.x, self.env.y, self.env.vx, self.env.vy)
            
            # Get in_target from info if available, otherwise compute
            if not paused and 'in_target' in locals():
                current_in_target = in_target
            else:
                from env.drone_env import TARGET_X_MIN, TARGET_X_MAX, TARGET_Y_MIN, TARGET_Y_MAX
                current_in_target = (
                    TARGET_X_MIN <= self.env.x <= TARGET_X_MAX and
                    TARGET_Y_MIN <= self.env.y <= TARGET_Y_MAX
                )
            
            # Draw info
            self.draw_info(step_count, reward if not paused else 0, action, current_in_target)
            
            # Draw pause indicator
            if paused:
                text = self.font.render("PAUSED (SPACE to resume)", True, YELLOW)
                text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, 30))
                self.screen.blit(text, text_rect)
            
            # Draw controls
            controls_y = WINDOW_HEIGHT - 80
            controls = [
                "SPACE: Pause/Resume",
                "R: Reset",
                "ESC: Quit"
            ]
            for i, control in enumerate(controls):
                text = self.small_font.render(control, True, GRAY)
                self.screen.blit(text, (10, controls_y + i * 20))
            
            pygame.display.flip()
            
            # Control speed
            if not paused:
                self.clock.tick(FPS * speed)
            else:
                self.clock.tick(10)
            
            # Auto-reset on done/truncated
            if (done or truncated) and not paused:
                pygame.time.wait(1000)  # Wait 1 second before reset
                obs, info = self.env.reset()
                done = False
                truncated = False
                step_count = 0
        
        pygame.quit()


def main():
    """Main function to run the visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize drone environment")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/liquid_policy.zip",
        help="Path to trained model (default: models/liquid_policy.zip)"
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random actions instead of trained model"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Animation speed multiplier (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # Create environment
    env = DroneWindEnv()
    
    # Load model if specified
    model = None
    if not args.random:
        if os.path.exists(args.model_path):
            print(f"Loading model from {args.model_path}...")
            model = PPO.load(args.model_path, env=env)
            print("Model loaded successfully!")
        else:
            print(f"Model not found at {args.model_path}, using random actions")
    
    # Create visualizer
    visualizer = DroneVisualizer(env, model)
    
    # Run visualization
    print("\nStarting visualization...")
    print("Controls:")
    print("  SPACE: Pause/Resume")
    print("  R: Reset episode")
    print("  ESC: Quit")
    print()
    
    visualizer.run(max_steps=args.max_steps, speed=args.speed)


if __name__ == "__main__":
    main()

