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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.drone_env import POSITION_MIN, POSITION_MAX


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
    
    def __init__(self, env: DroneWindEnv, model: Optional[PPO] = None, vec_env: Optional[VecNormalize] = None, model_name: str = "AI Agent"):
        """
        Initialize the visualizer.
        
        Args:
            env: DroneWindEnv instance (underlying unwrapped env)
            model: Optional trained PPO model (if None, uses random actions)
            vec_env: Optional VecNormalize wrapper (for normalized observations)
            model_name: Name of the model to display (e.g., "Liquid NN", "MLP Baseline", "Random Actions")
        """
        self.env = env
        self.model = model
        self.vec_env = vec_env
        self.model_name = model_name
        self.current_difficulty = getattr(env, 'difficulty', 2)  # Track current difficulty
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Drone RL - Visual Demonstration")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # World to screen scaling (generalized world range)
        self.world_margin = 50
        self.world_width = WINDOW_WIDTH - 2 * self.world_margin
        self.world_height = WINDOW_HEIGHT - 2 * self.world_margin
        
    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates [POSITION_MIN, POSITION_MAX] to screen coordinates."""
        xr = (x - POSITION_MIN) / (POSITION_MAX - POSITION_MIN)
        yr = (y - POSITION_MIN) / (POSITION_MAX - POSITION_MIN)
        screen_x = int(self.world_margin + xr * self.world_width)
        # Flip y-axis (world y min at bottom)
        screen_y = int(WINDOW_HEIGHT - self.world_margin - yr * self.world_height)
        return screen_x, screen_y
    
    def draw_drone(self, x: float, y: float, vx: float, vy: float):
        """Draw the drone as a circle with velocity vector."""
        screen_x, screen_y = self.world_to_screen(x, y)
        
        # Draw drone body (circle)
        drone_radius = 7  # Half the original size
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
        top_left = self.world_to_screen(POSITION_MIN, POSITION_MAX)
        top_right = self.world_to_screen(POSITION_MAX, POSITION_MAX)
        pygame.draw.line(self.screen, RED, top_left, top_right, 3)
        
        # Bottom boundary
        bot_left = self.world_to_screen(POSITION_MIN, POSITION_MIN)
        bot_right = self.world_to_screen(POSITION_MAX, POSITION_MIN)
        pygame.draw.line(self.screen, RED, bot_left, bot_right, 3)
        
        # Left boundary
        pygame.draw.line(self.screen, RED, top_left, bot_left, 3)
        
        # Right boundary
        pygame.draw.line(self.screen, RED, top_right, bot_right, 3)
    
    def draw_target_zone(self, target_spawned: bool = True):
        """Draw the target zone (box) that the drone needs to reach."""
        # Only draw if target has spawned
        if not target_spawned:
            return
        
        # Use env's randomized target bounds, with fallbacks
        tx_min = getattr(self.env, 'target_x_min', POSITION_MIN + 0.7 * (POSITION_MAX - POSITION_MIN))
        tx_max = getattr(self.env, 'target_x_max', POSITION_MIN + 0.9 * (POSITION_MAX - POSITION_MIN))
        ty_min = getattr(self.env, 'target_y_min', POSITION_MIN + 0.3 * (POSITION_MAX - POSITION_MIN))
        ty_max = getattr(self.env, 'target_y_max', POSITION_MIN + 0.7 * (POSITION_MAX - POSITION_MIN))
        # Get screen coordinates for target zone corners
        top_left = self.world_to_screen(tx_min, ty_max)
        top_right = self.world_to_screen(tx_max, ty_max)
        bot_left = self.world_to_screen(tx_min, ty_min)
        bot_right = self.world_to_screen(tx_max, ty_min)
        
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
        
        # Difficulty level
        difficulty_names = {0: "No Wind", 1: "Mild", 2: "Medium", 3: "Chaotic", 4: "Strong", 5: "Extreme"}
        diff_name = difficulty_names.get(self.current_difficulty, f"Level {self.current_difficulty}")
        diff_color = GREEN if self.current_difficulty <= 1 else YELLOW if self.current_difficulty <= 2 else ORANGE if self.current_difficulty <= 3 else RED
        text = self.font.render(f"Wind Difficulty: {self.current_difficulty} ({diff_name})", True, diff_color)
        self.screen.blit(text, (10, y_offset))
        y_offset += 30
        
        # In target zone status
        target_color = GREEN if in_target else GRAY
        target_text = "IN TARGET ZONE!" if in_target else "Not in target"
        text = self.font.render(target_text, True, target_color)
        self.screen.blit(text, (10, y_offset))
        y_offset += 30
        
        # Get underlying env for info display
        info_env = self.vec_env.envs[0] if self.vec_env is not None else self.env
        if hasattr(info_env, 'env'):
            info_env = info_env.env  # Unwrap Monitor if present
        
        # Position
        text = self.small_font.render(f"Position: ({info_env.x:.2f}, {info_env.y:.2f})", True, WHITE)
        self.screen.blit(text, (10, y_offset))
        y_offset += 25
        
        # Velocity
        text = self.small_font.render(f"Velocity: ({info_env.vx:.2f}, {info_env.vy:.2f})", True, WHITE)
        self.screen.blit(text, (10, y_offset))
        y_offset += 25
        
        # Wind
        text = self.small_font.render(f"Wind: ({info_env.wind_x:.2f}, {info_env.wind_y:.2f})", True, GREEN)
        self.screen.blit(text, (10, y_offset))
        y_offset += 25
        
        # Action (handle both discrete and continuous)
        if action is not None:
            if isinstance(action, (np.ndarray, list)) or hasattr(action, '__len__'):
                # Continuous action: show thrust vector
                if hasattr(info_env, 'ax_applied') and hasattr(info_env, 'ay_applied'):
                    # Show applied thrust (after smoothing)
                    ax = info_env.ax_applied
                    ay = info_env.ay_applied
                    thrust_mag = np.sqrt(ax**2 + ay**2)
                    text = self.small_font.render(
                        f"Thrust: ({ax:.3f}, {ay:.3f}) |{thrust_mag:.3f}|", 
                        True, YELLOW
                    )
                else:
                    # Fallback: show raw action
                    ax, ay = action[0], action[1]
                    text = self.small_font.render(
                        f"Action: ({ax:.3f}, {ay:.3f})", 
                        True, YELLOW
                    )
            else:
                # Discrete action (legacy support)
                action_names = ["No thrust", "Up", "Down", "Left", "Right"]
                action_idx = int(action) if hasattr(action, '__int__') else action
                text = self.small_font.render(f"Action: {action_names[action_idx]}", True, YELLOW)
            self.screen.blit(text, (10, y_offset))
            y_offset += 25
        
        # Model info
        if self.model is not None:
            text = self.small_font.render(f"Mode: AI Agent ({self.model_name})", True, CYAN)
        else:
            text = self.small_font.render(f"Mode: {self.model_name}", True, GRAY)
        self.screen.blit(text, (10, y_offset))
    
    def run(self, max_steps: int = 500, speed: float = 1.0):
        """
        Run the visualization.
        
        Args:
            max_steps: Maximum number of steps to run
            speed: Speed multiplier (1.0 = normal, higher = faster)
        """
        # Reset environment (use vec_env if available)
        if self.vec_env is not None:
            vec_obs = self.vec_env.reset()
            obs = vec_obs[0]
            info = {}  # VecEnv reset doesn't return info
        else:
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
                        if self.vec_env is not None:
                            vec_obs = self.vec_env.reset()
                            obs = vec_obs[0]
                            info = {}
                        else:
                            obs, info = self.env.reset()
                        done = False
                        truncated = False
                        step_count = 0
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]:
                        # Change difficulty level (0-5)
                        new_difficulty = event.key - pygame.K_0
                        if 0 <= new_difficulty <= 5:
                            self._change_difficulty(new_difficulty)
            
            if not paused and not done and not truncated:
                # Get action (use vec_env if available for normalization)
                if self.model is not None:
                    if self.vec_env is not None:
                        # Use vectorized env for normalized observations
                        action, _ = self.model.predict(obs, deterministic=True)
                    else:
                        action, _ = self.model.predict(obs, deterministic=True)
                else:
                    action = self.env.action_space.sample()
                
                # Step environment
                if self.vec_env is not None:
                    # Use vectorized env for stepping (handles normalization)
                    vec_obs, vec_reward, vec_done, vec_info = self.vec_env.step([action])
                    obs = vec_obs[0]
                    reward = vec_reward[0]
                    done = vec_done[0]
                    info = vec_info[0] if vec_info else {}
                    truncated = False  # VecEnv doesn't return truncated separately
                else:
                    obs, reward, done, truncated, info = self.env.step(action)
                step_count += 1
                # Get info from underlying env if using VecEnv
                if self.vec_env is not None:
                    # Access underlying env's attributes directly
                    underlying_env = self.vec_env.envs[0]
                    if hasattr(underlying_env, 'env'):
                        underlying_env = underlying_env.env  # Unwrap Monitor if present
                    in_target = getattr(underlying_env, 'target_x_min', None) is not None and (
                        underlying_env.target_x_min <= underlying_env.x <= underlying_env.target_x_max and
                        underlying_env.target_y_min <= underlying_env.y <= underlying_env.target_y_max
                    ) if underlying_env.step_count >= 50 else False
                    target_spawned = underlying_env.step_count >= 50
                else:
                    in_target = info.get("in_target", False)
                    target_spawned = info.get("target_spawned", False)
            
            # Draw everything
            self.screen.fill(BLACK)
            
            # Draw boundaries
            self.draw_boundaries()
            
            # Get underlying env for drawing
            draw_env = self.vec_env.envs[0] if self.vec_env is not None else self.env
            if hasattr(draw_env, 'env'):
                draw_env = draw_env.env  # Unwrap Monitor if present
            
            # Draw target zone (only if spawned)
            target_spawned_current = target_spawned if not paused else False
            self.draw_target_zone(target_spawned=target_spawned_current)
            
            # Draw wind arrows
            self.draw_wind(draw_env.wind_x, draw_env.wind_y)
            
            # Draw drone
            self.draw_drone(draw_env.x, draw_env.y, draw_env.vx, draw_env.vy)
            
            # Get in_target from info if available, otherwise compute
            if not paused and 'in_target' in locals():
                current_in_target = in_target
            else:
                tx_min = getattr(self.env, 'target_x_min', POSITION_MIN + 0.7 * (POSITION_MAX - POSITION_MIN))
                tx_max = getattr(self.env, 'target_x_max', POSITION_MIN + 0.9 * (POSITION_MAX - POSITION_MIN))
                ty_min = getattr(self.env, 'target_y_min', POSITION_MIN + 0.3 * (POSITION_MAX - POSITION_MIN))
                ty_max = getattr(self.env, 'target_y_max', POSITION_MIN + 0.7 * (POSITION_MAX - POSITION_MIN))
                current_in_target = (
                    tx_min <= self.env.x <= tx_max and
                    ty_min <= self.env.y <= ty_max
                )
            
            # Draw info
            self.draw_info(step_count, reward if not paused else 0, action, current_in_target)
            
            # Draw pause indicator
            if paused:
                text = self.font.render("PAUSED (SPACE to resume)", True, YELLOW)
                text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, 30))
                self.screen.blit(text, text_rect)
            
            # Draw controls
            controls_y = WINDOW_HEIGHT - 120
            controls = [
                "SPACE: Pause/Resume",
                "R: Reset",
                "0-5: Change Wind Difficulty",
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
                if self.vec_env is not None:
                    vec_obs = self.vec_env.reset()
                    obs = vec_obs[0]
                    info = {}  # VecEnv reset doesn't return info
                else:
                    obs, info = self.env.reset()
                done = False
                truncated = False
                step_count = 0
        
        pygame.quit()
    
    def _change_difficulty(self, new_difficulty: int) -> None:
        """
        Change the wind difficulty level during runtime.
        
        Args:
            new_difficulty: New difficulty level (0-5)
        """
        # Get the underlying environment
        if self.vec_env is not None:
            underlying_env = self.vec_env.envs[0]
            if hasattr(underlying_env, 'env'):
                underlying_env = underlying_env.env  # Unwrap Monitor if present
        else:
            underlying_env = self.env
        
        # Update difficulty
        if hasattr(underlying_env, 'difficulty'):
            underlying_env.difficulty = new_difficulty
            if hasattr(underlying_env, '_configure_difficulty'):
                underlying_env._configure_difficulty()
            self.current_difficulty = new_difficulty
            print(f"Wind difficulty changed to level {new_difficulty}")


def main():
    """Main function to run the visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize drone environment")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model (overrides --model option)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["liquid", "mlp", "random"],
        default="liquid",
        help="Which model to use: 'liquid' (default), 'mlp', or 'random'"
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random actions instead of trained model (deprecated: use --model random)"
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
    
    # Determine model path based on arguments
    if args.model_path:
        model_path = args.model_path
        model_name = "Custom"
    elif args.random or args.model == "random":
        model_path = None
        model_name = "Random"
    elif args.model == "mlp":
        model_path = "models/mlp_baseline.zip"
        model_name = "MLP Baseline"
    else:  # args.model == "liquid" (default)
        model_path = "models/liquid_policy.zip"
        model_name = "Liquid NN"
    
    # Create environment (vectorized for VecNormalize compatibility)
    def _make_env():
        return DroneWindEnv()
    venv = DummyVecEnv([_make_env])
    
    # Load VecNormalize stats if available
    vec_env = venv
    if model_path:
        vecnorm_path = model_path.replace(".zip", "_vecnorm.pkl")
        if os.path.exists(vecnorm_path):
            print(f"Loading VecNormalize stats from {vecnorm_path}...")
            vec_env = VecNormalize.load(vecnorm_path, venv)
            vec_env.training = False
            vec_env.norm_reward = False
        else:
            print("VecNormalize stats not found; using unnormalized env.")
    
    # Load model if specified
    model = None
    if model_path:
        if os.path.exists(model_path):
            print(f"Loading {model_name} model from {model_path}...")
            model = PPO.load(model_path, env=vec_env)
            print(f"{model_name} model loaded successfully!")
        else:
            print(f"Model not found at {model_path}, using random actions")
            model_path = None
            model_name = "Random"
    
    # Get the underlying environment for visualization
    env = venv.envs[0]
    
    # Create visualizer with model name
    visualizer = DroneVisualizer(env, model, vec_env, model_name=model_name)
    
    # Run visualization
    print("\nStarting visualization...")
    print(f"Model: {model_name}")
    print("Controls:")
    print("  SPACE: Pause/Resume")
    print("  R: Reset episode")
    print("  0-5: Change wind difficulty (0=No wind, 1=Mild, 2=Medium, 3=Chaotic, 4=Strong, 5=Extreme)")
    print("  ESC: Quit")
    print()
    print("To run different models:")
    print("  python demo/visualize_drone.py --model liquid  # Liquid NN (default)")
    print("  python demo/visualize_drone.py --model mlp     # MLP Baseline")
    print("  python demo/visualize_drone.py --model random  # Random actions")
    print()
    
    visualizer.run(max_steps=args.max_steps, speed=args.speed)


if __name__ == "__main__":
    main()

