"""
Side-by-side visualization comparing MLP and Liquid NN models.

This script runs two instances of the drone environment side-by-side,
one controlled by MLP baseline and one by Liquid NN, for direct comparison.
"""

import os
import sys
import pygame
import numpy as np
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.drone_env import DroneWindEnv, POSITION_MIN, POSITION_MAX
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Pygame constants
WINDOW_WIDTH = 1600  # Wider for side-by-side
WINDOW_HEIGHT = 600
FPS = 30
SIDE_WIDTH = WINDOW_WIDTH // 2  # Each side gets half the width

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
ORANGE = (255, 165, 0)


class SideBySideVisualizer:
    """Side-by-side visualizer for comparing two models."""
    
    def __init__(self, mlp_model: Optional[PPO], liquid_model: Optional[PPO], 
                 mlp_vec_env: Optional[VecNormalize], liquid_vec_env: Optional[VecNormalize]):
        """
        Initialize the side-by-side visualizer.
        
        Args:
            mlp_model: MLP PPO model (left side)
            liquid_model: Liquid NN PPO model (right side)
            mlp_vec_env: VecNormalize wrapper for MLP (if available)
            liquid_vec_env: VecNormalize wrapper for Liquid (if available)
        """
        self.mlp_model = mlp_model
        self.liquid_model = liquid_model
        self.mlp_vec_env = mlp_vec_env
        self.liquid_vec_env = liquid_vec_env
        
        # Get underlying environments
        if mlp_vec_env is not None:
            self.mlp_env = mlp_vec_env.envs[0]
            if hasattr(self.mlp_env, 'env'):
                self.mlp_env = self.mlp_env.env
        else:
            self.mlp_env = None
            
        if liquid_vec_env is not None:
            self.liquid_env = liquid_vec_env.envs[0]
            if hasattr(self.liquid_env, 'env'):
                self.liquid_env = self.liquid_env.env
        else:
            self.liquid_env = None
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Drone RL - MLP vs Liquid NN Comparison")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # World to screen scaling (for each side)
        self.world_margin = 50
        self.world_width = SIDE_WIDTH - 2 * self.world_margin
        self.world_height = WINDOW_HEIGHT - 2 * self.world_margin
        
        self.current_difficulty = 2  # Track current difficulty
        
    def world_to_screen(self, x: float, y: float, side: str = "left") -> tuple[int, int]:
        """Convert world coordinates to screen coordinates for left or right side."""
        xr = (x - POSITION_MIN) / (POSITION_MAX - POSITION_MIN)
        yr = (y - POSITION_MIN) / (POSITION_MAX - POSITION_MIN)
        offset_x = 0 if side == "left" else SIDE_WIDTH
        screen_x = int(offset_x + self.world_margin + xr * self.world_width)
        screen_y = int(WINDOW_HEIGHT - self.world_margin - yr * self.world_height)
        return screen_x, screen_y
    
    def draw_drone(self, x: float, y: float, vx: float, vy: float, side: str = "left", color: tuple = CYAN):
        """Draw the drone as a circle with velocity vector."""
        screen_x, screen_y = self.world_to_screen(x, y, side)
        drone_radius = 7
        pygame.draw.circle(self.screen, color, (screen_x, screen_y), drone_radius)
        pygame.draw.circle(self.screen, BLUE, (screen_x, screen_y), drone_radius, 2)
        
        # Draw velocity vector
        if abs(vx) > 0.01 or abs(vy) > 0.01:
            scale = 30
            end_x = screen_x + int(vx * scale)
            end_y = screen_y - int(vy * scale)
            pygame.draw.line(self.screen, YELLOW, (screen_x, screen_y), (end_x, end_y), 3)
    
    def draw_wind(self, wind_x: float, wind_y: float, side: str = "left"):
        """Draw wind arrows."""
        grid_size = 4  # Smaller grid for side-by-side
        for i in range(grid_size):
            for j in range(grid_size):
                x = (i + 0.5) / grid_size
                y = (j + 0.5) / grid_size
                screen_x, screen_y = self.world_to_screen(x, y, side)
                
                if abs(wind_x) > 0.01 or abs(wind_y) > 0.01:
                    scale = 20
                    end_x = screen_x + int(wind_x * scale)
                    end_y = screen_y - int(wind_y * scale)
                    
                    wind_strength = abs(wind_x) + abs(wind_y)
                    if wind_strength < 1.0:
                        color = GREEN
                    elif wind_strength < 1.5:
                        color = YELLOW
                    else:
                        color = ORANGE
                    
                    pygame.draw.line(self.screen, color, (screen_x, screen_y), (end_x, end_y), 2)
    
    def draw_target_zone(self, env, side: str = "left"):
        """Draw the target zone."""
        if not hasattr(env, 'target_x_min') or env.step_count < 50:
            return
        
        tx_min, tx_max = env.target_x_min, env.target_x_max
        ty_min, ty_max = env.target_y_min, env.target_y_max
        
        top_left = self.world_to_screen(tx_min, ty_max, side)
        top_right = self.world_to_screen(tx_max, ty_max, side)
        bot_left = self.world_to_screen(tx_min, ty_min, side)
        bot_right = self.world_to_screen(tx_max, ty_min, side)
        
        # Draw semi-transparent box
        target_surface = pygame.Surface((SIDE_WIDTH, WINDOW_HEIGHT))
        target_surface.set_alpha(100)
        rect = pygame.Rect(
            top_left[0] - (0 if side == "left" else SIDE_WIDTH), top_left[1],
            top_right[0] - top_left[0],
            bot_left[1] - top_left[1]
        )
        pygame.draw.rect(target_surface, MAGENTA, rect)
        self.screen.blit(target_surface, (0 if side == "left" else SIDE_WIDTH, 0))
        
        # Draw border
        pygame.draw.line(self.screen, MAGENTA, top_left, top_right, 2)
        pygame.draw.line(self.screen, MAGENTA, top_right, bot_right, 2)
        pygame.draw.line(self.screen, MAGENTA, bot_right, bot_left, 2)
        pygame.draw.line(self.screen, MAGENTA, bot_left, top_left, 2)
    
    def draw_boundaries(self, side: str = "left"):
        """Draw world boundaries."""
        offset_x = 0 if side == "left" else SIDE_WIDTH
        top_left = self.world_to_screen(POSITION_MIN, POSITION_MAX, side)
        top_right = self.world_to_screen(POSITION_MAX, POSITION_MAX, side)
        bot_left = self.world_to_screen(POSITION_MIN, POSITION_MIN, side)
        bot_right = self.world_to_screen(POSITION_MAX, POSITION_MIN, side)
        
        pygame.draw.line(self.screen, RED, top_left, top_right, 2)
        pygame.draw.line(self.screen, RED, bot_left, bot_right, 2)
        pygame.draw.line(self.screen, RED, top_left, bot_left, 2)
        pygame.draw.line(self.screen, RED, top_right, bot_right, 2)
    
    def draw_info(self, env, step: int, reward: float, side: str = "left", model_name: str = "Model"):
        """Draw information text."""
        offset_x = 0 if side == "left" else SIDE_WIDTH
        y_offset = 10
        
        # Model name
        text = self.font.render(f"{model_name}", True, WHITE)
        self.screen.blit(text, (offset_x + 10, y_offset))
        y_offset += 30
        
        # Step count
        text = self.font.render(f"Step: {step}", True, WHITE)
        self.screen.blit(text, (offset_x + 10, y_offset))
        y_offset += 25
        
        # Reward
        text = self.small_font.render(f"Reward: {reward:.2f}", True, WHITE)
        self.screen.blit(text, (offset_x + 10, y_offset))
        y_offset += 25
        
        # Position
        text = self.small_font.render(f"Pos: ({env.x:.2f}, {env.y:.2f})", True, WHITE)
        self.screen.blit(text, (offset_x + 10, y_offset))
        y_offset += 25
        
        # Velocity
        text = self.small_font.render(f"Vel: ({env.vx:.2f}, {env.vy:.2f})", True, WHITE)
        self.screen.blit(text, (offset_x + 10, y_offset))
        y_offset += 25
        
        # Wind
        text = self.small_font.render(f"Wind: ({env.wind_x:.2f}, {env.wind_y:.2f})", True, GREEN)
        self.screen.blit(text, (offset_x + 10, y_offset))
        y_offset += 25
        
        # In target
        in_target = (hasattr(env, 'target_x_min') and env.step_count >= 50 and
                    env.target_x_min <= env.x <= env.target_x_max and
                    env.target_y_min <= env.y <= env.target_y_max)
        target_color = GREEN if in_target else GRAY
        target_text = "IN TARGET!" if in_target else "Not in target"
        text = self.small_font.render(target_text, True, target_color)
        self.screen.blit(text, (offset_x + 10, y_offset))
    
    def draw_divider(self):
        """Draw divider line between left and right sides."""
        pygame.draw.line(self.screen, WHITE, (SIDE_WIDTH, 0), (SIDE_WIDTH, WINDOW_HEIGHT), 3)
        # Draw labels
        text = self.font.render("MLP Baseline", True, WHITE)
        text_rect = text.get_rect(center=(SIDE_WIDTH // 2, 30))
        self.screen.blit(text, text_rect)
        
        text = self.font.render("Liquid NN", True, WHITE)
        text_rect = text.get_rect(center=(SIDE_WIDTH + SIDE_WIDTH // 2, 30))
        self.screen.blit(text, text_rect)
    
    def _change_difficulty(self, new_difficulty: int):
        """Change difficulty for both environments."""
        for env in [self.mlp_env, self.liquid_env]:
            if env is not None and hasattr(env, 'difficulty'):
                env.difficulty = new_difficulty
                if hasattr(env, '_configure_difficulty'):
                    env._configure_difficulty()
        self.current_difficulty = new_difficulty
        print(f"Wind difficulty changed to level {new_difficulty}")
    
    def run(self, max_steps: int = 500):
        """Run the side-by-side visualization."""
        # Reset both environments
        if self.mlp_vec_env is not None:
            mlp_obs = self.mlp_vec_env.reset()
            mlp_obs = mlp_obs[0]
        else:
            mlp_obs, _ = self.mlp_env.reset()
        
        if self.liquid_vec_env is not None:
            liquid_obs = self.liquid_vec_env.reset()
            liquid_obs = liquid_obs[0]
        else:
            liquid_obs, _ = self.liquid_env.reset()
        
        mlp_done = False
        liquid_done = False
        mlp_step = 0
        liquid_step = 0
        mlp_reward = 0.0
        liquid_reward = 0.0
        
        running = True
        paused = False
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        # Reset both
                        if self.mlp_vec_env is not None:
                            mlp_obs = self.mlp_vec_env.reset()
                            mlp_obs = mlp_obs[0]
                        else:
                            mlp_obs, _ = self.mlp_env.reset()
                        
                        if self.liquid_vec_env is not None:
                            liquid_obs = self.liquid_vec_env.reset()
                            liquid_obs = liquid_obs[0]
                        else:
                            liquid_obs, _ = self.liquid_env.reset()
                        
                        mlp_done = False
                        liquid_done = False
                        mlp_step = 0
                        liquid_step = 0
                        mlp_reward = 0.0
                        liquid_reward = 0.0
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]:
                        new_difficulty = event.key - pygame.K_0
                        if 0 <= new_difficulty <= 5:
                            self._change_difficulty(new_difficulty)
            
            if not paused:
                # Step MLP side
                if not mlp_done and mlp_step < max_steps:
                    if self.mlp_model is not None:
                        mlp_action, _ = self.mlp_model.predict(mlp_obs, deterministic=True)
                    else:
                        mlp_action = self.mlp_env.action_space.sample()
                    
                    if self.mlp_vec_env is not None:
                        mlp_obs, mlp_r, mlp_d, mlp_info = self.mlp_vec_env.step([mlp_action])
                        mlp_obs = mlp_obs[0]
                        mlp_reward = mlp_r[0]
                        mlp_done = mlp_d[0]
                    else:
                        mlp_obs, mlp_reward, mlp_done, _, _ = self.mlp_env.step(mlp_action)
                    mlp_step += 1
                
                # Step Liquid side
                if not liquid_done and liquid_step < max_steps:
                    if self.liquid_model is not None:
                        liquid_action, _ = self.liquid_model.predict(liquid_obs, deterministic=True)
                    else:
                        liquid_action = self.liquid_env.action_space.sample()
                    
                    if self.liquid_vec_env is not None:
                        liquid_obs, liquid_r, liquid_d, liquid_info = self.liquid_vec_env.step([liquid_action])
                        liquid_obs = liquid_obs[0]
                        liquid_reward = liquid_r[0]
                        liquid_done = liquid_d[0]
                    else:
                        liquid_obs, liquid_reward, liquid_done, _, _ = self.liquid_env.step(liquid_action)
                    liquid_step += 1
            
            # Draw everything
            self.screen.fill(BLACK)
            
            # Draw divider
            self.draw_divider()
            
            # Draw MLP side (left)
            if self.mlp_env is not None:
                self.draw_boundaries("left")
                self.draw_target_zone(self.mlp_env, "left")
                self.draw_wind(self.mlp_env.wind_x, self.mlp_env.wind_y, "left")
                self.draw_drone(self.mlp_env.x, self.mlp_env.y, self.mlp_env.vx, self.mlp_env.vy, "left", CYAN)
                self.draw_info(self.mlp_env, mlp_step, mlp_reward, "left", "MLP Baseline")
            
            # Draw Liquid side (right)
            if self.liquid_env is not None:
                self.draw_boundaries("right")
                self.draw_target_zone(self.liquid_env, "right")
                self.draw_wind(self.liquid_env.wind_x, self.liquid_env.wind_y, "right")
                self.draw_drone(self.liquid_env.x, self.liquid_env.y, self.liquid_env.vx, self.liquid_env.vy, "right", GREEN)
                self.draw_info(self.liquid_env, liquid_step, liquid_reward, "right", "Liquid NN")
            
            # Draw controls
            controls_y = WINDOW_HEIGHT - 100
            controls = [
                "SPACE: Pause/Resume",
                "R: Reset both",
                "0-5: Change Wind Difficulty",
                "ESC: Quit"
            ]
            for i, control in enumerate(controls):
                text = self.small_font.render(control, True, GRAY)
                self.screen.blit(text, (WINDOW_WIDTH // 2 - 100, controls_y + i * 20))
            
            # Draw difficulty
            difficulty_names = {0: "No Wind", 1: "Mild", 2: "Medium", 3: "Chaotic", 4: "Strong", 5: "Extreme"}
            diff_name = difficulty_names.get(self.current_difficulty, f"Level {self.current_difficulty}")
            text = self.font.render(f"Wind Difficulty: {self.current_difficulty} ({diff_name})", True, YELLOW)
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 30))
            self.screen.blit(text, text_rect)
            
            pygame.display.flip()
            self.clock.tick(FPS)
            
            # Auto-reset if both done
            if (mlp_done or liquid_done) and not paused:
                pygame.time.wait(1000)
                if self.mlp_vec_env is not None:
                    mlp_obs = self.mlp_vec_env.reset()
                    mlp_obs = mlp_obs[0]
                else:
                    mlp_obs, _ = self.mlp_env.reset()
                
                if self.liquid_vec_env is not None:
                    liquid_obs = self.liquid_vec_env.reset()
                    liquid_obs = liquid_obs[0]
                else:
                    liquid_obs, _ = self.liquid_env.reset()
                
                mlp_done = False
                liquid_done = False
                mlp_step = 0
                liquid_step = 0
                mlp_reward = 0.0
                liquid_reward = 0.0
        
        pygame.quit()


def main():
    """Main function to run side-by-side comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Side-by-side comparison of MLP vs Liquid NN")
    parser.add_argument(
        "--mlp-path",
        type=str,
        default="models/mlp_baseline.zip",
        help="Path to MLP model (default: models/mlp_baseline.zip)"
    )
    parser.add_argument(
        "--liquid-path",
        type=str,
        default="models/liquid_policy.zip",
        help="Path to Liquid NN model (default: models/liquid_policy.zip)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500)"
    )
    
    args = parser.parse_args()
    
    # Create environments
    def _make_mlp_env():
        return DroneWindEnv(difficulty=2)
    def _make_liquid_env():
        return DroneWindEnv(difficulty=2)
    
    mlp_venv = DummyVecEnv([_make_mlp_env])
    liquid_venv = DummyVecEnv([_make_liquid_env])
    
    # Load VecNormalize stats if available
    mlp_vecnorm_path = args.mlp_path.replace(".zip", "_vecnorm.pkl")
    if os.path.exists(mlp_vecnorm_path):
        print(f"Loading MLP VecNormalize stats from {mlp_vecnorm_path}...")
        mlp_vec_env = VecNormalize.load(mlp_vecnorm_path, mlp_venv)
        mlp_vec_env.training = False
        mlp_vec_env.norm_reward = False
    else:
        print("MLP VecNormalize stats not found; using unnormalized env.")
        mlp_vec_env = mlp_venv
    
    liquid_vecnorm_path = args.liquid_path.replace(".zip", "_vecnorm.pkl")
    if os.path.exists(liquid_vecnorm_path):
        print(f"Loading Liquid VecNormalize stats from {liquid_vecnorm_path}...")
        liquid_vec_env = VecNormalize.load(liquid_vecnorm_path, liquid_venv)
        liquid_vec_env.training = False
        liquid_vec_env.norm_reward = False
    else:
        print("Liquid VecNormalize stats not found; using unnormalized env.")
        liquid_vec_env = liquid_venv
    
    # Load models
    mlp_model = None
    if os.path.exists(args.mlp_path):
        print(f"Loading MLP model from {args.mlp_path}...")
        mlp_model = PPO.load(args.mlp_path, env=mlp_vec_env)
        print("MLP model loaded successfully!")
    else:
        print(f"MLP model not found at {args.mlp_path}")
        return
    
    liquid_model = None
    if os.path.exists(args.liquid_path):
        print(f"Loading Liquid NN model from {args.liquid_path}...")
        liquid_model = PPO.load(args.liquid_path, env=liquid_vec_env)
        print("Liquid NN model loaded successfully!")
    else:
        print(f"Liquid NN model not found at {args.liquid_path}")
        return
    
    # Create visualizer
    visualizer = SideBySideVisualizer(mlp_model, liquid_model, mlp_vec_env, liquid_vec_env)
    
    # Run visualization
    print("\nStarting side-by-side comparison...")
    print("Controls:")
    print("  SPACE: Pause/Resume")
    print("  R: Reset both episodes")
    print("  0-5: Change wind difficulty (affects both)")
    print("  ESC: Quit")
    print()
    
    visualizer.run(max_steps=args.max_steps)


if __name__ == "__main__":
    main()

