"""
A 2D drone environment with dynamic wind forces for reinforcement learning.
The drone can apply discrete thrust actions while being affected by smoothly varying wind.
The goal is to navigate and survive within the bounded world.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


# Constants
DT = 0.1  # Time step
MAX_VEL = 2.0  # Maximum velocity magnitude
WIND_MAX = 2.0  # Maximum wind magnitude
WIND_SMOOTHING = 0.05  # Wind interpolation rate toward target
WIND_TARGET_INTERVAL = 50  # Steps between sampling new wind target
MAX_STEPS = 500  # Maximum episode length
POSITION_MIN = 0.0  # Minimum position (x, y)
POSITION_MAX = 1.0  # Maximum position (x, y)
THRUST = 0.25  # Thrust magnitude per action (slightly higher for control authority)

# Target zone (box) constants
TARGET_X_MIN = 0.7  # Target box left edge
TARGET_X_MAX = 0.9  # Target box right edge
TARGET_Y_MIN = 0.3  # Target box bottom edge
TARGET_Y_MAX = 0.7  # Target box top edge
TARGET_REWARD = 2.0  # Bonus reward for being in target zone
TARGET_SPAWN_DELAY = 50  # Steps before target zone appears (after wind starts)

# Stabilization and shaping
DRAG_COEFF = 0.3  # Linear velocity drag coefficient
SPEED_PENALTY_COEFF = 0.05  # Penalize high speeds to encourage smooth control
EDGE_MARGIN = 0.06  # Margin near boundaries where penalty increases
EDGE_PENALTY_COEFF = 0.5  # Strength of boundary proximity penalty


class DroneWindEnv(gym.Env):
    """
    A 2D drone environment with dynamic wind.
    
    Observation: [x, y, vx, vy, wind_x, wind_y]
    Action: Discrete(5) - 0: no thrust, 1: up, 2: down, 3: left, 4: right
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(self):
        super().__init__()
        
        # Observation space: [x, y, vx, vy, wind_x, wind_y]
        self.observation_space = spaces.Box(
            low=np.array([POSITION_MIN, POSITION_MIN, -MAX_VEL, -MAX_VEL, -WIND_MAX, -WIND_MAX], dtype=np.float32),
            high=np.array([POSITION_MAX, POSITION_MAX, MAX_VEL, MAX_VEL, WIND_MAX, WIND_MAX], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: 5 discrete thrust directions
        self.action_space = spaces.Discrete(5)
        
        # Internal state
        self.x: float = 0.0
        self.y: float = 0.0
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.wind_x: float = 0.0
        self.wind_y: float = 0.0
        self.wind_target_x: float = 0.0
        self.wind_target_y: float = 0.0
        self.step_count: int = 0
        
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Optional random seed
            options: Optional reset options
            
        Returns:
            observation: Initial observation array
            info: Empty info dict
        """
        # Always call super().reset to ensure seeding and np_random are initialized
        super().reset(seed=seed)
        
        # Initialize state
        self.x = 0.1
        self.y = 0.5
        self.vx = 0.0
        self.vy = 0.0
        self.wind_x = 0.0
        self.wind_y = 0.0
        self.wind_target_x = 0.0
        self.wind_target_y = 0.0
        self.step_count = 0
        
        # Build observation
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Discrete action (0-4)
            
        Returns:
            observation: New observation array
            reward: Reward for this step
            terminated: Whether episode ended due to boundary crash
            truncated: Whether episode ended due to max steps
            info: Info dict with step_count
        """
        # Increment step count
        self.step_count += 1
        
        # Update wind model
        self._update_wind()
        
        # Apply physics update
        self._apply_physics(action)
        
        # Compute reward
        base_reward = 1.0  # Survival reward
        
        # Check if drone is in target zone (only if target has spawned)
        target_spawned = self.step_count >= TARGET_SPAWN_DELAY
        in_target = False
        if target_spawned:
            in_target = (
                TARGET_X_MIN <= self.x <= TARGET_X_MAX and
                TARGET_Y_MIN <= self.y <= TARGET_Y_MAX
            )
        target_bonus = TARGET_REWARD if in_target else 0.0
        
        # Speed penalty (discourage excessive velocity)
        speed_sq = self.vx * self.vx + self.vy * self.vy
        speed_penalty = -SPEED_PENALTY_COEFF * float(speed_sq)
        # Boundary proximity penalty (discourage hovering near walls)
        dist_left = self.x - POSITION_MIN
        dist_right = POSITION_MAX - self.x
        dist_bottom = self.y - POSITION_MIN
        dist_top = POSITION_MAX - self.y
        min_dist = min(dist_left, dist_right, dist_bottom, dist_top)
        edge_penalty = 0.0
        if min_dist < EDGE_MARGIN:
            edge_penalty = -EDGE_PENALTY_COEFF * (EDGE_MARGIN - float(min_dist)) / EDGE_MARGIN
        
        reward = base_reward + target_bonus + speed_penalty + edge_penalty
        
        # Check termination (boundary crash)
        terminated = (
            self.x <= POSITION_MIN or 
            self.x >= POSITION_MAX or 
            self.y <= POSITION_MIN or 
            self.y >= POSITION_MAX
        )
        
        # Check truncation (max steps)
        truncated = self.step_count >= MAX_STEPS
        
        # Build observation
        obs = self._get_observation()
        
        # Check if in target zone (only if target has spawned)
        target_spawned = self.step_count >= TARGET_SPAWN_DELAY
        in_target = False
        if target_spawned:
            in_target = (
                TARGET_X_MIN <= self.x <= TARGET_X_MAX and
                TARGET_Y_MIN <= self.y <= TARGET_Y_MAX
            )
        info = {"step_count": self.step_count, "in_target": in_target, "target_spawned": target_spawned}
        
        return obs, reward, terminated, truncated, info
    
    def _update_wind(self) -> None:
        """Update wind by smoothly moving toward target, resampling target periodically."""
        # Resample wind target every WIND_TARGET_INTERVAL steps
        if self.step_count % WIND_TARGET_INTERVAL == 0:
            self.wind_target_x = self.np_random.uniform(-WIND_MAX, WIND_MAX)
            self.wind_target_y = self.np_random.uniform(-WIND_MAX, WIND_MAX)
        
        # Smoothly interpolate wind toward target
        self.wind_x += WIND_SMOOTHING * (self.wind_target_x - self.wind_x)
        self.wind_y += WIND_SMOOTHING * (self.wind_target_y - self.wind_y)
        
        # Clamp wind to bounds
        self.wind_x = np.clip(self.wind_x, -WIND_MAX, WIND_MAX)
        self.wind_y = np.clip(self.wind_y, -WIND_MAX, WIND_MAX)
    
    def _apply_physics(self, action: int) -> None:
        """Apply physics update: convert action to thrust, update velocity and position."""
        # Convert action to thrust vector
        if action == 0:  # No thrust
            ax, ay = 0.0, 0.0
        elif action == 1:  # Thrust up
            ax, ay = 0.0, THRUST
        elif action == 2:  # Thrust down
            ax, ay = 0.0, -THRUST
        elif action == 3:  # Thrust left
            ax, ay = -THRUST, 0.0
        elif action == 4:  # Thrust right
            ax, ay = THRUST, 0.0
        else:
            raise ValueError(f"Invalid action: {action}. Must be in [0, 4]")
        
        # Update velocity with thrust and wind
        self.vx = self.vx + ax + self.wind_x * DT
        self.vy = self.vy + ay + self.wind_y * DT
        # Apply linear drag (proportional to velocity) for stability
        self.vx -= DRAG_COEFF * self.vx * DT
        self.vy -= DRAG_COEFF * self.vy * DT
        
        # Clamp velocity
        self.vx = np.clip(self.vx, -MAX_VEL, MAX_VEL)
        self.vy = np.clip(self.vy, -MAX_VEL, MAX_VEL)
        
        # Update position
        self.x = self.x + self.vx * DT
        self.y = self.y + self.vy * DT
        
        # Clamp position to bounds
        self.x = np.clip(self.x, POSITION_MIN, POSITION_MAX)
        self.y = np.clip(self.y, POSITION_MIN, POSITION_MAX)
    
    def _get_observation(self) -> np.ndarray:
        """Build observation array from current state."""
        return np.array(
            [self.x, self.y, self.vx, self.vy, self.wind_x, self.wind_y],
            dtype=np.float32
        )
    
    def render(self) -> None:
        """
        Render the environment state (stub implementation for Phase 1).
        Prints state to stdout.
        """
        print(
            f"Step {self.step_count}: "
            f"x={self.x:.2f}, y={self.y:.2f}, "
            f"vx={self.vx:.2f}, vy={self.vy:.2f}, "
            f"wind=({self.wind_x:.2f}, {self.wind_y:.2f})"
        )


def make_drone_env() -> DroneWindEnv:
    """Helper function to create a DroneWindEnv instance."""
    return DroneWindEnv()


if __name__ == "__main__":
    # Manual test block
    print("Testing DroneWindEnv...")
    print("=" * 60)
    
    env = make_drone_env()
    obs, info = env.reset(seed=42)
    print(f"Initial observation: {obs}")
    print()
    
    for t in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated:
            print(f"\nEpisode terminated at step {t} (boundary crash)")
            break
        if truncated:
            print(f"\nEpisode truncated at step {t} (max steps reached)")
            break
    
    print("=" * 60)
    print("Test completed!")

