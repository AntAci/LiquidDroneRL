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
WIND_MAX = 1.5  # Maximum wind magnitude (reduced for smoother control)
WIND_SMOOTHING = 0.02  # Wind interpolation rate toward target (slower, smoother)
WIND_TARGET_INTERVAL = 50  # Steps between sampling new wind target
MAX_STEPS = 500  # Maximum episode length
POSITION_MIN = 0.0  # Minimum position (x, y)
POSITION_MAX = 1.0  # Maximum position (x, y)
THRUST = 0.26  # Thrust magnitude per action (increased to handle gravity + wind)
GRAVITY = -1.8  # Constant vertical acceleration (downwards), world units / s^2

# Target zone (box) constants
TARGET_REWARD = 4.0  # Bonus reward for being in target zone (increased to incentivize staying)
TARGET_SPAWN_DELAY = 50  # Steps before target zone appears (after wind starts)
BOUNDARY_CRASH_PENALTY = -20.0  # Large penalty for hitting boundary
TARGET_MOVE_SPEED = 0.015  # Target movement speed per step (world units)
TARGET_VELOCITY_UPDATE_INTERVAL = 100  # Steps between sampling new target velocity direction
TARGET_BOUNCE_ON_BOUNDARIES = True  # If True, target bounces off walls; if False, wraps around

# Stabilization and shaping
DRAG_COEFF = 0.45  # Linear velocity drag coefficient (increased damping)
SPEED_PENALTY_COEFF = 0.05  # Penalize high speeds to encourage smooth control
EDGE_MARGIN_FRAC = 0.06  # Fraction of world size near boundaries where penalty increases
EDGE_MARGIN = EDGE_MARGIN_FRAC * (POSITION_MAX - POSITION_MIN)
EDGE_PENALTY_COEFF = 0.5  # Strength of boundary proximity penalty
EFFORT_COEFF = 0.08  # Penalize thrust effort outside target (tuned for gravity)
EFFORT_IN_TARGET_MULT = 0.3  # Much weaker effort penalty inside target (allows thrust to counteract wind and stay)
BASE_STEP_COST = -0.01  # Small time penalty to discourage aimless survival
POTENTIAL_COEFF = 0.5  # Strength of potential-based shaping on distance reduction

# Target box relative size (fractions of world size)
TARGET_BOX_WIDTH_FRAC = 0.2
TARGET_BOX_HEIGHT_FRAC = 0.4

# Thrust slew limiting for smoother actuation (units of thrust per second)
SLEW_RATE = 3.0


class DroneWindEnv(gym.Env):
    """
    A 2D drone environment with dynamic wind.
    
    Observation: [x, y, vx, vy, wind_x, wind_y, dx_to_target, dy_to_target]
    Action: Box([-1, 1]^2) - continuous x/y thrust commands (normalized), scaled by THRUST
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(self, difficulty: int = 2):
        super().__init__()
        # Difficulty controls wind strength and noise (0..4)
        self.difficulty: int = int(difficulty)
        self._configure_difficulty()
        
        # Observation space: [x, y, vx, vy, wind_x, wind_y, dx_to_target, dy_to_target]
        self.observation_space = spaces.Box(
            low=np.array([POSITION_MIN, POSITION_MIN, -MAX_VEL, -MAX_VEL, -self.wind_max, -self.wind_max, -1.0, -1.0], dtype=np.float32),
            high=np.array([POSITION_MAX, POSITION_MAX, MAX_VEL, MAX_VEL, self.wind_max, self.wind_max, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: 2D continuous thrust command in [-1, 1] for x and y
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
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
        # Applied thrust (smoothed)
        self.ax_applied: float = 0.0
        self.ay_applied: float = 0.0
        # Track previous distance to target center for potential shaping
        self.prev_distance_to_target: Optional[float] = None
        # Target movement state
        self.target_vx: float = 0.0  # Target velocity in x direction
        self.target_vy: float = 0.0  # Target velocity in y direction
        self.target_velocity_target_x: float = 0.0  # Target velocity direction (x)
        self.target_velocity_target_y: float = 0.0  # Target velocity direction (y)
        
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
        self.ax_applied = 0.0
        self.ay_applied = 0.0
        # Randomize target box position (fixed size, random location)
        world_size = POSITION_MAX - POSITION_MIN
        box_w = TARGET_BOX_WIDTH_FRAC * world_size
        box_h = TARGET_BOX_HEIGHT_FRAC * world_size
        # Keep a small margin from borders
        margin = 0.05 * world_size
        x_min_low = POSITION_MIN + margin
        x_min_high = POSITION_MAX - margin - box_w
        y_min_low = POSITION_MIN + margin
        y_min_high = POSITION_MAX - margin - box_h
        if x_min_high <= x_min_low:
            x_min_high = x_min_low
        if y_min_high <= y_min_low:
            y_min_high = y_min_low
        self.target_x_min = float(self.np_random.uniform(x_min_low, x_min_high))
        self.target_y_min = float(self.np_random.uniform(y_min_low, y_min_high))
        self.target_x_max = self.target_x_min + box_w
        self.target_y_max = self.target_y_min + box_h
        # Initialize target movement (random initial velocity direction)
        angle = self.np_random.uniform(0, 2 * np.pi)
        self.target_vx = TARGET_MOVE_SPEED * np.cos(angle)
        self.target_vy = TARGET_MOVE_SPEED * np.sin(angle)
        self.target_velocity_target_x = self.target_vx
        self.target_velocity_target_y = self.target_vy
        # Initialize previous distance to target center (after target is defined)
        tx_c = (self.target_x_min + self.target_x_max) * 0.5
        ty_c = (self.target_y_min + self.target_y_max) * 0.5
        dx0 = self.x - tx_c
        dy0 = self.y - ty_c
        self.prev_distance_to_target = float(np.sqrt(dx0 * dx0 + dy0 * dy0))
        
        # Build observation
        obs = self._get_observation()
        info = {"target_spawned": False, "in_target": False}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Continuous action array [ax_cmd, ay_cmd] in [-1, 1]
            
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
        
        # Check if target has spawned
        target_spawned = self.step_count >= TARGET_SPAWN_DELAY
        
        # Update target position (move the target zone) if spawned
        if target_spawned:
            self._update_target_position()
        
        # Apply physics update
        self._apply_physics(action)
        
        # Compute reward
        base_reward = BASE_STEP_COST  # Small time penalty
        in_target = (
            target_spawned
            and self.target_x_min <= self.x <= self.target_x_max
            and self.target_y_min <= self.y <= self.target_y_max
        )
        target_bonus = TARGET_REWARD if in_target else 0.0
        
        # Potential-based shaping: reward progress toward target center
        shaping_reward = 0.0
        # Compute current distance
        tx_c_now = (self.target_x_min + self.target_x_max) * 0.5
        ty_c_now = (self.target_y_min + self.target_y_max) * 0.5
        dx_now = self.x - tx_c_now
        dy_now = self.y - ty_c_now
        current_distance = float(np.sqrt(dx_now * dx_now + dy_now * dy_now))
        if target_spawned and self.prev_distance_to_target is not None:
            shaping_reward = POTENTIAL_COEFF * (self.prev_distance_to_target - current_distance)
        # Update previous distance
        self.prev_distance_to_target = current_distance
        
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
        
        # Effort penalty (hover-offset): do not penalize the vertical thrust portion that merely counters gravity
        hover_comp = max(0.0, -GRAVITY * DT)  # upward thrust needed to counter gravity per step
        if self.ay_applied > 0.0:
            vertical_extra = max(0.0, self.ay_applied - hover_comp)
        else:
            vertical_extra = -self.ay_applied  # penalize downward thrust fully
        # Penalize horizontal thrust fully
        ax_eff = self.ax_applied
        # Normalized squared effort using effective components
        norm_sq = (ax_eff * ax_eff + vertical_extra * vertical_extra) / (THRUST * THRUST if THRUST > 0 else 1.0)
        effort_penalty = -EFFORT_COEFF * float(norm_sq) * (EFFORT_IN_TARGET_MULT if in_target else 1.0)
        
        # Check termination (boundary crash) BEFORE computing final reward
        terminated = (
            self.x <= POSITION_MIN or 
            self.x >= POSITION_MAX or 
            self.y <= POSITION_MIN or 
            self.y >= POSITION_MAX
        )
        
        # Apply boundary crash penalty if terminated
        boundary_penalty = BOUNDARY_CRASH_PENALTY if terminated else 0.0
        
        reward = base_reward + target_bonus + speed_penalty + edge_penalty + effort_penalty + shaping_reward + boundary_penalty
        
        # Check truncation (max steps)
        truncated = self.step_count >= MAX_STEPS
        
        # Build observation
        obs = self._get_observation()

        # Distance to target center (when available)
        distance_to_target = None
        target_center = None
        if target_spawned:
            tx_c = (self.target_x_min + self.target_x_max) * 0.5
            ty_c = (self.target_y_min + self.target_y_max) * 0.5
            target_center = (tx_c, ty_c)
            dx = self.x - tx_c
            dy = self.y - ty_c
            distance_to_target = float(np.sqrt(dx * dx + dy * dy))
        
        info = {
            "step_count": self.step_count,
            "in_target": in_target,
            "target_spawned": target_spawned,
            "target_bounds": (self.target_x_min, self.target_x_max, self.target_y_min, self.target_y_max),
            "thrust_applied": float(np.sqrt(self.ax_applied * self.ax_applied + self.ay_applied * self.ay_applied)),
            "effort": -float(effort_penalty),  # positive effort cost magnitude
            "distance_to_target": distance_to_target,
            "target_center": target_center,
            "distance_shaping": shaping_reward,
            "base_cost": base_reward,
            "speed_penalty": speed_penalty,
            "edge_penalty": edge_penalty,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _update_wind(self) -> None:
        """Update wind by smoothly moving toward target, resampling target periodically."""
        # Resample wind target every WIND_TARGET_INTERVAL steps
        if self.step_count % WIND_TARGET_INTERVAL == 0:
            self.wind_target_x = self.np_random.uniform(-self.wind_max, self.wind_max)
            self.wind_target_y = self.np_random.uniform(-self.wind_max, self.wind_max)
        
        # Smoothly interpolate wind toward target
        self.wind_x += WIND_SMOOTHING * (self.wind_target_x - self.wind_x)
        self.wind_y += WIND_SMOOTHING * (self.wind_target_y - self.wind_y)
        
        # Add turbulence for higher difficulty (random small perturbations)
        if self.difficulty >= 3 and self.turbulence_std > 0.0:
            self.wind_x += self.np_random.normal(loc=0.0, scale=self.turbulence_std)
            self.wind_y += self.np_random.normal(loc=0.0, scale=self.turbulence_std)
        
        # Add occasional gusts for the highest difficulty
        if self.difficulty >= 4 and self.np_random.random() < self.gust_prob:
            amp = float(self.gust_amp_mult * self.wind_max)
            self.wind_x += self.np_random.uniform(-amp, amp)
            self.wind_y += self.np_random.uniform(-amp, amp)
        
        # Clamp wind to bounds
        self.wind_x = np.clip(self.wind_x, -self.wind_max, self.wind_max)
        self.wind_y = np.clip(self.wind_y, -self.wind_max, self.wind_max)
    
    def _update_target_position(self) -> None:
        """Update target zone position by moving it smoothly, changing direction periodically."""
        # Resample target velocity direction periodically
        if self.step_count % TARGET_VELOCITY_UPDATE_INTERVAL == 0:
            # Sample new random direction
            angle = self.np_random.uniform(0, 2 * np.pi)
            self.target_velocity_target_x = TARGET_MOVE_SPEED * np.cos(angle)
            self.target_velocity_target_y = TARGET_MOVE_SPEED * np.sin(angle)
        
        # Smoothly interpolate target velocity toward target direction (similar to wind)
        self.target_vx += WIND_SMOOTHING * (self.target_velocity_target_x - self.target_vx)
        self.target_vy += WIND_SMOOTHING * (self.target_velocity_target_y - self.target_vy)
        
        # Clamp target velocity magnitude
        speed = np.sqrt(self.target_vx**2 + self.target_vy**2)
        if speed > TARGET_MOVE_SPEED * 1.5:  # Allow some overshoot
            self.target_vx = (self.target_vx / speed) * TARGET_MOVE_SPEED * 1.5
            self.target_vy = (self.target_vy / speed) * TARGET_MOVE_SPEED * 1.5
        
        # Calculate target box dimensions
        world_size = POSITION_MAX - POSITION_MIN
        box_w = TARGET_BOX_WIDTH_FRAC * world_size
        box_h = TARGET_BOX_HEIGHT_FRAC * world_size
        margin = 0.05 * world_size
        
        # Move target box
        new_x_min = self.target_x_min + self.target_vx
        new_y_min = self.target_y_min + self.target_vy
        new_x_max = new_x_min + box_w
        new_y_max = new_y_min + box_h
        
        # Handle boundary collisions
        if TARGET_BOUNCE_ON_BOUNDARIES:
            # Bounce off boundaries
            if new_x_min < POSITION_MIN + margin:
                new_x_min = POSITION_MIN + margin
                self.target_vx = -self.target_vx
                self.target_velocity_target_x = -self.target_velocity_target_x
            elif new_x_max > POSITION_MAX - margin:
                new_x_min = POSITION_MAX - margin - box_w
                self.target_vx = -self.target_vx
                self.target_velocity_target_x = -self.target_velocity_target_x
            
            if new_y_min < POSITION_MIN + margin:
                new_y_min = POSITION_MIN + margin
                self.target_vy = -self.target_vy
                self.target_velocity_target_y = -self.target_velocity_target_y
            elif new_y_max > POSITION_MAX - margin:
                new_y_min = POSITION_MAX - margin - box_h
                self.target_vy = -self.target_vy
                self.target_velocity_target_y = -self.target_velocity_target_y
        else:
            # Wrap around boundaries
            if new_x_min < POSITION_MIN + margin:
                new_x_min = POSITION_MAX - margin - box_w
            elif new_x_max > POSITION_MAX - margin:
                new_x_min = POSITION_MIN + margin
            
            if new_y_min < POSITION_MIN + margin:
                new_y_min = POSITION_MAX - margin - box_h
            elif new_y_max > POSITION_MAX - margin:
                new_y_min = POSITION_MIN + margin
        
        # Update target box position
        self.target_x_min = new_x_min
        self.target_y_min = new_y_min
        self.target_x_max = new_x_min + box_w
        self.target_y_max = new_y_min + box_h
    
    def _apply_physics(self, action: np.ndarray) -> None:
        """Apply physics update: convert action to thrust, update velocity and position."""
        # Expect 2D continuous action in [-1, 1]
        ax_cmd: float
        ay_cmd: float
        if isinstance(action, (list, tuple, np.ndarray)) and len(action) == 2:
            ax_cmd = float(np.clip(action[0], -1.0, 1.0))
            ay_cmd = float(np.clip(action[1], -1.0, 1.0))
        else:
            # Fallback: treat invalid action as no thrust
            ax_cmd, ay_cmd = 0.0, 0.0
        # Map to target thrust
        ax_target, ay_target = THRUST * ax_cmd, THRUST * ay_cmd
        
        # Smooth thrust using slew limiting
        max_delta = SLEW_RATE * DT
        delta_ax = ax_target - self.ax_applied
        if delta_ax > max_delta:
            delta_ax = max_delta
        elif delta_ax < -max_delta:
            delta_ax = -max_delta
        self.ax_applied += delta_ax
        
        delta_ay = ay_target - self.ay_applied
        if delta_ay > max_delta:
            delta_ay = max_delta
        elif delta_ay < -max_delta:
            delta_ay = -max_delta
        self.ay_applied += delta_ay
        
        # Update velocity with smoothed thrust, wind and gravity
        self.vx = self.vx + self.ax_applied + self.wind_x * DT
        self.vy = self.vy + self.ay_applied + self.wind_y * DT + GRAVITY * DT
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
        # Compute relative vector to target center
        tx_c = (self.target_x_min + self.target_x_max) * 0.5
        ty_c = (self.target_y_min + self.target_y_max) * 0.5
        dx = float(tx_c - self.x)
        dy = float(ty_c - self.y)
        return np.array(
            [self.x, self.y, self.vx, self.vy, self.wind_x, self.wind_y, dx, dy],
            dtype=np.float32
        )
    
    def _configure_difficulty(self) -> None:
        """
        Configure wind bounds and stochasticity based on difficulty level.
        
        Difficulty mapping (wind_max):
            0 -> 0.0 (no wind)
            1 -> 0.5 (mild)
            2 -> 1.0 (medium)
            3 -> 2.0 (chaotic + turbulence)
            4 -> 2.5 (chaotic + stronger turbulence + gusts)
            5 -> 3.0 (test-only: very strong wind + heavier turbulence + frequent strong gusts)
        """
        level_to_wind_max = {
            0: 0.0,
            1: 0.5,
            2: 1.0,
            3: 2.0,
            4: 2.5,
            5: 3.0,
        }
        self.wind_max: float = float(level_to_wind_max.get(self.difficulty, 1.0))
        # Turbulence amplitude (std dev) as a fraction of wind_max per step
        if self.difficulty >= 5:
            self.turbulence_std: float = 0.08 * self.wind_max
            self.gust_prob: float = 0.02
            self.gust_amp_mult: float = 2.5
        elif self.difficulty >= 4:
            self.turbulence_std = 0.06 * self.wind_max
            self.gust_prob = 0.01
            self.gust_amp_mult = 2.0
        elif self.difficulty >= 3:
            self.turbulence_std = 0.04 * self.wind_max
            self.gust_prob = 0.0
            self.gust_amp_mult = 2.0
        else:
            self.turbulence_std = 0.0
            self.gust_prob = 0.0
            self.gust_amp_mult = 2.0
    
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

