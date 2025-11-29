import numpy as np
import gymnasium as gym
from opensimplex import OpenSimplex
import time

# PyFlyt import removed to avoid PyBullet overhead/issues
# try:
#     from PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env import QuadXWaypointsEnv
#     PYFLYT_AVAILABLE = True
# except ImportError:
#     PYFLYT_AVAILABLE = False
#     # Create a dummy base class if PyFlyt is missing
#     class QuadXWaypointsEnv(gym.Env):
#         metadata = {"render_modes": ["human", "rgb_array"]}
#         def __init__(self, render_mode=None):
#             self.render_mode = render_mode

class WindField:
    def __init__(self, seed=42, scale=0.1, speed=1.0):
        self.noise = OpenSimplex(seed=seed)
        self.scale = scale
        self.speed = speed
        self.time_offset = 0.0

    def get_wind(self, x, y, z, dt):
        self.time_offset += dt * self.speed
        u = self.noise.noise4(x * self.scale, y * self.scale, z * self.scale, self.time_offset)
        v = self.noise.noise4(x * self.scale + 100, y * self.scale + 100, z * self.scale, self.time_offset)
        w = self.noise.noise4(x * self.scale + 200, y * self.scale + 200, z * self.scale, self.time_offset)
        return np.array([u, v, w])

class Drone3DEnv(gym.Env):
    def __init__(self, render_mode=None, wind_scale=10.0, wind_speed=1.0):
        super().__init__()
        self.render_mode = render_mode
        self.wind_field = WindField(scale=0.05, speed=wind_speed)
        self.wind_strength = wind_scale
        
        # Define spaces
        # Obs: [x, y, z, roll, pitch, yaw, u, v, w, p, q, r]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        # Action: [motor1, motor2, motor3, motor4] or [thrust, roll, pitch, yaw]
        # We'll assume simple [thrust_x, thrust_y, thrust_z, yaw] for the mock
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        self.state = np.zeros(12)
        self.dt = 0.05
        self.step_count = 0
        self.max_steps = 1000

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(12)
        self.state[2] = 10.0 # Start at 10m height
        self.step_count = 0
        
        # Randomize Target
        # x, y in [-5, 5], z in [5, 15]
        self.target = np.random.uniform(low=[-5, -5, 5], high=[5, 5, 15])
        
        # Randomize Wind
        # We can just re-initialize the noise with a random seed
        new_seed = np.random.randint(0, 10000)
        self.wind_field = WindField(seed=new_seed, scale=0.05, speed=self.wind_field.speed)
        
        return self.state.astype(np.float32), {}

    def step(self, action):
        self.step_count += 1
        
        # Unpack state
        pos = self.state[0:3]
        vel = self.state[6:9]
        
        # Get Wind
        raw_wind = self.wind_field.get_wind(pos[0], pos[1], pos[2], self.dt)
        wind_force = raw_wind * self.wind_strength
        
        # Simple Kinematics (Double Integrator)
        # Action is roughly acceleration command
        # We need enough authority to fight gravity (9.81) + wind
        # Let's say max thrust is 20 m/s^2 (~2G)
        # Action [-1, 1] -> [-20, 20] ? 
        # No, usually thrust is positive 0..Max. 
        # But for simplified "QuadX" control often inputs are roll/pitch/yaw/thrust.
        # Here we are abstracting to "Force/Accel command in 3D".
        # Let's map action [-1, 1] to [-15, 15] acceleration.
        accel = action[:3] * 15.0 
        
        # Gravity
        gravity = np.array([0, 0, -9.81])
        
        # Total Force = Control + Wind + Gravity
        # Note: We REMOVED the "anti-gravity" offset. 
        # The agent MUST output positive Z acceleration to hover.
        # If action[2] is 0, accel[2] is 0, and it falls due to gravity.
        total_accel = accel + wind_force + gravity
        
        # Update State
        vel += total_accel * self.dt
        pos += vel * self.dt
        
        # Floor collision
        if pos[2] < 0:
            pos[2] = 0
            vel[2] = 0 # Crash stop
        
        # Drag (Damping)
        vel *= 0.95
        
        self.state[0:3] = pos
        self.state[6:9] = vel
        
        # Reward: Stay close to Target
        dist = np.linalg.norm(pos - self.target)
        
        # Smoothness: Penalty for high velocity (instability)
        vel_mag = np.linalg.norm(vel)
        
        # Components:
        # 1. Distance Reward: Higher is better (closer to 0)
        r_dist = -dist
        
        # 2. Stability Penalty: Penalize erratic high-speed movements if far from target
        # But we need speed to get there. Let's just penalize extreme speed.
        r_vel = -0.01 * vel_mag
        
        # 3. Survival Reward: Bonus for not crashing
        r_survive = 0.1
        
        reward = r_dist + r_vel + r_survive
        
        # Terminate if crashed or too far
        term = False # Let it crash and stay on floor
        trunc = self.step_count >= self.max_steps
        
        info = {"wind": wind_force, "target": self.target}
        
        return self.state.astype(np.float32), reward, term, trunc, info

    def render(self):
        # We will handle rendering in the demo script using matplotlib
        pass
