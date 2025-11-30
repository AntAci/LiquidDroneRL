"""
Run the trained PPO agent in DroneWindEnv and stream 3D state to the browser.

Usage:
  # Start the bridge first:
  python demo/bridge.py
  # Open demo/renderer_3d.html in Chrome
  # Then run this streamer (MLP example):
  python demo/run_3d_demo.py --model-path models/mlp_baseline.zip --difficulty 2 --fps 30 --liquid False
  # Or Liquid:
  python demo/run_3d_demo.py --model-path models/liquid_policy.zip --difficulty 3 --fps 30 --liquid True

This script:
  - Loads the PPO policy (with VecNormalize if available)
  - Creates DroneWindEnv(difficulty=...)
  - Steps the env at target FPS, sends JSON packets via WebSocket to ws://localhost:8765
  - Packet includes x, y, z (pseudo-3D), wind, full obs, timestamp, scale (from scene_config.json), and 'liquid' flag

Recording:
  - Use OBS (or browser capture) to record the renderer_3d.html window at 60fps.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Tuple, Optional

import numpy as np
import websockets

# Add project root to import env and models
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env.drone_env import DroneWindEnv  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402
import gymnasium as gym  # noqa: E402


class ObsAdapterWrapper(gym.Wrapper):
    """
    Wrap DroneWindEnv to adapt observation to an older/trained policy format, if needed.
    Current env obs: [x, y, vx, vy, z, vz, wind_x, wind_y]
    Many trained models expect: [x, y, vx, vy, wind_x, wind_y, dx, dy]
    We reconstruct dx, dy from env internal target center.
    """
    def __init__(self, env: DroneWindEnv) -> None:
        super().__init__(env)
        # Emulate the old observation space (8-dim)
        high = np.array([1.0, 1.0, 2.0, 2.0, env.wind_max, env.wind_max, 1.0, 1.0], dtype=np.float32)
        low = -high
        # Clip x,y to [0,1], but keep symmetrical bounds for simplicity
        low[:2] = 0.0
        high[:2] = 1.0
        from gymnasium import spaces
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self._map_obs(obs), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._map_obs(obs), reward, terminated, truncated, info

    def _map_obs(self, obs: np.ndarray) -> np.ndarray:
        x, y, vx, vy, z, vz, wind_x, wind_y = obs.tolist()
        # Compute target center
        tx_c = (self.env.target_x_min + self.env.target_x_max) * 0.5
        ty_c = (self.env.target_y_min + self.env.target_y_max) * 0.5
        dx = float(tx_c - x)
        dy = float(ty_c - y)
        old_obs = np.array([x, y, vx, vy, wind_x, wind_y, dx, dy], dtype=np.float32)
        return old_obs

    def render(self) -> None:
        return self.env.render()


async def ensure_ws(url: str) -> websockets.WebSocketClientProtocol:
    """Retry connecting until the bridge is available."""
    while True:
        try:
            ws = await websockets.connect(url, ping_interval=20, ping_timeout=20)
            return ws
        except Exception:
            print("Waiting for renderer_3d.html to connect… (bridge not ready)")
            await asyncio.sleep(1.0)


def load_scene_scale() -> float:
    cfg_path = os.path.join(ROOT, "demo", "assets", "scene_config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("scale", 10.0))
    except Exception:
        return 10.0


async def stream_loop(args: argparse.Namespace) -> None:
    url = "ws://127.0.0.1:8765"
    print(f"[3D DEMO] Connecting to WebSocket at {url}...")
    ws = await ensure_ws(url)
    print("[3D DEMO] WebSocket connected!")

    # Load model first to check its observation space
    print(f"[3D DEMO] Loading model: {args.model_path}")
    model = PPO.load(args.model_path)
    model_obs_shape = model.observation_space.shape[0] if hasattr(model.observation_space, 'shape') else None
    print(f"[3D DEMO] Model expects observation shape: {model_obs_shape}")
    
    # Create env
    print("[3D DEMO] Creating environment...")
    base_env = DroneWindEnv(difficulty=args.difficulty, enable_pseudo_3d=True)
    env_obs_shape = base_env.observation_space.shape[0]
    print(f"[3D DEMO] Environment observation shape: {env_obs_shape}")
    
    # Use adapter only if model expects 8 dims but env has 10 dims
    use_adapter = (model_obs_shape == 8 and env_obs_shape == 10)
    if use_adapter:
        print("[3D DEMO] Using observation adapter (8-dim model with 10-dim env)")
        env = ObsAdapterWrapper(base_env)
    else:
        print(f"[3D DEMO] Using direct environment (no adapter needed)")
        env = base_env

    vecnorm_path = args.model_path.replace(".zip", "_vecnorm.pkl")
    use_vecnorm = os.path.exists(vecnorm_path)
    if use_vecnorm:
        print(f"[3D DEMO] Loading VecNormalize stats: {vecnorm_path}")
        # Create venv matching the env we'll use (with or without adapter)
        if use_adapter:
            venv = DummyVecEnv([lambda: ObsAdapterWrapper(DroneWindEnv(difficulty=args.difficulty, enable_pseudo_3d=True))])
        else:
            venv = DummyVecEnv([lambda: DroneWindEnv(difficulty=args.difficulty, enable_pseudo_3d=True)])
        try:
            vec_env = VecNormalize.load(vecnorm_path, venv)
            vec_env.training = False
            vec_env.norm_reward = False
            print("[3D DEMO] VecNormalize loaded successfully")
        except AssertionError as e:
            print(f"[3D DEMO] Warning: VecNormalize observation space mismatch: {e}")
            print("[3D DEMO] Continuing without VecNormalize (may affect performance)")
            vec_env = None
    else:
        vec_env = None

    scale = load_scene_scale()
    print("[3D DEMO] Ready. Open demo/renderer_3d.html in a browser.")

    episodes = int(args.episodes)
    fps_dt = 1.0 / float(args.fps)
    print(f"[3D DEMO] Starting {episodes} episode(s) at {args.fps} FPS...")

    for ep in range(episodes):
        print(f"[3D DEMO] Episode {ep + 1}/{episodes}")
        if vec_env is not None:
            vec_obs = vec_env.reset()
            obs = vec_obs[0]
        else:
            obs, _ = env.reset()
        done = False
        truncated = False
        step_count = 0

        while not (done or truncated):
            step_count += 1
            # Get environment reference for debug output
            debug_env = vec_env.venv.envs[0].unwrapped if vec_env is not None else env.unwrapped
            if step_count % 30 == 0:
                print(f"[3D DEMO] Step {step_count}, x={debug_env.x:.2f}, y={debug_env.y:.2f}")
            # Predict action
            if vec_env is not None:
                action, _ = model.predict(obs, deterministic=True)
                vec_obs, rewards, dones, infos = vec_env.step(np.array([action]))
                obs = vec_obs[0]
                done = bool(dones[0])
                truncated = False
            else:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)

            # Read current world state from unwrapped env
            # When using VecNormalize, the actual environment is inside vec_env
            if vec_env is not None:
                e = vec_env.venv.envs[0].unwrapped
            else:
                e = env.unwrapped
            
            # Get reward (from step return or info)
            current_reward = float(reward) if not vec_env else float(rewards[0])
            
            # Check if drone is in target zone
            in_target = False
            if hasattr(e, 'target_x_min') and hasattr(e, 'target_x_max'):
                in_target = (e.target_x_min <= e.x <= e.target_x_max and 
                            e.target_y_min <= e.y <= e.target_y_max)
            
            # Determine policy name
            policy_name = "Liquid NN" if args.liquid else "MLP Baseline"
            
            packet = {
                "x": float(e.x),
                "y": float(e.y),
                "z": float(e.z),
                "vx": float(e.vx),
                "vy": float(e.vy),
                "wind_x": float(e.wind_x),
                "wind_y": float(e.wind_y),
                "step": int(step_count),
                "reward": current_reward,
                "in_target": bool(in_target),
                "target_x_min": float(getattr(e, 'target_x_min', 0.0)),
                "target_x_max": float(getattr(e, 'target_x_max', 0.0)),
                "target_y_min": float(getattr(e, 'target_y_min', 0.0)),
                "target_y_max": float(getattr(e, 'target_y_max', 0.0)),
                "target_spawned": bool(getattr(e, 'target_spawned', False)),
                "obs": obs.tolist() if isinstance(obs, np.ndarray) else list(obs),
                "timestamp": time.time(),
                "liquid": bool(args.liquid),
                "policy": policy_name,
                "scale": float(scale),
            }
            try:
                await ws.send(json.dumps(packet))
                if step_count == 1:
                    print(f"[3D DEMO] Sent first packet: x={packet['x']:.2f}, y={packet['y']:.2f}, z={packet['z']:.2f}")
            except Exception as e:
                print(f"[3D DEMO] Error sending packet: {e}")
                # Reconnect if bridge dropped
                ws = await ensure_ws(url)
                await ws.send(json.dumps(packet))

            await asyncio.sleep(fps_dt)

        # Reset for next episode


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run real-time 3D demo streaming to browser.")
    ap.add_argument("--model-path", type=str, required=True, help="Path to trained PPO .zip")
    ap.add_argument("--difficulty", type=int, default=2, help="Env difficulty level (0-5)")
    ap.add_argument("--fps", type=int, default=30, help="Streaming FPS")
    ap.add_argument("--liquid", type=lambda v: str(v).lower() in ("1","true","yes","y","t"), default=False, help="Set True if model is Liquid policy")
    ap.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(stream_loop(args))
    except KeyboardInterrupt:
        pass


