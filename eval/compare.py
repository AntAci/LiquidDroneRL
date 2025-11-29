#!/usr/bin/env python3
"""
Compare MLP vs Liquid PPO agents across multiple wind difficulty levels.

Usage:
    python eval/compare.py --episodes 20 --render
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Local imports
import sys
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from env.drone_env import DroneWindEnv  # noqa: E402


def evaluate_agent(
    model: PPO,
    difficulty: int,
    episodes: int,
    render: bool = False,
    use_vecnorm: bool = False,
    vecnorm_path: str | None = None,
) -> Tuple[float, float, float, float]:
    """
    Evaluate a single agent for a number of episodes at a given difficulty.
    
    Args:
        model: The PPO model to evaluate
        difficulty: Wind difficulty level (0-4)
        episodes: Number of episodes to run
        render: Whether to render the environment
        use_vecnorm: Whether to use VecNormalize for observations
        vecnorm_path: Path to VecNormalize stats file (if use_vecnorm is True)
    
    Returns:
        (mean_reward, mean_survival_time, std_reward, std_survival_time)
    """
    total_rewards: List[float] = []
    survival_times: List[int] = []

    # Setup environment with VecNormalize if needed
    if use_vecnorm and vecnorm_path and os.path.exists(vecnorm_path):
        def _make_env():
            return DroneWindEnv(difficulty=difficulty)
        venv = DummyVecEnv([_make_env])
        vec_env = VecNormalize.load(vecnorm_path, venv)
        vec_env.training = False
        vec_env.norm_reward = False
        env = vec_env.envs[0].unwrapped if hasattr(vec_env.envs[0], 'unwrapped') else vec_env.envs[0]
    else:
        vec_env = None
        env = DroneWindEnv(difficulty=difficulty)

    for _ in range(episodes):
        if vec_env is not None:
            obs = vec_env.reset()
            obs = obs[0]  # Unwrap from vectorized format
        else:
            obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        steps = 0

        while not done and not truncated:
            # Protect against invalid observations
            if not np.all(np.isfinite(obs)):
                print(f"[WARNING] Model failed at difficulty {difficulty} (NaNs detected in observation).")
                break
            action, _ = model.predict(obs, deterministic=True)
            # Guard against invalid actions
            if isinstance(action, np.ndarray) and not np.all(np.isfinite(action)):
                print(f"[WARNING] Model produced NaN action at difficulty {difficulty}; replacing with zeros.")
                action = np.zeros_like(action)
            
            if vec_env is not None:
                obs, rewards, dones, infos = vec_env.step(np.array([action]))
                obs = obs[0]
                reward = rewards[0]
                # VecEnv returns done (which is terminated or truncated)
                done = dones[0]
                truncated = False  # VecEnv doesn't separate terminated/truncated
                info = infos[0] if infos else {}
            else:
                obs, reward, done, truncated, info = env.step(action)
            
            # Reward safety
            if not np.isfinite(reward):
                print(f"[WARNING] Model failed at difficulty {difficulty} (NaNs detected in reward).")
                reward = 0.0
                done = True
            ep_reward += float(reward)
            steps += 1
            if render:
                env.render()

        total_rewards.append(ep_reward)
        survival_times.append(steps)

    # Aggregate
    rewards_arr = np.array(total_rewards, dtype=np.float64)
    times_arr = np.array(survival_times, dtype=np.float64)
    return (
        float(np.nanmean(rewards_arr)),
        float(np.nanmean(times_arr)),
        float(np.nanstd(rewards_arr)),
        float(np.nanstd(times_arr)),
    )


def main() -> None:
    """
    Load MLP and Liquid PPO models and compare across wind difficulty levels.
    Produces bar charts and prints a summary table.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--mlp-path", type=str, default="models/mlp_baseline.zip")
    parser.add_argument("--liquid-path", type=str, default="models/liquid_policy.zip")
    parser.add_argument(
        "--difficulties",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated difficulty levels to evaluate (default: 0,1,2,3,4)"
    )
    args = parser.parse_args()

    # Ensure results dir exists
    os.makedirs("results", exist_ok=True)

    # Load models
    try:
        mlp_model = PPO.load(args.mlp_path)
    except Exception as e:
        print(f"[ERROR] Failed to load MLP model at '{args.mlp_path}': {e}")
        return
    try:
        liquid_model = PPO.load(args.liquid_path)
    except Exception as e:
        print(f"[ERROR] Failed to load Liquid model at '{args.liquid_path}': {e}")
        return

    try:
        difficulties = [int(x) for x in args.difficulties.split(",") if x.strip() != ""]
    except Exception:
        difficulties = [0, 1, 2, 3, 4]

    mlp_rewards: List[float] = []
    mlp_survival: List[float] = []
    mlp_rewards_std: List[float] = []
    mlp_survival_std: List[float] = []

    lnn_rewards: List[float] = []
    lnn_survival: List[float] = []
    lnn_rewards_std: List[float] = []
    lnn_survival_std: List[float] = []

    # Check for VecNormalize stats for MLP model
    mlp_vecnorm_path = args.mlp_path.replace(".zip", "_vecnorm.pkl")
    use_mlp_vecnorm = os.path.exists(mlp_vecnorm_path)
    if use_mlp_vecnorm:
        print(f"Using VecNormalize for MLP model (stats from {mlp_vecnorm_path})")
    else:
        print("MLP model: VecNormalize stats not found, using raw observations")

    # Check for VecNormalize stats for liquid model
    liquid_vecnorm_path = args.liquid_path.replace(".zip", "_vecnorm.pkl")
    use_liquid_vecnorm = os.path.exists(liquid_vecnorm_path)
    if use_liquid_vecnorm:
        print(f"Using VecNormalize for Liquid model (stats from {liquid_vecnorm_path})")
    else:
        print("Liquid model: VecNormalize stats not found, using raw observations")
    
    print("Running evaluation across wind difficulties...")
    for d in difficulties:
        r_mlp, t_mlp, rstd_mlp, tstd_mlp = evaluate_agent(
            mlp_model, d, args.episodes, render=args.render,
            use_vecnorm=use_mlp_vecnorm, vecnorm_path=mlp_vecnorm_path if use_mlp_vecnorm else None
        )
        r_lnn, t_lnn, rstd_lnn, tstd_lnn = evaluate_agent(
            liquid_model, d, args.episodes, render=args.render,
            use_vecnorm=use_liquid_vecnorm, vecnorm_path=liquid_vecnorm_path if use_liquid_vecnorm else None
        )

        mlp_rewards.append(r_mlp)
        mlp_survival.append(t_mlp)
        mlp_rewards_std.append(rstd_mlp)
        mlp_survival_std.append(tstd_mlp)

        lnn_rewards.append(r_lnn)
        lnn_survival.append(t_lnn)
        lnn_rewards_std.append(rstd_lnn)
        lnn_survival_std.append(tstd_lnn)

    # Plot Survival Time comparison
    x = np.arange(len(difficulties))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, mlp_survival, width, label="MLP", yerr=mlp_survival_std, capsize=4)
    ax.bar(x + width/2, lnn_survival, width, label="Liquid", yerr=lnn_survival_std, capsize=4)
    ax.set_xlabel("Wind Difficulty")
    ax.set_ylabel("Average Survival Time (steps)")
    ax.set_title("Survival Time vs Wind Difficulty")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in difficulties])
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/survival_comparison.png", dpi=150)
    plt.close(fig)

    # Plot Average Reward comparison
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(x - width/2, mlp_rewards, width, label="MLP", yerr=mlp_rewards_std, capsize=4)
    ax2.bar(x + width/2, lnn_rewards, width, label="Liquid", yerr=lnn_rewards_std, capsize=4)
    ax2.set_xlabel("Wind Difficulty")
    ax2.set_ylabel("Average Reward")
    ax2.set_title("Average Reward vs Wind Difficulty")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(d) for d in difficulties])
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("results/reward_comparison.png", dpi=150)
    plt.close(fig2)

    # Print summary table
    print()
    print("Difficulty | MLP Survival | Liquid Survival | MLP Reward | Liquid Reward")
    print("-----------|--------------|-----------------|------------|----------------")
    for i, d in enumerate(difficulties):
        print(
            f"{d:<10} | "
            f"{mlp_survival[i]:<12.2f} | "
            f"{lnn_survival[i]:<15.2f} | "
            f"{mlp_rewards[i]:<10.2f} | "
            f"{lnn_rewards[i]:<14.2f}"
        )

    # Note for maintainers:
    # If your env/drone_env.py did not yet accept a `difficulty` parameter,
    # ensure it is patched to include:
    #   - __init__(self, difficulty: int = 2) storing self.difficulty
    #   - Mapping wind bounds per difficulty:
    #       0: WIND_MAX = 0.0
    #       1: WIND_MAX = 0.5
    #       2: WIND_MAX = 1.0
    #       3: WIND_MAX = 2.0 (add turbulence)
    #       4: WIND_MAX = 2.5 (add turbulence + 1% gusts)
    #   - In _update_wind, add turbulence for difficulty >=3 and gusts for 4:
    #       if self.difficulty >= 4 and self.np_random.random() < 0.01:
    #           self.wind_x += self.np_random.uniform(-2*self.wind_max, 2*self.wind_max)
    #           self.wind_y += self.np_random.uniform(-2*self.wind_max, 2*self.wind_max)


if __name__ == "__main__":
    main()


