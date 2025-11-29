"""
Train a PPO agent with Liquid Neural Network policy on the DroneWindEnv environment.

This script uses stable-baselines3 PPO with a Liquid Neural Network feature extractor
to train an agent to survive and navigate in the 2D drone environment with wind.
The trained model is saved to models/liquid_policy.zip and TensorBoard logs
are written to logs/ppo_liquid/.
"""

import os
import sys
import argparse
from typing import Optional
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.drone_env import DroneWindEnv
from models.liquid_policy import LiquidFeatureExtractor


def make_env(seed: Optional[int] = None) -> gym.Env:
    """
    Create and wrap a DroneWindEnv instance with Monitor.
    
    Args:
        seed: Optional random seed for the environment
        
    Returns:
        Wrapped Gymnasium environment
    """
    env = DroneWindEnv()
    env = Monitor(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_vec_env(num_envs: int = 4) -> DummyVecEnv:
    """
    Create a vectorized environment with multiple parallel instances.
    
    Args:
        num_envs: Number of parallel environments
        
    Returns:
        Vectorized environment
    """
    def make_vec_env_fn(seed: Optional[int] = None):
        def _init():
            return make_env(seed)
        return _init
    
    vec_env = DummyVecEnv([make_vec_env_fn(seed=i) for i in range(num_envs)])
    return vec_env


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PPO agent with Liquid NN on DroneWindEnv")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total number of training timesteps (default: 100000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs/ppo_liquid",
        help="Directory for TensorBoard logs (default: logs/ppo_liquid)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/liquid_policy.zip",
        help="Path to save the trained model (default: models/liquid_policy.zip)"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="Hidden size for liquid cell (default: 32)"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Time step for liquid cell (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)
    
    print("=" * 60)
    print("Training PPO Agent with Liquid NN on DroneWindEnv")
    print("=" * 60)
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Number of parallel environments: {args.num_envs}")
    print(f"Liquid cell hidden size: {args.hidden_size}")
    print(f"Liquid cell dt: {args.dt}")
    print(f"Model will be saved to: {args.model_path}")
    print(f"TensorBoard logs: {args.logdir}")
    print("=" * 60)
    
    # Create vectorized environment
    print("Creating vectorized environment...")
    vec_env = make_vec_env(num_envs=args.num_envs)
    
    # Get observation space for feature extractor
    obs_space = vec_env.observation_space
    
    # Configure policy with liquid feature extractor
    policy_kwargs = dict(
        features_extractor_class=LiquidFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=args.hidden_size,
            hidden_size=args.hidden_size,
            dt=args.dt,
        ),
        net_arch=dict(pi=[64], vf=[64]),  # Policy and value heads with 64 hidden units
    )
    
    # Create PPO agent
    print("Initializing PPO agent with Liquid NN...")
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=args.logdir,
        seed=args.seed,
    )
    
    # Training with curriculum (commented out for now - use fixed mild wind)
    # For curriculum learning, you could do:
    # 
    # # Phase 1: Mild wind (0-30k steps)
    # if args.timesteps > 30000:
    #     print("Training phase 1: Mild wind (0-30k steps)...")
    #     model.learn(total_timesteps=30000, progress_bar=True)
    #     
    #     # Phase 2: Medium wind (30k-60k steps)
    #     if args.timesteps > 60000:
    #         print("Training phase 2: Medium wind (30k-60k steps)...")
    #         # Would need to recreate env with difficulty=1
    #         model.learn(total_timesteps=30000, progress_bar=True, reset_num_timesteps=False)
    #         
    #         # Phase 3: Strong wind (60k+ steps)
    #         if args.timesteps > 60000:
    #             print("Training phase 3: Strong wind (60k+ steps)...")
    #             # Would need to recreate env with difficulty=2
    #             model.learn(total_timesteps=args.timesteps - 60000, progress_bar=True, reset_num_timesteps=False)
    #     else:
    #         model.learn(total_timesteps=args.timesteps - 30000, progress_bar=True, reset_num_timesteps=False)
    # else:
    #     model.learn(total_timesteps=args.timesteps, progress_bar=True)
    
    # For now, train on fixed mild wind
    print("\nStarting training...")
    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=True
    )
    
    # Save the model
    print(f"\nSaving model to {args.model_path}...")
    model.save(args.model_path)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Model saved to: {args.model_path}")
    print(f"TensorBoard logs available at: {args.logdir}")
    print("=" * 60)
    print("\nTo view training progress, run:")
    print(f"  tensorboard --logdir {args.logdir}")
    print("\nTo evaluate the model, run:")
    print(f"  python eval/eval_liquid_policy.py --model-path {args.model_path}")


if __name__ == "__main__":
    main()

