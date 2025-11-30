"""
Train a PPO agent with MLP policy on the DroneWindEnv environment.

This script uses stable-baselines3 PPO with a 2-layer MLP (64, 64) to train
an agent to survive and navigate in the 2D drone environment with wind.
The trained model is saved to models/mlp_baseline.zip and TensorBoard logs
are written to logs/ppo_mlp/.
"""

import os
import sys
import argparse
from typing import Optional
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.drone_env import DroneWindEnv


class RandomDifficultyWrapper(gym.Wrapper):
    """Wrap an env to randomize difficulty at every reset."""
    def __init__(self, env: gym.Env, difficulties: Optional[list[int]] = None):
        super().__init__(env)
        self.difficulties = difficulties or [0, 1, 2, 3, 4]
    
    def reset(self, **kwargs):
        d = int(np.random.choice(self.difficulties))
        if hasattr(self.env, "difficulty"):
            self.env.difficulty = d
            if hasattr(self.env, "_configure_difficulty"):
                self.env._configure_difficulty()
        return self.env.reset(**kwargs)


class CurriculumDifficultyWrapper(gym.Wrapper):
    """Wrap an env to progressively increase difficulty during training."""
    def __init__(self, env: gym.Env, min_difficulty: int = 0, max_difficulty: int = 4):
        super().__init__(env)
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.current_difficulty = min_difficulty
        # Shared difficulty state (will be updated by callback)
        self._shared_difficulty = [min_difficulty]  # Use list for mutable shared state
    
    def reset(self, **kwargs):
        # Use shared difficulty level (updated by curriculum callback)
        if hasattr(self.env, "difficulty"):
            self.env.difficulty = self._shared_difficulty[0]
            self.current_difficulty = self._shared_difficulty[0]
            if hasattr(self.env, "_configure_difficulty"):
                self.env._configure_difficulty()
        return self.env.reset(**kwargs)
    
    def set_difficulty(self, difficulty: int) -> None:
        """Set the difficulty level (called by curriculum callback)."""
        difficulty = int(np.clip(difficulty, self.min_difficulty, self.max_difficulty))
        self._shared_difficulty[0] = difficulty
        self.current_difficulty = difficulty
    
    def get_current_difficulty(self) -> int:
        """Get the current difficulty level."""
        return self._shared_difficulty[0]


def make_env(seed: Optional[int] = None, random_difficulty: bool = False, curriculum_difficulty: bool = False,
             difficulties: Optional[list[int]] = None, min_difficulty: int = 0, max_difficulty: int = 4) -> gym.Env:
    """
    Create and wrap a DroneWindEnv instance with Monitor.
    
    Args:
        seed: Optional random seed for the environment
        random_difficulty: If True, resample difficulty randomly on each reset
        curriculum_difficulty: If True, progressively increase difficulty during training
        difficulties: Difficulty levels to sample from
        min_difficulty: Minimum difficulty for curriculum learning
        max_difficulty: Maximum difficulty for curriculum learning
        
    Returns:
        Wrapped Gymnasium environment
    """
    base_difficulty = max_difficulty if (random_difficulty or curriculum_difficulty) else 2
    env = DroneWindEnv(difficulty=base_difficulty)
    if curriculum_difficulty:
        env = CurriculumDifficultyWrapper(env, min_difficulty=min_difficulty, max_difficulty=max_difficulty)
    elif random_difficulty:
        env = RandomDifficultyWrapper(env, difficulties=difficulties)
    env = Monitor(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_vec_env(num_envs: int = 4, random_difficulty: bool = False, curriculum_difficulty: bool = False,
                 difficulties: Optional[list[int]] = None, min_difficulty: int = 0, max_difficulty: int = 4) -> DummyVecEnv:
    """
    Create a vectorized environment with multiple parallel instances.
    
    Args:
        num_envs: Number of parallel environments
        random_difficulty: Whether to randomize difficulty per reset
        curriculum_difficulty: Whether to progressively increase difficulty during training
        difficulties: Difficulty levels to sample from
        min_difficulty: Minimum difficulty for curriculum learning
        max_difficulty: Maximum difficulty for curriculum learning
        
    Returns:
        Vectorized environment
    """
    def make_vec_env_fn(seed: Optional[int] = None):
        def _init():
            return make_env(seed, random_difficulty=random_difficulty, curriculum_difficulty=curriculum_difficulty,
                          difficulties=difficulties, min_difficulty=min_difficulty, max_difficulty=max_difficulty)
        return _init
    
    vec_env = DummyVecEnv([make_vec_env_fn(seed=i) for i in range(num_envs)])
    return vec_env


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PPO agent on DroneWindEnv")
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
        default="logs/ppo_mlp",
        help="Directory for TensorBoard logs (default: logs/ppo_mlp)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/mlp_baseline.zip",
        help="Path to save the trained model (default: models/mlp_baseline.zip)"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)"
    )
    parser.add_argument(
        "--random-difficulty",
        action="store_true",
        help="Sample difficulty randomly at each episode reset (default: False)"
    )
    parser.add_argument(
        "--curriculum-difficulty",
        action="store_true",
        help="Progressively increase difficulty from min to max during training (curriculum learning, default: False)"
    )
    parser.add_argument(
        "--difficulties",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated difficulty levels to sample when randomizing (default: 0,1,2,3,4)"
    )
    parser.add_argument(
        "--min-difficulty",
        type=int,
        default=0,
        help="Minimum difficulty for curriculum learning (default: 0)"
    )
    parser.add_argument(
        "--max-difficulty",
        type=int,
        default=4,
        help="Maximum difficulty for curriculum learning (default: 4)"
    )
    
    args = parser.parse_args()
    # Lazy import for numpy used in wrapper
    import numpy as np
    difficulties = [int(x) for x in args.difficulties.split(",") if x.strip() != ""]
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)
    
    print("=" * 60)
    print("Training PPO Agent on DroneWindEnv")
    print("=" * 60)
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Number of parallel environments: {args.num_envs}")
    print(f"Model will be saved to: {args.model_path}")
    print(f"TensorBoard logs: {args.logdir}")
    if args.curriculum_difficulty:
        print(f"Curriculum learning enabled: difficulty {args.min_difficulty} → {args.max_difficulty}")
    elif args.random_difficulty:
        print(f"Random difficulty enabled over: {difficulties}")
    else:
        print(f"Fixed difficulty: 2 (medium)")
    print("=" * 60)
    
    # Create vectorized environment
    print("Creating vectorized environment...")
    vec_env = make_vec_env(
        num_envs=args.num_envs, 
        random_difficulty=args.random_difficulty,
        curriculum_difficulty=args.curriculum_difficulty,
        difficulties=difficulties,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty
    )
    # Normalize observations and rewards for parity with Liquid setup
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    
    # Configure policy (2-layer MLP with 64 hidden units each)
    policy_kwargs = dict(net_arch=[64, 64])
    
    # Create PPO agent
    print("Initializing PPO agent...")
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
    
    # Setup curriculum callback if enabled
    callbacks = []
    if args.curriculum_difficulty:
        class CurriculumCallback(BaseCallback):
            """Callback to progressively increase difficulty during training."""
            def __init__(self, vec_env, min_difficulty: int = 0, max_difficulty: int = 4, total_timesteps: int = 100000):
                super().__init__()
                self.vec_env = vec_env
                self.min_difficulty = min_difficulty
                self.max_difficulty = max_difficulty
                self.total_timesteps = total_timesteps
                self.num_levels = max_difficulty - min_difficulty + 1
                self.current_level = -1  # Start at -1 so first update triggers
            
            def _on_step(self) -> bool:
                # Calculate which difficulty level we should be at based on progress
                progress = self.num_timesteps / self.total_timesteps
                target_level = int(progress * self.num_levels)
                target_level = min(target_level, self.num_levels - 1)
                target_difficulty = self.min_difficulty + target_level
                
                # Update difficulty if it changed
                if target_level != self.current_level:
                    self.current_level = target_level
                    # Update all wrapped environments
                    # VecNormalize wraps the base vec_env, so we need to access .venv
                    base_vec_env = self.vec_env.venv if hasattr(self.vec_env, 'venv') else self.vec_env
                    for env_idx in range(base_vec_env.num_envs):
                        env = base_vec_env.envs[env_idx]
                        # Unwrap to find CurriculumDifficultyWrapper
                        while hasattr(env, 'env'):
                            if isinstance(env, CurriculumDifficultyWrapper):
                                env.set_difficulty(target_difficulty)
                                break
                            env = env.env
                    print(f"[Curriculum] Progress: {progress*100:.1f}% | Difficulty: {target_difficulty} (level {target_level+1}/{self.num_levels})")
                    if hasattr(self, 'logger') and self.logger is not None:
                        self.logger.record("curriculum/current_difficulty", float(target_difficulty))
                return True
        
        callbacks.append(CurriculumCallback(vec_env, min_difficulty=args.min_difficulty, 
                                          max_difficulty=args.max_difficulty, total_timesteps=args.timesteps))
    
    # Train the agent
    print("\nStarting training...")
    callback = CallbackList(callbacks) if callbacks else None
    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=True,
        callback=callback
    )
    
    # Save the model
    print(f"\nSaving model to {args.model_path}...")
    model.save(args.model_path)
    vecnorm_path = args.model_path.replace(".zip", "_vecnorm.pkl")
    print(f"Saving VecNormalize stats to {vecnorm_path}...")
    vec_env.save(vecnorm_path)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Model saved to: {args.model_path}")
    print(f"VecNormalize stats saved to: {vecnorm_path}")
    print(f"TensorBoard logs available at: {args.logdir}")
    print("=" * 60)
    print("\nTo view training progress, run:")
    print(f"  tensorboard --logdir {args.logdir}")
    print("\nTo evaluate the model, run:")
    print(f"  python eval/eval_mlp_baseline.py --model-path {args.model_path}")


if __name__ == "__main__":
    main()

