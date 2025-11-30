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
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.drone_env import DroneWindEnv
from models.liquid_policy import LiquidFeatureExtractor


class RandomDifficultyWrapper(gym.Wrapper):
    """Wrap an env to randomize difficulty at every reset."""
    def __init__(self, env: gym.Env, difficulties: Optional[list[int]] = None):
        super().__init__(env)
        self.difficulties = difficulties or [0, 1, 2, 3, 4]
    
    def reset(self, **kwargs):
        # Pick a random difficulty uniformly
        d = int(np.random.choice(self.difficulties))
        # Apply to underlying env (assumes DroneWindEnv)
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
        difficulties: List of difficulties to sample from when random_difficulty=True
        min_difficulty: Minimum difficulty for curriculum learning
        max_difficulty: Maximum difficulty for curriculum learning
        
    Returns:
        Wrapped Gymnasium environment
    """
    # Create with max difficulty to ensure observation space safely bounds all winds
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
        random_difficulty: Whether to resample difficulty randomly on reset
        curriculum_difficulty: Whether to progressively increase difficulty during training
        difficulties: List of difficulty levels to sample from
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
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)
    difficulties = [int(x) for x in args.difficulties.split(",") if x.strip() != ""]
    
    print("=" * 60)
    print("Training PPO Agent with Liquid NN on DroneWindEnv")
    print("=" * 60)
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Number of parallel environments: {args.num_envs}")
    print(f"Liquid cell hidden size: {args.hidden_size}")
    print(f"Liquid cell dt: {args.dt}")
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
    # Normalize observations and rewards for PPO stability
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    
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
    
    # Train
    print("\nStarting training...")
    
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
    
    class MetricsCallback(BaseCallback):
        """Custom callback to log environment-level metrics from infos."""
        def __init__(self, log_interval_steps: int = 1000):
            super().__init__()
            self.log_interval_steps = log_interval_steps
            self._reset_sums()
        
        def _reset_sums(self):
            self.sum_effort = 0.0
            self.sum_thrust = 0.0
            self.sum_in_target = 0
            self.sum_spawned = 0
            self.sum_dist = 0.0
            self.count_dist = 0
            self.count_steps = 0
        
        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                if not isinstance(info, dict):
                    continue
                self.count_steps += 1
                self.sum_effort += float(info.get("effort", 0.0) or 0.0)
                self.sum_thrust += float(info.get("thrust_applied", 0.0) or 0.0)
                if info.get("target_spawned", False):
                    self.sum_spawned += 1
                    if info.get("in_target", False):
                        self.sum_in_target += 1
                    dist = info.get("distance_to_target", None)
                    if dist is not None:
                        self.sum_dist += float(dist)
                        self.count_dist += 1
            # Periodic logging
            if self.num_timesteps % self.log_interval_steps == 0 and self.count_steps > 0:
                avg_effort = self.sum_effort / self.count_steps
                avg_thrust = self.sum_thrust / self.count_steps
                pct_in_target = (self.sum_in_target / self.sum_spawned) if self.sum_spawned > 0 else 0.0
                avg_dist = (self.sum_dist / self.count_dist) if self.count_dist > 0 else 0.0
                self.logger.record("custom/avg_effort", avg_effort)
                self.logger.record("custom/avg_thrust_applied", avg_thrust)
                self.logger.record("custom/percent_in_target", pct_in_target)
                self.logger.record("custom/avg_distance_to_target", avg_dist)
                # Reset accumulators
                self._reset_sums()
            return True
    
    # Setup callbacks
    callbacks = [MetricsCallback(log_interval_steps=2000)]
    if args.curriculum_difficulty:
        callbacks.append(CurriculumCallback(vec_env, min_difficulty=args.min_difficulty, 
                                          max_difficulty=args.max_difficulty, total_timesteps=args.timesteps))
    
    # Combine callbacks
    from stable_baselines3.common.callbacks import CallbackList
    callback = CallbackList(callbacks)
    
    model.learn(total_timesteps=args.timesteps, progress_bar=True, callback=callback)
    
    # Save the model and VecNormalize stats
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
    print(f"  python eval/eval_liquid_policy.py --model-path {args.model_path}")


if __name__ == "__main__":
    main()

