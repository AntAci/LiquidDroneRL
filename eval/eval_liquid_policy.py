"""
Evaluate a trained PPO Liquid Neural Network policy on the DroneWindEnv environment.

This script loads a saved PPO model with liquid policy and runs evaluation episodes,
printing statistics about average reward and episode length.
"""

import os
import sys
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.drone_env import DroneWindEnv


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate PPO Liquid NN agent on DroneWindEnv")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/liquid_policy.zip",
        help="Path to the trained model (default: models/liquid_policy.zip)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Print environment state to console during evaluation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for evaluation (default: None)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Evaluating PPO Liquid NN Agent on DroneWindEnv")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Number of episodes: {args.episodes}")
    print("=" * 60)
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"\nError: Model file not found at {args.model_path}")
        print("Please train a model first using:")
        print("  python train/train_liquid_ppo.py")
        return
    
    # Create environment (vectorized) and load VecNormalize stats if available
    print("\nCreating environment...")
    def _make_env():
        return DroneWindEnv()
    venv = DummyVecEnv([_make_env])
    vecnorm_path = args.model_path.replace(".zip", "_vecnorm.pkl")
    if os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize stats from {vecnorm_path}...")
        vec_env = VecNormalize.load(vecnorm_path, venv)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print("VecNormalize stats not found; using unnormalized env (may degrade performance).")
        vec_env = venv
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    try:
        model = PPO.load(args.model_path, env=vec_env)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"\nError loading model: {e}")
        return
    
    # Run evaluation episodes
    print(f"\nRunning {args.episodes} evaluation episodes...")
    print("-" * 60)
    
    rewards = []
    episode_lengths = []
    
    # Target zone metrics
    target_time_steps = []  # Steps spent in target
    target_time_percent = []  # Percentage of episode in target
    target_entries = []  # Number of times entering target
    time_to_first_target = []  # Steps until first target entry
    ever_reached_target = []  # Whether target was ever reached
    avg_distance_to_target = []  # Average distance to target center
    min_distance_to_target = []  # Minimum distance to target center
    max_distance_to_target = []  # Maximum distance to target center
    # Effort/Thrust metrics
    avg_effort_per_step = []
    avg_thrust_per_step = []
    
    for episode in range(args.episodes):
        obs = vec_env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        
        # Target tracking for this episode
        steps_in_target = 0
        target_entry_count = 0
        first_target_step = None
        was_in_target_prev = False
        distances_to_target = []
        
        if args.render:
            print(f"\nEpisode {episode + 1}:")
        
        effort_sum = 0.0
        thrust_sum = 0.0
        while not done:
            # Get action from the model (deterministic)
            action, _ = model.predict(obs, deterministic=True)
            # Step vectorized env
            obs, rew_vec, done_vec, infos = vec_env.step(action)
            info = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else {}
            total_reward += float(rew_vec[0])
            step_count += 1
            done = bool(done_vec[0])
            
            # Track target zone and effort metrics
            in_target = info.get("in_target", False)
            target_spawned = info.get("target_spawned", False)
            effort = float(info.get("effort", 0.0) or 0.0)
            thrust_applied = float(info.get("thrust_applied", 0.0) or 0.0)
            # Accumulate per-episode averages (running mean using sums)
            effort_sum += effort
            thrust_sum += thrust_applied
            
            if target_spawned:
                # Use provided distance metric if available
                dist = info.get("distance_to_target", None)
                if dist is not None:
                    distances_to_target.append(float(dist))
                
                # Track time in target
                if in_target:
                    steps_in_target += 1
                    if not was_in_target_prev:
                        target_entry_count += 1
                        if first_target_step is None:
                            first_target_step = step_count
                    was_in_target_prev = True
                else:
                    was_in_target_prev = False
            
            if args.render:
                pass
        
        rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        # Store target metrics
        target_time_steps.append(steps_in_target)
        target_time_percent.append((steps_in_target / step_count * 100) if step_count > 0 else 0)
        target_entries.append(target_entry_count)
        time_to_first_target.append(first_target_step if first_target_step else step_count)
        ever_reached_target.append(1 if first_target_step is not None else 0)
        
        if distances_to_target:
            avg_distance_to_target.append(np.mean(distances_to_target))
            min_distance_to_target.append(np.min(distances_to_target))
            max_distance_to_target.append(np.max(distances_to_target))
        else:
            avg_distance_to_target.append(np.nan)
            min_distance_to_target.append(np.nan)
            max_distance_to_target.append(np.nan)
        
        # Store effort/thrust metrics
        avg_effort_per_step.append((effort_sum / step_count) if step_count > 0 else 0.0)
        avg_thrust_per_step.append((thrust_sum / step_count) if step_count > 0 else 0.0)
        
        status = "done" if done else "running"
        target_info = f", Target: {steps_in_target}/{step_count} steps ({steps_in_target/step_count*100:.1f}%)" if step_count > 0 else ""
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, "
              f"Length = {step_count} steps ({status}){target_info}")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Evaluation Results - General Performance")
    print("=" * 70)
    
    # Effort/Thrust statistics
    print("\n" + "=" * 70)
    print("Control Effort & Thrust Metrics")
    print("=" * 70)
    print(f"Average effort per step: {np.mean(avg_effort_per_step):.4f} ± {np.std(avg_effort_per_step):.4f}")
    print(f"Average thrust magnitude per step: {np.mean(avg_thrust_per_step):.4f} ± {np.std(avg_thrust_per_step):.4f}")
    print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average survival time: {np.mean(episode_lengths):.1f} steps")
    print(f"Min reward: {np.min(rewards):.2f}")
    print(f"Max reward: {np.max(rewards):.2f}")
    print(f"Min episode length: {np.min(episode_lengths)}")
    print(f"Max episode length: {np.max(episode_lengths)}")
    
    print("\n" + "=" * 70)
    print("Target Zone Performance Metrics")
    print("=" * 70)
    
    # Target reach statistics
    episodes_reached = np.sum(ever_reached_target)
    reach_rate = (episodes_reached / args.episodes) * 100
    print(f"Target reach rate: {reach_rate:.1f}% ({episodes_reached}/{args.episodes} episodes)")
    
    # Time in target
    valid_target_time = [t for t in target_time_steps if t > 0]
    if valid_target_time:
        print(f"Average time in target: {np.mean(valid_target_time):.1f} ± {np.std(valid_target_time):.1f} steps")
        print(f"Max time in target: {np.max(valid_target_time)} steps")
    else:
        print("Average time in target: 0.0 steps (never reached)")
    
    print(f"Average % of episode in target: {np.mean(target_time_percent):.1f}% ± {np.std(target_time_percent):.1f}%")
    
    # Target entries
    print(f"Average target entries per episode: {np.mean(target_entries):.2f} ± {np.std(target_entries):.2f}")
    print(f"Max target entries in one episode: {np.max(target_entries)}")
    
    # Time to first target
    valid_first_target = [time_to_first_target[i] for i in range(len(time_to_first_target)) 
                          if ever_reached_target[i] and time_to_first_target[i] < episode_lengths[i]]
    if valid_first_target:
        print(f"Average steps to first target entry: {np.mean(valid_first_target):.1f} ± {np.std(valid_first_target):.1f}")
        print(f"Fastest target entry: {np.min(valid_first_target)} steps")
    else:
        print("Average steps to first target entry: N/A (never reached)")
    
    # Distance metrics
    valid_distances = [d for d in avg_distance_to_target if not np.isnan(d)]
    if valid_distances:
        print(f"\nDistance to Target Center:")
        print(f"  Average distance: {np.mean(valid_distances):.3f} ± {np.std(valid_distances):.3f}")
        print(f"  Minimum distance: {np.mean(min_distance_to_target):.3f}")
        print(f"  Maximum distance: {np.mean(max_distance_to_target):.3f}")
    else:
        print("\nDistance to Target Center: N/A (target never spawned)")
    
    print("=" * 70)
    
    # Print detailed per-episode breakdown
    print("\nPer-Episode Detailed Breakdown:")
    print("-" * 70)
    for i in range(args.episodes):
        reached = "✓" if ever_reached_target[i] else "✗"
        print(f"Episode {i+1}: Reward={rewards[i]:.2f}, Length={episode_lengths[i]}, "
              f"Target={target_time_steps[i]}/{episode_lengths[i]} steps ({target_time_percent[i]:.1f}%), "
              f"Entries={target_entries[i]}, Reached={reached}")
        if not np.isnan(avg_distance_to_target[i]):
            print(f"  Avg distance to target: {avg_distance_to_target[i]:.3f}, "
                  f"Min: {min_distance_to_target[i]:.3f}, Max: {max_distance_to_target[i]:.3f}")
    
    # Optional: Try to plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        
        # Plot 1: Episode rewards
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(rewards) + 1), rewards, 'o-', linewidth=2, markersize=6)
        plt.axhline(y=np.mean(rewards), color='r', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode Rewards (Liquid NN)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Episode lengths
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(episode_lengths) + 1), episode_lengths, 's-', 
                linewidth=2, markersize=6, color='green')
        plt.axhline(y=np.mean(episode_lengths), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(episode_lengths):.1f}')
        plt.xlabel('Episode')
        plt.ylabel('Episode Length')
        plt.title('Episode Lengths (Liquid NN)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('eval_liquid_results.png', dpi=150, bbox_inches='tight')
        print("\n✓ Evaluation plots saved to eval_liquid_results.png")
        print("  (Close the plot window to continue)")
        plt.show(block=False)
        plt.pause(2)  # Show for 2 seconds
        plt.close()
        
    except ImportError:
        # Matplotlib not available, skip plotting
        pass
    except Exception as e:
        print(f"\nNote: Could not generate plots: {e}")


if __name__ == "__main__":
    main()

