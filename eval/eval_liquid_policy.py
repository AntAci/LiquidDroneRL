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
    
    # Create environment
    print("\nCreating environment...")
    env = DroneWindEnv()
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    try:
        model = PPO.load(args.model_path, env=env)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"\nError loading model: {e}")
        return
    
    # Run evaluation episodes
    print(f"\nRunning {args.episodes} evaluation episodes...")
    print("-" * 60)
    
    rewards = []
    episode_lengths = []
    
    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed)
        done = False
        truncated = False
        total_reward = 0.0
        step_count = 0
        
        if args.render:
            print(f"\nEpisode {episode + 1}:")
            env.render()
        
        while not (done or truncated):
            # Get action from the model (deterministic)
            action, _ = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            if args.render:
                env.render()
        
        rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        status = "terminated" if done else "truncated"
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, "
              f"Length = {step_count} steps ({status})")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average survival time: {np.mean(episode_lengths):.1f} steps")
    print(f"Min reward: {np.min(rewards):.2f}")
    print(f"Max reward: {np.max(rewards):.2f}")
    print(f"Min episode length: {np.min(episode_lengths)}")
    print(f"Max episode length: {np.max(episode_lengths)}")
    print("=" * 60)
    
    # Print per-episode rewards
    print("\nPer-episode rewards:")
    for i, reward in enumerate(rewards, 1):
        print(f"  Episode {i}: {reward:.2f}")
    
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

