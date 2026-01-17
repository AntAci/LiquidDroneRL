#!/usr/bin/env python3
"""
Create a presentation-quality comparison graph for Liquid vs MLP models.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from stable_baselines3 import PPOaasasas
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.drone_env import DroneWindEnv
from eval.compare import evaluate_agent
print"test")

def create_presentation_graph(episodes=30):
    """Create a polished comparison graph for presentations."""
    
    # Ensure results dir exists
    os.makedirs("results", exist_ok=True)
    
    # Model paths
    mlp_path = "models/mlp_baseline.zip"
    liquid_path = "models/liquid_policy.zip"
    mlp_vecnorm_path = mlp_path.replace(".zip", "_vecnorm.pkl")
    liquid_vecnorm_path = liquid_path.replace(".zip", "_vecnorm.pkl")
    
    # Load models
    print("Loading models...")
    mlp_model = PPO.load(mlp_path)
    liquid_model = PPO.load(liquid_path)
    
    use_mlp_vecnorm = os.path.exists(mlp_vecnorm_path)
    use_liquid_vecnorm = os.path.exists(liquid_vecnorm_path)
    
    difficulties = [0, 1, 2, 3, 4]
    difficulty_labels = ["No Wind", "Mild", "Medium", "Chaotic", "Extreme"]
    
    mlp_rewards = []
    mlp_survival = []
    mlp_rewards_std = []
    mlp_survival_std = []
    
    lnn_rewards = []
    lnn_survival = []
    lnn_rewards_std = []
    lnn_survival_std = []
    
    print(f"Evaluating models across {len(difficulties)} difficulty levels ({episodes} episodes each)...")
    for d in difficulties:
        print(f"  Difficulty {d} ({difficulty_labels[d]})...")
        r_mlp, t_mlp, rstd_mlp, tstd_mlp = evaluate_agent(
            mlp_model, d, episodes, render=False,
            use_vecnorm=use_mlp_vecnorm, vecnorm_path=mlp_vecnorm_path if use_mlp_vecnorm else None
        )
        r_lnn, t_lnn, rstd_lnn, tstd_lnn = evaluate_agent(
            liquid_model, d, episodes, render=False,
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
    
    # Set up matplotlib style for presentations
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Liquid Neural Network vs MLP Baseline Performance Comparison', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    x = np.arange(len(difficulties))
    width = 0.35
    
    # Color scheme - professional and distinct
    mlp_color = '#2E86AB'  # Blue
    liquid_color = '#A23B72'  # Purple/Magenta
    
    # Plot 1: Survival Time
    bars1_mlp = ax1.bar(x - width/2, mlp_survival, width, label='MLP Baseline', 
                        color=mlp_color, alpha=0.85, edgecolor='black', linewidth=1.5,
                        yerr=mlp_survival_std, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    bars1_liq = ax1.bar(x + width/2, lnn_survival, width, label='Liquid NN', 
                       color=liquid_color, alpha=0.85, edgecolor='black', linewidth=1.5,
                       yerr=lnn_survival_std, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    
    ax1.set_xlabel('Wind Difficulty Level', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Average Survival Time (steps)', fontsize=13, fontweight='bold')
    ax1.set_title('Survival Time Across Wind Conditions', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(difficulty_labels, fontsize=11)
    ax1.legend(loc='upper right', fontsize=12, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Add value labels on bars
    for bars in [bars1_mlp, bars1_liq]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Average Reward
    bars2_mlp = ax2.bar(x - width/2, mlp_rewards, width, label='MLP Baseline', 
                        color=mlp_color, alpha=0.85, edgecolor='black', linewidth=1.5,
                        yerr=mlp_rewards_std, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    bars2_liq = ax2.bar(x + width/2, lnn_rewards, width, label='Liquid NN', 
                       color=liquid_color, alpha=0.85, edgecolor='black', linewidth=1.5,
                       yerr=lnn_rewards_std, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    
    ax2.set_xlabel('Wind Difficulty Level', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Average Episode Reward', fontsize=13, fontweight='bold')
    ax2.set_title('Episode Reward Across Wind Conditions', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(difficulty_labels, fontsize=11)
    ax2.legend(loc='upper right', fontsize=12, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    # Add value labels on bars
    for bars in [bars2_mlp, bars2_liq]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save high-resolution version
    output_path = "results/liquid_vs_mlp_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\n✓ Presentation graph saved to: {output_path}")
    
    # Also save a combined single-metric version (survival time is usually most important)
    fig2, ax = plt.subplots(figsize=(12, 7))
    bars_mlp = ax.bar(x - width/2, mlp_survival, width, label='MLP Baseline', 
                      color=mlp_color, alpha=0.85, edgecolor='black', linewidth=1.5,
                      yerr=mlp_survival_std, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    bars_liq = ax.bar(x + width/2, lnn_survival, width, label='Liquid Neural Network', 
                     color=liquid_color, alpha=0.85, edgecolor='black', linewidth=1.5,
                     yerr=lnn_survival_std, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    
    ax.set_xlabel('Wind Difficulty Level', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Survival Time (steps)', fontsize=14, fontweight='bold')
    ax.set_title('Liquid Neural Network vs MLP Baseline: Survival Performance', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(difficulty_labels, fontsize=12)
    ax.legend(loc='best', fontsize=13, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels
    for bars in [bars_mlp, bars_liq]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path2 = "results/liquid_vs_mlp_survival.png"
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Survival-focused graph saved to: {output_path2}")
    
    plt.close('all')
    
    # Print summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print(f"{'Difficulty':<12} {'MLP Survival':<15} {'Liquid Survival':<18} {'MLP Reward':<15} {'Liquid Reward':<15}")
    print("-"*70)
    for i, d in enumerate(difficulties):
        print(f"{difficulty_labels[i]:<12} {mlp_survival[i]:<15.1f} {lnn_survival[i]:<18.1f} "
              f"{mlp_rewards[i]:<15.2f} {lnn_rewards[i]:<15.2f}")
    print("="*70)
    
    # Calculate improvements
    print("\nLiquid NN Improvements over MLP:")
    for i, d in enumerate(difficulties):
        surv_improvement = ((lnn_survival[i] - mlp_survival[i]) / mlp_survival[i] * 100) if mlp_survival[i] > 0 else 0
        reward_improvement = ((lnn_rewards[i] - mlp_rewards[i]) / abs(mlp_rewards[i]) * 100) if mlp_rewards[i] != 0 else 0
        print(f"  {difficulty_labels[i]}: {surv_improvement:+.1f}% survival, {reward_improvement:+.1f}% reward")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create presentation-quality comparison graphs")
    parser.add_argument("--episodes", type=int, default=15, help="Number of episodes per difficulty (default: 15)")
    args = parser.parse_args()
    
    create_presentation_graph(episodes=args.episodes)

