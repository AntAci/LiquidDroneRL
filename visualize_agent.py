import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO
from env.drone_3d import Drone3DEnv
import os

def run_visualization(model_path="liquid_ppo_drone_final.zip"):
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Could not find {model_path}")
        print("Please download the file from your Hugging Face Space 'Files' tab")
        print("and place it in this directory.")
        return

    # Create Environment
    env = Drone3DEnv()
    
    # Load Model
    # We need to force cpu load since we trained on cpu (or to be safe)
    model = PPO.load(model_path, env=env, device='cpu')
    
    obs, _ = env.reset()
    frames = []
    
    print("Running simulation...")
    for i in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Capture state for rendering
        # State: [x, y, z, ...]
        pos = env.state[0:3]
        target = env.target
        
        frames.append((pos.copy(), target.copy()))
        
        if terminated or truncated:
            obs, _ = env.reset()

    print(f"Simulation complete. Generating GIF from {len(frames)} frames...")
    
    # Create Animation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame_idx):
        ax.clear()
        pos, target = frames[frame_idx]
        
        # Set bounds
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(0, 20)
        
        # Plot Target
        ax.scatter(target[0], target[1], target[2], c='red', marker='x', s=100, label='Target')
        
        # Plot Drone
        ax.scatter(pos[0], pos[1], pos[2], c='blue', marker='o', s=50, label='Drone')
        
        # Plot "Ground"
        xx, yy = np.meshgrid(range(-10, 11, 5), range(-10, 11, 5))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='green')
        
        ax.set_title(f"Neuro-Flyt 3D - Frame {frame_idx}")
        ax.legend()

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50)
    
    output_file = "drone_flight.gif"
    ani.save(output_file, writer='pillow')
    print(f"✅ GIF saved to: {output_file}")
    print("Open this file to see your drone fly!")

if __name__ == "__main__":
    # Check if user provided a path arg, else default
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "liquid_ppo_drone_final.zip"
    run_visualization(path)
