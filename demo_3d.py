import numpy as np
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from env.drone_3d import Drone3DEnv
from models.liquid_ppo import make_liquid_ppo
from stable_baselines3 import PPO

def run_demo():
    print("Initializing Project Neuro-Flyt 3D Demo (Matplotlib Mode)...")
    
    env = Drone3DEnv(render_mode="human", wind_scale=5.0, wind_speed=2.0)
    
    model_path = "liquid_ppo_drone_final.zip"
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}...")
        model = PPO.load(model_path, env=env)
    else:
        print("No trained model found. Using untrained Liquid Brain.")
        model = make_liquid_ppo(env, verbose=1)
    
    print("\n=== DEMO STARTING ===")
    
    obs, info = env.reset()
    
    # Setup Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    from matplotlib.animation import FuncAnimation
    
    def update(frame):
        nonlocal obs, info
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        
        ax.clear()
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_zlim(0, 20)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Neuro-Flyt 3D | Step: {frame}')
        
        pos = obs[0:3]
        wind = info.get("wind", np.zeros(3))
        
        # Draw Drone
        ax.scatter(pos[0], pos[1], pos[2], c='blue', s=100, label='Drone')
        
        # Draw Wind Vector
        ax.quiver(pos[0], pos[1], pos[2], wind[0], wind[1], wind[2], length=1.0, color='red', label='Wind Force')
        
        # Draw Target
        ax.scatter(0, 0, 10, c='green', marker='x', s=100, label='Target')
        
        ax.legend()
        
        if term or trunc:
            obs, info = env.reset()
            
    print("Generating Animation (demo.gif)...")
    anim = FuncAnimation(fig, update, frames=200, interval=50)
    anim.save('demo.gif', writer='pillow', fps=20)
    print("Animation saved to demo.gif")
    
    env.close()

if __name__ == "__main__":
    run_demo()
