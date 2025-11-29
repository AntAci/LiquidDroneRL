import numpy as np
import time
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from env.drone_3d import Drone3DEnv
from models.liquid_ppo import make_liquid_ppo
from stable_baselines3 import PPO

def run_interactive_demo():
    print("Initializing Interactive Dashboard...")
    
    env = Drone3DEnv(render_mode="human", wind_scale=5.0, wind_speed=2.0)
    
    model_path = "liquid_ppo_drone_final.zip"
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}...")
        model = PPO.load(model_path, env=env)
    else:
        print("No trained model found. Using untrained Liquid Brain.")
        model = make_liquid_ppo(env, verbose=1)
    
    obs, info = env.reset()
    
    # Setup Dashboard
    plt.ion()
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, width_ratios=[2, 1])
    
    # 3D View (Left, spanning both rows)
    ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
    
    # Altitude Plot (Top Right)
    ax_alt = fig.add_subplot(gs[0, 1])
    ax_alt.set_title("Altitude (Z)")
    ax_alt.set_ylim(0, 15)
    ax_alt.set_xlim(0, 100)
    line_alt, = ax_alt.plot([], [], 'b-')
    
    # Wind Speed Plot (Bottom Right)
    ax_wind = fig.add_subplot(gs[1, 1])
    ax_wind.set_title("Wind Magnitude")
    ax_wind.set_ylim(0, 10)
    ax_wind.set_xlim(0, 100)
    line_wind, = ax_wind.plot([], [], 'r-')
    
    # Data Buffers
    history_len = 100
    alt_history = [10.0] * history_len
    wind_history = [0.0] * history_len
    
    print("\n=== DASHBOARD LIVE ===")
    print("Close the window to exit.")
    
    try:
        step = 0
        while True:
            # Predict & Step
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            
            # Update Data
            pos = obs[0:3]
            wind = info.get("wind", np.zeros(3))
            wind_mag = np.linalg.norm(wind)
            
            alt_history.append(pos[2])
            alt_history.pop(0)
            wind_history.append(wind_mag)
            wind_history.pop(0)
            
            # --- Render 3D View ---
            ax_3d.clear()
            ax_3d.set_xlim(-20, 20)
            ax_3d.set_ylim(-20, 20)
            ax_3d.set_zlim(0, 20)
            ax_3d.set_xlabel('X')
            ax_3d.set_ylabel('Y')
            ax_3d.set_zlabel('Z')
            ax_3d.set_title(f'Neuro-Flyt 3D | Step: {step}')
            
            # Drone
            ax_3d.scatter(pos[0], pos[1], pos[2], c='blue', s=100, label='Drone')
            # Wind Vector
            ax_3d.quiver(pos[0], pos[1], pos[2], wind[0], wind[1], wind[2], length=1.0, color='red', label='Wind')
            
            # Target
            target = info.get("target", np.array([0, 0, 10.0]))
            ax_3d.scatter(target[0], target[1], target[2], c='green', marker='x', s=100, label='Target')
            ax_3d.legend(loc='upper left')
            
            # --- Render Stats ---
            line_alt.set_ydata(alt_history)
            line_alt.set_xdata(range(history_len))
            
            line_wind.set_ydata(wind_history)
            line_wind.set_xdata(range(history_len))
            
            # Stats Text
            stats = f"Alt: {pos[2]:.2f}m\nWind: {wind_mag:.2f} N\nDrift: {np.linalg.norm(pos[:2]):.2f}m"
            ax_3d.text2D(0.05, 0.95, stats, transform=ax_3d.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
            
            plt.draw()
            plt.pause(0.01)
            
            if term or trunc:
                obs, info = env.reset()
                
            step += 1
            
            # Check if window is closed
            if not plt.fignum_exists(fig.number):
                break
                
    except KeyboardInterrupt:
        print("Interrupted.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        plt.close()
        env.close()

if __name__ == "__main__":
    run_interactive_demo()
