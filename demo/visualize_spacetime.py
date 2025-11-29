"""
Spacetime Trajectory Visualization

Creates a high-quality 3D visualization where:
- X, Y axes = Drone position
- Z axis = Time (steps)
- Target Zone = Vertical semi-transparent prism (like obstacles)
- Path = Solid line rising through time
- Wind = Color-coded arrows along the path
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.drone_env import DroneWindEnv, POSITION_MIN, POSITION_MAX

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

def collect_data(env, model, max_steps=300, seed=42):
    """Collect trajectory data."""
    obs, info = env.reset(seed=seed)
    
    positions = [(env.x, env.y)]
    winds = [(env.wind_x, env.wind_y)]
    times = [0]
    reached_target = False
    
    for step in range(1, max_steps + 1):
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
            
        obs, reward, done, truncated, info = env.step(action)
        
        positions.append((env.x, env.y))
        winds.append((env.wind_x, env.wind_y))
        times.append(step)
        
        if info.get('in_target', False):
            reached_target = True
            
        if done or truncated:
            break
            
    return np.array(positions), np.array(times), np.array(winds), reached_target

def visualize_spacetime(env, model, max_steps=300, seed=42, output_file="spacetime_plot.png"):
    """Create the 3D spacetime visualization."""
    
    # 1. Collect Data
    positions, times, winds, reached = collect_data(env, model, max_steps, seed)
    
    # 2. Setup Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Style settings
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # 3. Draw Target Zone (Vertical Prism)
    tx_min, tx_max = env.target_x_min, env.target_x_max
    ty_min, ty_max = env.target_y_min, env.target_y_max
    
    # Create prism vertices
    t_start = 50  # Target spawn time
    t_end = times[-1]
    
    # Define the 4 corners of the target box
    corners = [
        [tx_min, ty_min], [tx_max, ty_min],
        [tx_max, ty_max], [tx_min, ty_max]
    ]
    
    # Create faces for the prism
    faces = []
    # Side faces
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i+1)%4]
        faces.append([
            [p1[0], p1[1], t_start],
            [p2[0], p2[1], t_start],
            [p2[0], p2[1], t_end],
            [p1[0], p1[1], t_end]
        ])
    
    # Top and bottom faces
    faces.append([[c[0], c[1], t_start] for c in corners])
    faces.append([[c[0], c[1], t_end] for c in corners])
    
    # Add prism to plot
    prism = Poly3DCollection(faces, alpha=0.2, facecolor='purple', edgecolor='purple', linewidth=0.5)
    ax.add_collection3d(prism)
    
    # 4. Plot Trajectory (The "Learned Path")
    ax.plot(positions[:, 0], positions[:, 1], times, 
            color='blue', linewidth=3, label='Learned Path')
    
    # 5. Add Start/End Markers
    ax.scatter(positions[0, 0], positions[0, 1], times[0], 
              color='green', s=150, label='Start', zorder=10)
    
    # End marker (Red if failed, Gold star if succeeded)
    end_color = 'gold' if reached else 'red'
    end_marker = '*' if reached else 'o'
    end_label = 'Goal Reached' if reached else 'End Position'
    
    ax.scatter(positions[-1, 0], positions[-1, 1], times[-1], 
              color=end_color, s=200, marker=end_marker, label=end_label, zorder=10)
    
    # 6. Add Wind Arrows (Color-coded)
    # Show wind every ~10 steps to avoid clutter
    skip = max(1, len(times) // 25)
    
    for i in range(0, len(times), skip):
        wx, wy = winds[i]
        w_mag = np.sqrt(wx**2 + wy**2)
        
        # Color scheme matching Pygame
        if w_mag < 1.0: color = 'green'
        elif w_mag < 1.5: color = 'gold' # Yellow is hard to see on white
        else: color = 'orange'
        
        # Arrow scaling
        scale = 0.15
        ax.quiver(positions[i, 0], positions[i, 1], times[i],
                 wx * scale, wy * scale, 0,
                 color=color, alpha=0.6, arrow_length_ratio=0.3)
    
    # 7. Formatting like the reference image
    ax.set_xlabel('X Position', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y Position', fontsize=11, fontweight='bold')
    ax.set_zlabel('Time (Steps)', fontsize=11, fontweight='bold')
    
    ax.set_xlim(POSITION_MIN, POSITION_MAX)
    ax.set_ylim(POSITION_MIN, POSITION_MAX)
    ax.set_zlim(0, times[-1] * 1.1)
    
    # Legend with custom handles for clarity
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = [
        Line2D([0], [0], color='blue', lw=3, label='Learned Path'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Start'),
        Line2D([0], [0], marker=end_marker, color='w', markerfacecolor=end_color, markersize=15, label=end_label),
        Patch(facecolor='purple', alpha=0.2, label='Target Zone'),
        Line2D([0], [0], color='green', lw=1, label='Weak Wind'),
        Line2D([0], [0], color='gold', lw=1, label='Medium Wind'),
        Line2D([0], [0], color='orange', lw=1, label='Strong Wind')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
    
    # Info text overlay
    status_text = f"Steps: {len(times)}\nFinal Reward: {sum([0]):.2f}" # We need reward tracking to show this accurately
    ax.text2D(0.05, 0.95, status_text, transform=ax.transAxes, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # View angle similar to reference
    ax.view_init(elev=30, azim=-60)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_file}")
    if not output_file.endswith('.gif'):
        plt.show()

def animate_spacetime(env, model, max_steps=300, seed=42, output_file="spacetime_animation.gif"):
    """Create an animated 3D spacetime visualization."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    # 1. Collect Data
    positions, times, winds, reached = collect_data(env, model, max_steps, seed)
    
    # 2. Setup Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Style
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # 3. Setup Fixed Elements (Target Zone)
    tx_min, tx_max = env.target_x_min, env.target_x_max
    ty_min, ty_max = env.target_y_min, env.target_y_max
    t_start, t_end = 50, times[-1]
    
    corners = [[tx_min, ty_min], [tx_max, ty_min], [tx_max, ty_max], [tx_min, ty_max]]
    faces = []
    for i in range(4):
        p1, p2 = corners[i], corners[(i+1)%4]
        faces.append([[p1[0], p1[1], t_start], [p2[0], p2[1], t_start], 
                      [p2[0], p2[1], t_end], [p1[0], p1[1], t_end]])
    faces.append([[c[0], c[1], t_start] for c in corners])
    faces.append([[c[0], c[1], t_end] for c in corners])
    
    prism = Poly3DCollection(faces, alpha=0.15, facecolor='purple', edgecolor='purple', linewidth=0.5)
    ax.add_collection3d(prism)
    
    # 4. Initialize Animation Elements
    line, = ax.plot([], [], [], color='blue', linewidth=3, label='Learned Path')
    head = ax.scatter([], [], [], color='blue', s=100, label='Current Position')
    start = ax.scatter([positions[0,0]], [positions[0,1]], [times[0]], color='green', s=150, label='Start')
    
    # Wind arrows container
    wind_arrows = []
    
    # Pre-calculate wind field slices (domain-wide gusts)
    # We can't record every point in the grid during collection easily without changing collect_data
    # But since wind is uniform across space in this env (mostly), we can use the recorded wind
    # to populate a grid at specific time slices.
    
    def plot_wind_slice(time_idx, z_height):
        """Plot a grid of wind arrows at a specific time height."""
        wx, wy = winds[time_idx]
        w_mag = np.sqrt(wx**2 + wy**2)
        
        if w_mag < 1.0: color = 'green'
        elif w_mag < 1.5: color = 'gold'
        else: color = 'orange'
        
        # Create a 3x3 grid of arrows for this slice
        grid_x = np.linspace(POSITION_MIN + 0.2, POSITION_MAX - 0.2, 4)
        grid_y = np.linspace(POSITION_MIN + 0.2, POSITION_MAX - 0.2, 4)
        X, Y = np.meshgrid(grid_x, grid_y)
        Z = np.full_like(X, z_height)
        U = np.full_like(X, wx)
        V = np.full_like(X, wy)
        W = np.zeros_like(X)
        
        scale = 0.15
        # Quiver returns a collection, easier to manage
        return ax.quiver(X, Y, Z, U*scale, V*scale, W, color=color, alpha=0.25, arrow_length_ratio=0.3)

    # Axis limits
    ax.set_xlim(POSITION_MIN, POSITION_MAX)
    ax.set_ylim(POSITION_MIN, POSITION_MAX)
    ax.set_zlim(0, times[-1] * 1.1)
    
    ax.set_xlabel('X Position', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y Position', fontsize=11, fontweight='bold')
    ax.set_zlabel('Time (Steps)', fontsize=11, fontweight='bold')
    
    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color='blue', lw=3, label='Learned Path'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Start'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Current Position'),
        Patch(facecolor='purple', alpha=0.2, label='Target Zone'),
        Line2D([0], [0], color='green', lw=1, label='Weak Wind (Field)'),
        Line2D([0], [0], color='gold', lw=1, label='Medium Wind (Field)'),
        Line2D([0], [0], color='orange', lw=1, label='Strong Wind (Field)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

    def update(frame):
        # Update path
        line.set_data(positions[:frame, 0], positions[:frame, 1])
        line.set_3d_properties(times[:frame])
        
        # Update head
        head._offsets3d = ([positions[frame, 0]], [positions[frame, 1]], [times[frame]])
        
        # Add wind slice occasionally (every 50 steps) to show domain-wide conditions
        if frame % 40 == 0:
            arrows = plot_wind_slice(frame, times[frame])
            wind_arrows.append(arrows)
            
        # Rotate camera slightly
        ax.view_init(elev=30, azim=-60 + frame * 0.2)
        return line, head
    
    print(f"Creating animation ({len(times)} frames)...")
    anim = FuncAnimation(fig, update, frames=len(times), interval=50, blit=False)
    
    writer = PillowWriter(fps=30)
    anim.save(output_file, writer=writer)
    print(f"✓ Animation saved to {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Spacetime Trajectory Visualization")
    parser.add_argument("--model-path", type=str, default="models/liquid_policy.zip")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="spacetime_plot.png")
    parser.add_argument("--animate", action="store_true", help="Create GIF animation instead of static plot")
    args = parser.parse_args()
    
    env = DroneWindEnv()
    
    model = None
    if not args.random and SB3_AVAILABLE and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}...")
        model = PPO.load(args.model_path, env=env)
    else:
        print("Using random actions.")
        
    if args.animate:
        output = args.output if args.output.endswith('.gif') else args.output.replace('.png', '.gif')
        animate_spacetime(env, model, seed=args.seed, output_file=output)
    else:
        visualize_spacetime(env, model, seed=args.seed, output_file=args.output)


if __name__ == "__main__":
    main()
