import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from env.drone_3d import Drone3DEnv
from models.liquid_ppo import make_liquid_ppo

def train():
    print("Setting up Training Environment...")
    # Create environment
    # We use a lower wind scale for training initially to help it learn
    env = Drone3DEnv(render_mode=None, wind_scale=2.0, wind_speed=1.0)
    
    print("Creating Liquid PPO Agent...")
    model = make_liquid_ppo(env, verbose=1)
    
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="liquid_ppo_drone"
    )
    
    print("Starting Training (This may take a while)...")
    # Training for 500,000 steps (500 episodes) as requested for proper convergence
    total_timesteps = 500000
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    print("Training Complete.")
    model.save("liquid_ppo_drone_final")
    print("Model saved to 'liquid_ppo_drone_final.zip'")

if __name__ == "__main__":
    train()
