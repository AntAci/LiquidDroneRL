import os
import argparse
from huggingface_hub import HfApi, login
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from env.drone_3d import Drone3DEnv
from models.liquid_ppo import make_liquid_ppo, LTCFeatureExtractor

def train_hf(repo_id, token, total_timesteps=500000):
    print(f"Starting HF Training for Repo: {repo_id}")
    
    # Login to HF
    if token:
        login(token=token)
    
    # Create Optimized Model (Parallel Envs + A100 Tuning)
    # Note: make_liquid_ppo now handles env creation internally for parallelism
    print("Creating Liquid PPO Model...")
    try:
        model = make_liquid_ppo(None, verbose=1) 
        print("Model created successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR creating model: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    # Checkpoint Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='/tmp/checkpoints/',
        name_prefix='liquid_ppo_drone'
    )
    
    print(f"Training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    # Save Final Model
    model_path = "/tmp/liquid_ppo_drone_final.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Push to Hub
    print("Pushing to Hugging Face Hub...")
    api = HfApi()
    
    try:
        # Create repo if it doesn't exist
        api.create_repo(repo_id=repo_id, exist_ok=True)
        
        # Upload Model
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="liquid_ppo_drone_final.zip",
            repo_id=repo_id,
            repo_type="model"
        )
        print("Upload Complete!")
        print("SCRIPT FINISHED SUCCESSFULLY")
        
    except Exception as e:
        print(f"Error uploading to Hub: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, help="HF Repo ID (e.g., username/neuro-flyt-3d)")
    parser.add_argument("--token", type=str, help="HF Write Token")
    parser.add_argument("--steps", type=int, default=500000, help="Total training steps")
    
    args = parser.parse_args()
    
    # Get token from env var if not provided
    token = args.token or os.environ.get("HF_TOKEN")
    
    train_hf(args.repo_id, token, args.steps)
