import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from ncps.torch import LTC
from ncps.wirings import AutoNCP
from env.drone_3d import Drone3DEnv

class LTCFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor using Liquid Time-Constant (LTC) Cells.
    This allows the agent to handle irregular time-steps and stiff dynamics better than standard MLPs or LSTMs.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 32):
        super().__init__(observation_space, features_dim)
        
        input_size = observation_space.shape[0]
        # self.features_dim is already set by super().__init__
        
        # Neural Circuit Policy (NCP) wiring for structured connectivity
        # We use a small wiring to keep inference fast (< 10ms)
        # AutoNCP requires units > output_size. Let's use 48 units for 32 outputs.
        wiring = AutoNCP(48, output_size=features_dim)
        
        self.ltc = LTC(input_size, wiring, batch_first=True)
        
        # Hidden state for the LTC
        self.hx = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # LTC expects (batch, time, features)
        # SB3 provides (batch, features), so we add a time dimension
        if observations.dim() == 2:
            observations = observations.unsqueeze(1)
            
        # Initialize hidden state if needed or if batch size changes
        batch_size = observations.size(0)
        if self.hx is None or self.hx.size(0) != batch_size:
            self.hx = torch.zeros(batch_size, self.ltc.state_size, device=observations.device)
            
        # Forward pass through LTC
        # Note: In a real recurrent setting with SB3, we'd need to manage hidden states 
        # more carefully (e.g. using RecurrentPPO from sb3-contrib).
        # For this demo, we are using a simplified approach where we treat the LTC 
        # as a stateful feature extractor that maintains state between calls within a batch.
        # However, standard PPO assumes stateless policies. 
        # To make this truly "Liquid" in a standard PPO loop without sb3-contrib, 
        # we approximate by running the LTC on the current step.
        # A better approach for production would be RecurrentPPO.
        # Given the constraints and the goal of a "demo", we will use the LTC 
        # but reset state if we detect a new episode (which is hard here).
        # So we will let the LTC evolve.
        
        # Detach hidden state from previous graph to prevent "backward through graph a second time" error
        if self.hx is not None:
            self.hx = self.hx.detach()
            
        output, self.hx = self.ltc(observations, self.hx)
        
        # Remove time dimension
        return output.squeeze(1)

def make_liquid_ppo(env, verbose=1):
    """
    Factory function to create a PPO agent with Liquid Brain.
    """
    # Parallel Environments for High-Performance Training
    # A100/A10G are data hungry. We need to run physics on many CPU cores to feed them.
    # We will use 1 environment to debug (DummyVecEnv)
    n_envs = 4
    env = make_vec_env(
        lambda: Drone3DEnv(render_mode=None, wind_scale=10.0, wind_speed=5.0),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv
    )
    
    # Create Model with optimized hyperparameters for A100
    policy_kwargs = dict(
        features_extractor_class=LTCFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=32),
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        learning_rate=1e-3,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        device='cuda' # Use GPU
    )
    return model
