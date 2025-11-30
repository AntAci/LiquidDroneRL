"""
Liquid Neural Network Policy for Stable-Baselines3.

Implements a custom feature extractor using LiquidCell that can be used
with PPO and other SB3 algorithms.
"""

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from models.liquid_cell import LiquidCell


class LiquidFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using a Liquid Neural Network cell.
    
    This extractor processes observations through a liquid cell to produce
    rich temporal features suitable for policy/value networks.
    
    Args:
        observation_space: Gymnasium observation space
        features_dim: Output feature dimension (default: 32)
        hidden_size: Number of hidden neurons in liquid cell (default: 32)
        dt: Time step for liquid cell (default: 0.1)
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 32,
        hidden_size: int = 32,
        dt: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)
        
        # Get observation dimension
        if isinstance(observation_space, gym.spaces.Box):
            obs_dim = observation_space.shape[0]
        else:
            raise ValueError(f"Unsupported observation space: {observation_space}")
        
        self.hidden_size = hidden_size
        self.dt = dt
        
        # Input projection layer: maps observation to hidden space
        self.input_layer = nn.Linear(obs_dim, hidden_size)
        
        # Liquid cell: processes hidden state
        self.liquid_cell = LiquidCell(hidden_size, hidden_size, dt)
        
        # Output projection: maps liquid cell output to feature dimension
        self.output_layer = nn.Linear(hidden_size, features_dim)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the liquid feature extractor.
        
        Args:
            observations: Input tensor of shape (batch, obs_dim)
            
        Returns:
            Feature tensor of shape (batch, features_dim)
        """
        # Project input to hidden space and apply tanh
        x = torch.tanh(self.input_layer(observations))  # (batch, hidden_size)
        
        # Initialize hidden state from input
        h = x
        
        # Apply one liquid cell step
        # The liquid cell uses both the hidden state and the input
        h = self.liquid_cell(h, x)  # (batch, hidden_size)
        
        # Project to output feature dimension
        features = self.output_layer(h)  # (batch, features_dim)
        
        return features





