"""
Liquid Neural Network Cell - Discrete-time approximation of continuous-time dynamics.

Implements a liquid cell with learnable per-neuron time constants.
The cell updates hidden state using a differential equation approximation.
"""
print("pease?")
import torch
import torch.nn as nn
import torch.nn.functional as F
print("test")

class LiquidCell(nn.Module):
    """
    A discrete-time liquid neural network cell.
    
    Hidden state update rule:
        h_{t+1,i} = h_{t,i} + dt / tau_i * ( tanh( W_hh[i]·h_t + W_xh[i]·x_t + b[i] ) - h_{t,i} )
    
    where tau_i is a learnable per-neuron time constant.
    
    Args:
        hidden_size: Number of hidden neurons
        input_size: Size of input vector
        dt: Time step for discrete approximation (default: 0.1)
    """
    
    def __init__(self, hidden_size: int, input_size: int, dt: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.dt = dt
        
        # Recurrent weight matrix: (hidden_size, hidden_size)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        
        # Input weight matrix: (hidden_size, input_size)
        self.W_xh = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)
        
        # Bias vector: (hidden_size,)
        self.b = nn.Parameter(torch.zeros(hidden_size))
        
        # Raw time constants (will be transformed to positive values)
        # Shape: (hidden_size,)
        self.tau_raw = nn.Parameter(torch.ones(hidden_size))
        
    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the liquid cell.
        
        Args:
            h: Hidden state tensor of shape (batch, hidden_size)
            x: Input tensor of shape (batch, input_size)
            
        Returns:
            Next hidden state tensor of shape (batch, hidden_size)
        """
        # Compute time constants: tau = softplus(tau_raw) + 1e-3
        # This ensures tau is always positive
        tau = F.softplus(self.tau_raw) + 1e-3
        
        # Compute preactivation:
        # preact = tanh( W_hh @ h^T + W_xh @ x^T + b )
        # Using batch matrix multiplication
        
        # W_hh @ h^T: (hidden_size, hidden_size) @ (hidden_size, batch) -> (hidden_size, batch)
        # Then transpose to (batch, hidden_size)
        h_proj = torch.matmul(h, self.W_hh.t())  # (batch, hidden_size)
        
        # W_xh @ x^T: (hidden_size, input_size) @ (input_size, batch) -> (hidden_size, batch)
        # Then transpose to (batch, hidden_size)
        x_proj = torch.matmul(x, self.W_xh.t())  # (batch, hidden_size)
        
        # Add bias and apply tanh
        preact = torch.tanh(h_proj + x_proj + self.b)  # (batch, hidden_size)
        
        # Update hidden state:
        # h_next = h + dt * (preact - h) / tau
        # tau is (hidden_size,), so we need to broadcast
        h_next = h + self.dt * (preact - h) / tau.unsqueeze(0)  # (batch, hidden_size)
        
        # Clamp to reasonable range for stability
        h_next = torch.clamp(h_next, -5.0, 5.0)
        
        return h_next





