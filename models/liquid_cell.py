
        h_next = h + self.dt * (preact - h) / tau.unsqueeze(0)  # (batch, hidden_size)
        
        # Clamp to reasonable range for stability
        h_next = torch.clamp(h_next, -5.0, 5.0)
        
        return h_next





