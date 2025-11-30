"""
Main entry point for the Drone RL project.
Tests imports and basic functionality.
"""

import sys
import numpy as np

print("=" * 50)
print("Drone RL Project - Import Test")
print("=" * 50)

# Test PyTorch
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} imported successfully")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

# Test NumPy
try:
    print(f"✓ NumPy {np.__version__} imported successfully")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")
    sys.exit(1)

# Test Gymnasium
try:
    import gymnasium as gym
    print(f"✓ Gymnasium {gym.__version__} imported successfully")
    
    # Test creating a simple environment
    try:
        env = gym.make("CartPole-v1")
        print(f"✓ Gymnasium environment 'CartPole-v1' created successfully")
        env.close()
    except Exception as e:
        print(f"⚠ Could not create test environment: {e}")
except ImportError as e:
    print(f"✗ Gymnasium import failed: {e}")
    sys.exit(1)

# Test Stable-Baselines3
try:
    import stable_baselines3
    print(f"✓ Stable-Baselines3 imported successfully")
except ImportError as e:
    print(f"⚠ Stable-Baselines3 import failed: {e}")
    print("  (This is optional, you can use cleanrl instead)")

# Test Pygame
pygame_works = False
try:
    import pygame
    print(f"✓ Pygame {pygame.__version__} imported successfully")
    
    # Try to initialize pygame
    try:
        pygame.init()
        print("✓ Pygame initialized successfully")
        
        # Create a test window
        screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption("Drone RL - Pygame Test")
        print("✓ Pygame window created successfully")
        print("  Window should be visible. Close it to continue...")
        
        # Keep window open briefly, then close
        import time
        clock = pygame.time.Clock()
        running = True
        start_time = time.time()
        
        while running and (time.time() - start_time) < 2.0:  # Show for 2 seconds
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            screen.fill((50, 50, 50))  # Dark gray background
            font = pygame.font.Font(None, 36)
            text = font.render("Pygame Works!", True, (255, 255, 255))
            text_rect = text.get_rect(center=(320, 240))
            screen.blit(text, text_rect)
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        pygame_works = True
        print("✓ Pygame window test completed successfully")
        
    except Exception as e:
        print(f"⚠ Pygame initialization failed: {e}")
        pygame.quit()
        
except ImportError as e:
    print(f"✗ Pygame import failed: {e}")

# Fallback to matplotlib if pygame failed
if not pygame_works:
    print("\n" + "=" * 50)
    print("Falling back to matplotlib animation...")
    print("=" * 50)
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        
        print("✓ Matplotlib imported successfully")
        
        # Create a simple animation
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title("Drone RL - Matplotlib Test")
        
        line, = ax.plot([], [], 'o-', lw=2)
        
        def animate(frame):
            x = np.linspace(0, 10, 100)
            y = np.sin(x + frame * 0.1) * 5 + 5
            line.set_data(x, y)
            return line,
        
        anim = animation.FuncAnimation(fig, animate, frames=100, interval=50, blit=True)
        print("✓ Matplotlib animation created")
        print("  Animation window should be visible. Close it to continue...")
        plt.show(block=False)
        
        # Keep it open briefly
        import time
        time.sleep(2)
        plt.close()
        
        print("✓ Matplotlib animation test completed successfully")
        
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        print("  Please install matplotlib: pip install matplotlib")

print("\n" + "=" * 50)
print("All tests completed!")
print("=" * 50)





