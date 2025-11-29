import numpy as np
import time
from env.drone_3d import Drone3DEnv

def test_wind_physics():
    print("Testing Wind Physics...")
    env = Drone3DEnv(render_mode=None, wind_scale=10.0, wind_speed=5.0)
    env.reset()
    
    # 1. Test Non-Zero Wind
    obs, _, _, _, info = env.step(np.zeros(4)) # Hover action (approx)
    wind_0 = info.get("wind", np.zeros(3))
    target_0 = info.get("target", np.zeros(3))
    print(f"Initial Wind Vector: {wind_0}")
    print(f"Target Location: {target_0}")
    
    if np.linalg.norm(wind_0) == 0:
        print("WARNING: Wind vector is zero. Check noise generation.")
    else:
        print("SUCCESS: Wind vector is non-zero.")

    # 2. Test Temporal Variation
    print("Stepping environment to test temporal variation...")
    winds = []
    for _ in range(10):
        _, _, _, _, info = env.step(np.zeros(4))
        winds.append(info["wind"])
        
    winds = np.array(winds)
    # Check if wind changes
    diffs = np.diff(winds, axis=0)
    mean_diff = np.mean(np.abs(diffs))
    print(f"Mean frame-to-frame wind change: {mean_diff:.4f}")
    
    if mean_diff > 0:
        print("SUCCESS: Wind varies over time.")
    else:
        print("FAILURE: Wind is static.")

    print("Physics Test Complete.")

if __name__ == "__main__":
    test_wind_physics()
