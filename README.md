---
title: Neuro Flyt Training
emoji: 🚁
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🚁 Neuro-Flyt 3D: Liquid Neural Network Drone Control

**Neuro-Flyt 3D** is a cutting-edge reinforcement learning project that trains a drone to navigate complex 3D environments with dynamic wind fields using **Liquid Neural Networks (LNNs)**.

This project demonstrates the power of **Liquid Time-Constant (LTC)** networks to handle continuous-time dynamics and irregular sampling, making them ideal for robotic control tasks like drone flight in turbulent conditions.

## 🌟 Key Features

*   **Liquid Brain**: Uses `ncps` (Neural Circuit Policies) to implement an LTC-based feature extractor. This allows the agent to adapt to changing dynamics (like wind gusts) more effectively than standard MLPs.
*   **3D Physics Environment**: A custom `Gymnasium` environment (`Drone3DEnv`) that simulates:
    *   6-DOF Drone Dynamics.
    *   **3D Perlin Noise Wind Field**: Realistic, spatially continuous wind turbulence.
    *   "Antigravity" Force Fields (optional demo mode).
*   **Hugging Face Spaces Integration**: Fully containerized (`Dockerfile`) for training on Hugging Face Spaces with GPU support.
*   **PPO Algorithm**: Uses `Stable-Baselines3` Proximal Policy Optimization for robust training.

## 📂 Project Structure

*   `env/drone_3d.py`: **The World**. Defines the drone physics, wind field, and reward function.
*   `models/liquid_ppo.py`: **The Brain**. Defines the PPO agent with the custom `LTCFeatureExtractor`.
*   `train_hf.py`: **The Trainer**. Script to launch training on Hugging Face Spaces.
*   `visualize_agent.py`: **The Demo**. Loads a trained model and generates a GIF of the drone flying.
*   `Dockerfile`: **The Container**. Defines the environment for cloud training.

## 🚀 How to Train (Hugging Face Spaces)

This project is optimized to run on Hugging Face Spaces with GPU acceleration.

### 1. Hardware Setup
*   Create a Space or go to Settings.
*   Select **T4 small** or **A10G** GPU hardware.

### 2. Configuration
The `Dockerfile` is set to train for **500,000 steps** by default.
The `models/liquid_ppo.py` is configured for **CUDA** (GPU) and **4 parallel environments**.

### 3. Deployment
Push the code to your Space:
```bash
git push space abi-3d:main
```
*(Note: If deploying to a Space, you usually push to the `main` branch of the Space remote).*

## 💻 Local Usage

### Installation
```bash
pip install -r requirements.txt
```

### Visualization
To see the trained agent in action:
1.  Download `liquid_ppo_drone_final.zip` from your Space's "Files" tab.
2.  Run:
    ```bash
    python visualize_agent.py
    ```
    This will generate `drone_flight.gif`.

## 🧠 The "Liquid" Advantage
Standard neural networks (RNNs/LSTMs) have a fixed tick rate. Liquid networks are defined by differential equations, allowing them to:
1.  **Generalize better** to unseen physics.
2.  **Process irregular time series** (e.g., if the drone sensor lags).
3.  **Maintain stability** in chaotic environments (like high wind).

---
*Branch: `abi-3d`*
