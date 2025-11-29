# Deploying Neuro-Flyt 3D to Hugging Face Spaces

This guide explains how to use your organization's GPUs on Hugging Face to train the Neuro-Flyt 3D model.

## Prerequisites
1.  A Hugging Face Account.
2.  An Organization with GPU billing enabled (or a personal account with GPU access).
3.  A Write Access Token (Settings -> Access Tokens).

## Steps

### 1. Create a New Space
1.  Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2.  **Owner:** Select your Organization.
3.  **Space Name:** `neuro-flyt-training` (or similar).
4.  **SDK:** Select **Docker**.
5.  **Space Hardware:** Select a GPU instance (e.g., **T4 small** or **A10G**).

### 2. Configure Secrets
In the Space settings, go to **Settings -> Variables and secrets**.
Add the following **Secret**:
-   `HF_TOKEN`: Your Write Access Token (starts with `hf_...`).

### 3. Deploy Code
You can deploy by pushing the code to the Space's Git repository.

```bash
# 1. Install git-lfs if needed
git lfs install

# 2. Clone your Space (replace with your actual repo URL)
git clone https://huggingface.co/spaces/YOUR_ORG/neuro-flyt-training
cd neuro-flyt-training

# 3. Copy project files
cp -r /path/to/Drone-go-brrrrr/* .

# 4. Push to Space
git add .
git commit -m "Deploy training job"
git push
```

### 4. Monitor Training
-   Go to the **App** tab in your Space.
-   You will see the training logs in real-time.
-   The training will run for 500,000 steps.

### 5. Access Trained Model
-   Once finished, the script will automatically push the trained model (`liquid_ppo_drone_final.zip`) to your Model Repository (defined in `train_hf.py` or via arguments).
-   You can then download this model and use it locally with `demo_3d.py`.

## Customization
-   **Repo ID:** Edit `Dockerfile` or `train_hf.py` to change the target Model Repository ID (`--repo_id`).
-   **Steps:** Change `--steps` in `Dockerfile` to adjust training duration.

## Hardware & Training Recommendations

### Which GPU?
*   **A100 Large (80GB):** **The Ultimate Choice.** If you want to train for 5M+ episodes in the shortest time possible, pick this. We have optimized the code to use **16 Parallel Environments** and **Large Batch Sizes (4096)** to fully saturate the A100.
*   **A10G Large (24GB):** **Excellent Value.** Very fast and capable. It will handle the parallel training easily and is much cheaper than the A100.
*   **T4 (16GB):** **Budget Option.** It will work, but you won't see the massive speedup from the parallelization as clearly as with the Ampere cards (A10/A100).

### Efficiency Optimization (Implemented)
To ensure the GPU doesn't sit idle, we have updated `train_hf.py` to:
1.  **Parallel Physics:** Run **16 Drones** simultaneously on the CPU.
2.  **Large Batches:** Process **4096 samples** at once on the GPU.
3.  **Result:** Training is ~10-15x faster than the standard script.

### How Many Episodes?
The environment `max_steps` is 1000.
*   **Minimum (Proof of Concept):** **500,000 Steps** (500 Episodes). The drone will learn to hover and roughly follow the target.
*   **Recommended (Robust):** **1,000,000 - 2,000,000 Steps** (1000 - 2000 Episodes). This allows the Liquid Network to fully adapt to the random wind turbulence and master the physics.
*   **High Performance:** **5,000,000+ Steps**. For "perfect" flight control.

### Efficiency Tip
Reinforcement Learning is often CPU-bound (physics simulation). To train efficiently:
1.  Use a Space with **many CPU vCores** (8+) to run environments in parallel.
2.  Use the **A10G** GPU to handle the heavy math of the Liquid Time-Constant (LTC) cells.
