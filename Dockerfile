FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies for PyBullet/OpenGL
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install huggingface_hub

# Fix for Numba and Matplotlib permission errors
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV MPLCONFIGDIR=/tmp/matplotlib_config
ENV HF_HOME=/tmp/huggingface

# Copy project files
COPY . .

# Default command (can be overridden in Space settings)
# Expects HF_TOKEN and REPO_ID env vars to be set in the Space
CMD ["python", "train_hf.py", "--repo_id", "ylop/neuro-flyt-3d", "--steps", "500000"]
