#!/bin/bash
set -e

# Use env var for token
TOKEN=$HF_TOKEN
if [ -z "$TOKEN" ]; then
    echo "Error: HF_TOKEN env var is not set."
    exit 1
fi

REPO="https://user:$TOKEN@huggingface.co/spaces/iteratehack/team_22"

# 1. Setup Deploy Directory
if [ -d "hf_deploy" ]; then
    rm -rf hf_deploy
fi
mkdir hf_deploy

# 2. Clone the Space
echo "Cloning Space..."
git clone $REPO hf_deploy

# 3. Copy Files (Explicitly to avoid recursion and secrets)
echo "Copying files..."
cp Dockerfile hf_deploy/
cp README.md hf_deploy/
cp README_HF.md hf_deploy/
cp requirements.txt hf_deploy/
cp run_demo.sh hf_deploy/
cp setup.sh hf_deploy/
cp main.py hf_deploy/
cp train_hf.py hf_deploy/
cp demo_3d.py hf_deploy/
cp demo_interactive.py hf_deploy/
cp test_physics.py hf_deploy/
cp debug_imports.py hf_deploy/

# Copy directories
cp -r env hf_deploy/
cp -r models hf_deploy/
cp -r train hf_deploy/
cp -r eval hf_deploy/
cp -r utils hf_deploy/
cp -r demo hf_deploy/

# 4. Push
cd hf_deploy
git config user.email "agent@antigravity.com"
git config user.name "Antigravity Agent"
git add .
git commit -m "Deploy Fix: Disable interactive build"
git push

echo "Deployment Complete!"
