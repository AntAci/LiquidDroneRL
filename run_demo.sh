#!/bin/bash
set -e

echo "=== Project Neuro-Flyt 3D Setup ==="
echo "Installing dependencies..."
pip install -r requirements.txt

echo "=== Launching Demo ==="
python demo_3d.py
