#!/bin/bash
# Setup script for Drone RL project

echo "Setting up Drone RL project..."
echo ""

# Check Python version
PYTHON_CMD="python3.10"
if ! command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_CMD="python3.11"
fi

if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Error: Python 3.10+ not found. Please install Python 3.10 or higher."
    exit 1
fi

echo "Using: $($PYTHON_CMD --version)"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Setup complete! To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the setup, run:"
echo "  python main.py"


