#!/usr/bin/env bash
set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Training model..."
python train_model.py

echo "Build complete."
