#!/bin/bash
# Training script for fraud detection models

set -e

echo "=== Fraud Detection Model Training ==="
echo "Starting training pipeline..."

# Build the Docker image
echo "Building Docker image..."
docker-compose build

# Run training
echo "Starting model training..."
docker-compose --profile training up --abort-on-container-exit

# Run evaluation
echo "Starting model evaluation..."
docker-compose --profile evaluation up --abort-on-container-exit

# Cleanup
echo "Cleaning up containers..."
docker-compose --profile training down
docker-compose --profile evaluation down

echo "Training pipeline completed!"
echo "Check the 'models/' directory for trained artifacts."