#!/bin/bash

set -e

echo "=========================================="
echo "CUDA Image Processing Project"
echo "=========================================="

echo ""
echo "GPU information:"
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv

echo ""
echo "Compiling CUDA program..."
make

echo ""
echo "Running CUDA image processing..."
./image_filter 100 256 256 output

echo ""
echo "Execution completed successfully."
