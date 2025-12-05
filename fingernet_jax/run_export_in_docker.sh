#!/bin/bash
# Script tự động export weights bằng Docker

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "FingerNet Weight Export - Docker Auto"
echo "=========================================="
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "✗ Docker not found. Please install Docker first."
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "✗ Docker daemon is not running."
    echo ""
    echo "Please start Docker Desktop and try again."
    echo "On macOS: Open Docker Desktop application"
    exit 1
fi

echo "✓ Docker is available"
echo ""

# Check if Model.model exists
MODEL_PATH="$PROJECT_ROOT/FingerNet/models/released_version/Model.model"
if [ ! -f "$MODEL_PATH" ]; then
    echo "✗ Model.model not found at: $MODEL_PATH"
    exit 1
fi

echo "✓ Found Model.model"
echo ""

# Output path
OUTPUT_NPZ="$SCRIPT_DIR/fingernet_keras_tf1_weights.npz"

# Run Docker container
echo "Starting Docker container with TensorFlow 1.15..."
echo "This may take a few minutes on first run (downloading image)..."
echo ""

docker run --rm \
  -v "$PROJECT_ROOT:/workspace" \
  -w /workspace/fingernet_jax \
  tensorflow/tensorflow:1.15.5-gpu-py3 \
  bash -c "
    set -e
    echo '[1/4] Installing dependencies...'
    echo '  → numpy...'
    pip install --no-cache-dir numpy 2>&1 | grep -E '(Collecting|Installing|Successfully)' || true
    echo '  → scipy...'
    pip install --no-cache-dir scipy 2>&1 | grep -E '(Collecting|Installing|Successfully)' || true
    echo '  → opencv-python...'
    pip install --no-cache-dir opencv-python 2>&1 | grep -E '(Collecting|Installing|Successfully)' || true
    echo '  → matplotlib...'
    pip install --no-cache-dir matplotlib 2>&1 | grep -E '(Collecting|Installing|Successfully)' || true
    echo '  ✓ All dependencies installed'
    echo ''
    
    echo '[2/4] Starting export script...'
    python export_fingernet_weights_tf1.py
    echo ''
    
    echo '[3/4] Verifying output...'
    if [ -f fingernet_keras_tf1_weights.npz ]; then
        ls -lh fingernet_keras_tf1_weights.npz
        echo ''
        echo '✓ Export successful!'
    else
        echo '✗ Export failed - output file not found'
        exit 1
    fi
  "

DOCKER_EXIT=$?

if [ $DOCKER_EXIT -eq 0 ] && [ -f "$OUTPUT_NPZ" ]; then
    echo ""
    echo "=========================================="
    echo "✓ Export completed successfully!"
    echo "=========================================="
    echo ""
    echo "Output file: $OUTPUT_NPZ"
    echo "File size: $(du -h "$OUTPUT_NPZ" | cut -f1)"
    echo ""
    echo "[4/4] Next step: Convert to Flax format"
    echo "  cd $SCRIPT_DIR"
    echo "  python convert_from_npz.py"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ Export failed"
    echo "=========================================="
    echo ""
    if [ $DOCKER_EXIT -ne 0 ]; then
        echo "Docker container exited with code: $DOCKER_EXIT"
    fi
    if [ ! -f "$OUTPUT_NPZ" ]; then
        echo "Output file not found: $OUTPUT_NPZ"
    fi
    echo ""
    echo "Please check the error messages above."
    exit 1
fi

