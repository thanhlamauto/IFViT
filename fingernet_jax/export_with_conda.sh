#!/bin/bash
# Export weights using Conda environment (alternative to Docker)

set -e

cd /Users/nguyenthanhlam/SSL_Correspondence/IFViT/fingernet_jax

echo "=========================================="
echo "FingerNet Weight Export - Conda Method"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "✗ Conda not found. Please install Miniconda/Anaconda first."
    echo ""
    echo "Install from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda found: $(conda --version)"
echo ""

# Check if tf1_env exists
if conda env list | grep -q "^tf1_env "; then
    echo "✓ Found existing tf1_env"
    echo ""
    echo "Activating environment and running export..."
    echo ""
    
    # Activate and run
    eval "$(conda shell.bash hook)"
    conda activate tf1_env
    
    python export_fingernet_weights_tf1.py
    
    conda deactivate
    
    if [ -f fingernet_keras_tf1_weights.npz ]; then
        echo ""
        echo "=========================================="
        echo "✓ Export completed!"
        echo "=========================================="
        echo ""
        echo "File: fingernet_keras_tf1_weights.npz"
        echo "Size: $(du -h fingernet_keras_tf1_weights.npz | cut -f1)"
        echo ""
        echo "Next: python convert_from_npz.py"
    else
        echo ""
        echo "✗ Export failed - file not found"
        exit 1
    fi
else
    echo "⚠️  tf1_env not found. Creating it now..."
    echo ""
    echo "This will:"
    echo "  1. Create conda environment 'tf1_env' with Python 3.7"
    echo "  2. Install TensorFlow 1.15, Keras 2.3.1, and dependencies"
    echo "  3. Run export script"
    echo ""
    echo "Auto-continuing (use Ctrl+C to cancel)..."
    sleep 2
    
    echo ""
    echo "[1/3] Creating conda environment..."
    conda create -n tf1_env python=3.7 -y
    
    echo ""
    echo "[2/3] Installing TensorFlow 1.15 and dependencies..."
    eval "$(conda shell.bash hook)"
    conda activate tf1_env
    conda install tensorflow=1.15 keras=2.3.1 -y
    pip install numpy scipy opencv-python matplotlib
    
    echo ""
    echo "[3/3] Running export..."
    python export_fingernet_weights_tf1.py
    
    conda deactivate
    
    if [ -f fingernet_keras_tf1_weights.npz ]; then
        echo ""
        echo "=========================================="
        echo "✓ Export completed!"
        echo "=========================================="
        echo ""
        echo "File: fingernet_keras_tf1_weights.npz"
        echo "Size: $(du -h fingernet_keras_tf1_weights.npz | cut -f1)"
        echo ""
        echo "Next: python convert_from_npz.py"
    else
        echo ""
        echo "✗ Export failed - file not found"
        exit 1
    fi
fi

