#!/bin/bash
# Export script with timeout protection

set -e

cd /Users/nguyenthanhlam/SSL_Correspondence/IFViT/fingernet_jax

echo "=========================================="
echo "FingerNet Weight Export (with timeout)"
echo "=========================================="
echo ""

# Set overall timeout (10 minutes)
timeout 600 docker run --rm \
  -v "$(pwd)/..:/workspace" \
  -w /workspace/fingernet_jax \
  tensorflow/tensorflow:1.15.5-gpu-py3 \
  bash -c "
    set -e
    echo '[1/4] Installing dependencies (this may take 2-3 minutes)...'
    pip install --no-cache-dir numpy scipy opencv-python matplotlib 2>&1 | \
      grep -E '(Collecting|Installing|Successfully|Requirement already)' | head -20
    echo '  ✓ Dependencies ready'
    echo ''
    
    echo '[2/4] Loading model and weights (this may take 1-2 minutes)...'
    python export_fingernet_weights_tf1.py
    echo ''
    
    echo '[3/4] Verifying output...'
    if [ -f fingernet_keras_tf1_weights.npz ]; then
        ls -lh fingernet_keras_tf1_weights.npz
        echo '  ✓ File created successfully'
    else
        echo '  ✗ File not found!'
        exit 1
    fi
  "

if [ $? -eq 0 ] && [ -f fingernet_keras_tf1_weights.npz ]; then
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
    echo "✗ Export failed or timed out"
    exit 1
fi

