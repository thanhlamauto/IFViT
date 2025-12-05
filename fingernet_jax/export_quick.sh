#!/bin/bash
# Quick export script - chạy trong Docker với output rõ ràng

set -e

cd /Users/nguyenthanhlam/SSL_Correspondence/IFViT

echo "=========================================="
echo "Exporting FingerNet weights..."
echo "=========================================="
echo ""

docker run --rm \
  -v "$(pwd):/workspace" \
  -w /workspace/fingernet_jax \
  tensorflow/tensorflow:1.15.5-gpu-py3 \
  bash -c "
    set -e
    echo '[1/3] Installing dependencies...'
    pip install -q numpy scipy opencv-python matplotlib 2>&1 | grep -v 'already satisfied' || true
    echo '[2/3] Exporting weights from Model.model...'
    python export_fingernet_weights_tf1.py
    echo '[3/3] Checking output...'
    if [ -f fingernet_keras_tf1_weights.npz ]; then
        echo ''
        echo '✓ SUCCESS!'
        ls -lh fingernet_keras_tf1_weights.npz
    else
        echo '✗ FAILED - output file not found'
        exit 1
    fi
  "

echo ""
echo "=========================================="
if [ -f fingernet_jax/fingernet_keras_tf1_weights.npz ]; then
    echo "✓ Export completed!"
    echo "File: fingernet_jax/fingernet_keras_tf1_weights.npz"
    echo ""
    echo "Next step: Convert to Flax"
    echo "  cd fingernet_jax"
    echo "  python convert_from_npz.py"
else
    echo "✗ Export failed"
    exit 1
fi

