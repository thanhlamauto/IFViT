#!/bin/bash
# Helper script để convert FingerNet weights từ Model.model sang JAX

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="$SCRIPT_DIR/../FingerNet/models/released_version/Model.model"
EXPORT_SCRIPT="$SCRIPT_DIR/export_fingernet_weights_tf1.py"
CONVERT_SCRIPT="$SCRIPT_DIR/convert_from_npz.py"
EXPORTED_NPZ="$SCRIPT_DIR/fingernet_keras_tf1_weights.npz"
OUTPUT_NPZ="$SCRIPT_DIR/fingernet_flax_params.npz"

echo "=========================================="
echo "FingerNet Weight Conversion Helper"
echo "=========================================="
echo ""

# Check if Model.model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "✗ Error: Model.model not found at: $MODEL_PATH"
    exit 1
fi

echo "✓ Found Model.model at: $MODEL_PATH"
echo ""

# Check if already converted
if [ -f "$OUTPUT_NPZ" ]; then
    echo "✓ Found existing Flax weights: $OUTPUT_NPZ"
    echo "  To reconvert, delete this file first."
    exit 0
fi

# Check if exported .npz exists
if [ -f "$EXPORTED_NPZ" ]; then
    echo "✓ Found exported weights: $EXPORTED_NPZ"
    echo "  Converting to Flax format..."
    python "$CONVERT_SCRIPT" "$EXPORTED_NPZ"
    exit 0
fi

# Need to export first
echo "⚠️  Need to export weights from TensorFlow 1.x / Keras 2.x first."
echo ""
echo "Options:"
echo ""
echo "1. Using Docker (Recommended):"
echo "   docker run -it --rm \\"
echo "     -v $SCRIPT_DIR/../..:/workspace \\"
echo "     tensorflow/tensorflow:1.15.5-gpu-py3 bash"
echo "   # Then inside container:"
echo "   cd /workspace/IFViT/fingernet_jax"
echo "   pip install numpy scipy"
echo "   python export_fingernet_weights_tf1.py"
echo ""
echo "2. Using Conda:"
echo "   conda create -n tf1_env python=3.7"
echo "   conda activate tf1_env"
echo "   conda install tensorflow=1.15 keras=2.3.1 numpy scipy"
echo "   cd $SCRIPT_DIR"
echo "   python export_fingernet_weights_tf1.py"
echo ""
echo "3. After exporting, run:"
echo "   python $CONVERT_SCRIPT"
echo ""

exit 1

