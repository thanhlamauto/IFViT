# FingerNet Setup Instructions

## ⚠️ Compatibility Issue

The original FingerNet code requires:
- Python 2.7 or Python 3.6-3.7
- TensorFlow 1.0.1 or 1.15.x
- Keras 2.0.2 or 2.2.4
- Old versions of numpy, scipy, etc.

**These are NOT compatible with Python 3.13.**

## ✅ Recommended Solution: Use Conda

### Step 1: Install Miniconda (if not installed)
```bash
# Download from https://docs.conda.io/en/latest/miniconda.html
# Or install via Homebrew on macOS:
brew install miniconda
```

### Step 2: Create Conda Environment
```bash
cd /Users/nguyenthanhlam/SSL_Correspondence/IFViT/FingerNet

# Create environment with Python 3.7
conda create -n fingernet_tf1 python=3.7 -y
conda activate fingernet_tf1

# Install TensorFlow 1.15 (last stable TF1 version)
pip install tensorflow==1.15.5

# Install other dependencies
pip install keras==2.2.4
pip install numpy==1.16.6
pip install scipy==1.2.3
pip install opencv-python==4.2.0.32
pip install matplotlib==3.1.3
pip install h5py==2.10.0
pip install Pillow==6.2.2
```

### Step 3: Test FingerNet
```bash
cd src
python train_test_deploy.py 0 deploy
```

Or test on a single image:
```bash
python test_single_image.py ../datasets/NIST4/F0001_01.bmp ../test_output
```

## Alternative: Docker

If conda doesn't work, use Docker:

```dockerfile
FROM python:3.7-slim

WORKDIR /app
COPY FingerNet/ /app/FingerNet/

RUN pip install tensorflow==1.15.5 keras==2.2.4 \
    numpy==1.16.6 scipy==1.2.3 opencv-python==4.2.0.32 \
    matplotlib==3.1.3 h5py==2.10.0 Pillow==6.2.2

WORKDIR /app/FingerNet/src
CMD ["python", "train_test_deploy.py", "0", "deploy"]
```

## Why Not Python 3.13?

- TensorFlow 1.x doesn't support Python 3.8+
- Keras 2.x doesn't support Python 3.8+
- Old numpy/scipy versions don't build on Python 3.13
- The code uses Python 2-style syntax in places

## Current Status

The venv_fingernet created with Python 3.13 **will NOT work** with the original FingerNet code due to incompatibility.

**Recommendation:** Use conda with Python 3.7 as shown above.

