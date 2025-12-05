# Quick Start: Convert FingerNet Weights

## Cách nhanh nhất (Docker - Tự động)

### Bước 1: Start Docker Desktop
- Mở Docker Desktop application trên macOS
- Đợi Docker daemon khởi động (icon Docker ở menu bar)

### Bước 2: Chạy script tự động

```bash
cd /Users/nguyenthanhlam/SSL_Correspondence/IFViT/fingernet_jax
./run_export_in_docker.sh
```

Script sẽ:
1. Tự động pull TensorFlow 1.15 Docker image (lần đầu)
2. Export weights từ `Model.model` → `fingernet_keras_tf1_weights.npz`
3. Báo kết quả

### Bước 3: Convert sang Flax

```bash
python convert_from_npz.py
```

Kết quả: `fingernet_flax_params.npz`

### Bước 4: Test với weights đã convert

```bash
python deploy_jax.py \
  ../FingerNet/datasets/NIST4/F0001_01.bmp \
  ./test_output_jax \
  ./fingernet_flax_params.npz
```

---

## Cách thủ công (nếu Docker không hoạt động)

### Option 1: Conda Environment

```bash
# Tạo environment
conda create -n tf1_env python=3.7 -y
conda activate tf1_env

# Cài đặt TF1
conda install tensorflow=1.15 keras=2.3.1 numpy scipy -y

# Export weights
cd /Users/nguyenthanhlam/SSL_Correspondence/IFViT/fingernet_jax
python export_fingernet_weights_tf1.py

# Deactivate và convert
conda deactivate
python convert_from_npz.py
```

### Option 2: Docker Manual

```bash
# Start Docker container
docker run -it --rm \
  -v /Users/nguyenthanhlam/SSL_Correspondence/IFViT:/workspace \
  tensorflow/tensorflow:1.15.5-gpu-py3 bash

# Trong container
cd /workspace/IFViT/fingernet_jax
pip install numpy scipy
python export_fingernet_weights_tf1.py
exit

# Ngoài container - convert
cd /Users/nguyenthanhlam/SSL_Correspondence/IFViT/fingernet_jax
python convert_from_npz.py
```

---

## Troubleshooting

### "Docker daemon is not running"
- Mở Docker Desktop application
- Đợi icon Docker xuất hiện ở menu bar
- Thử lại

### "Cannot connect to Docker daemon"
- Kiểm tra Docker Desktop đang chạy
- Thử restart Docker Desktop

### "File format not supported"
- Đảm bảo đang dùng TensorFlow 1.x (không phải TF2)
- Kiểm tra `Model.model` file có tồn tại không

### "ModuleNotFoundError"
- Trong Docker: `pip install numpy scipy`
- Trong Conda: `conda install numpy scipy`

---

## Files

- `run_export_in_docker.sh` - Script tự động (khuyến nghị)
- `export_fingernet_weights_tf1.py` - Export script (chạy trong TF1)
- `convert_from_npz.py` - Convert script (chạy trong JAX env)
- `convert_weights.sh` - Helper script

