# Hướng dẫn Convert FingerNet Weights từ TensorFlow sang JAX

File `Model.model` là format custom của TensorFlow 1.x / Keras 2.x, không thể load trực tiếp bằng Keras 3. Cần 2 bước:

## Bước 1: Export weights từ TF1/Keras2 environment

### Cách 1: Dùng Docker (Khuyến nghị)

```bash
# Pull TensorFlow 1.x image
docker pull tensorflow/tensorflow:1.15.5-gpu-py3

# Chạy container và mount thư mục
docker run -it --rm \
  -v /Users/nguyenthanhlam/SSL_Correspondence/IFViT:/workspace \
  tensorflow/tensorflow:1.15.5-gpu-py3 bash

# Trong container
cd /workspace/fingernet_jax
pip install numpy scipy
python export_fingernet_weights_tf1.py
```

### Cách 2: Dùng Conda

```bash
# Tạo environment TF1
conda create -n tf1_env python=3.7
conda activate tf1_env
conda install tensorflow=1.15 keras=2.3.1 numpy scipy

# Chạy export script
cd /Users/nguyenthanhlam/SSL_Correspondence/IFViT/fingernet_jax
python export_fingernet_weights_tf1.py
```

### Cách 3: Dùng Kaggle Notebook

1. Upload `export_fingernet_weights_tf1.py` và `FingerNet/` folder lên Kaggle
2. Tạo notebook với TF1 environment
3. Chạy script export
4. Download file `fingernet_keras_tf1_weights.npz`

## Bước 2: Convert sang Flax format

Sau khi có file `fingernet_keras_tf1_weights.npz`, chạy:

```bash
cd /Users/nguyenthanhlam/SSL_Correspondence/IFViT/fingernet_jax

# Nếu file .npz đã có trong thư mục
python convert_from_npz.py

# Hoặc chỉ định đường dẫn
python convert_from_npz.py /path/to/fingernet_keras_tf1_weights.npz
```

Kết quả: File `fingernet_flax_params.npz` sẽ được tạo ra.

## Bước 3: Sử dụng weights đã convert

```bash
# Test với weights đã convert
python deploy_jax.py ../FingerNet/datasets/NIST4/F0001_01.bmp ./test_output_jax ./fingernet_flax_params.npz
```

## Troubleshooting

### Lỗi: "File format not supported"
- **Nguyên nhân**: Đang dùng Keras 3, không load được `Model.model`
- **Giải pháp**: Phải export weights trong TF1/Keras2 environment trước

### Lỗi: "ModuleNotFoundError: No module named 'keras'"
- **Nguyên nhân**: Thiếu dependencies trong TF1 environment
- **Giải pháp**: `pip install tensorflow==1.15 keras==2.3.1`

### Lỗi: "Cannot find layer name"
- **Nguyên nhân**: Layer names không khớp giữa Keras và Flax
- **Giải pháp**: Kiểm tra `map_layer_name_keras_to_flax()` trong `convert_from_npz.py`

## Files

- `export_fingernet_weights_tf1.py` - Export weights từ TF1 (chạy trong TF1 env)
- `convert_from_npz.py` - Convert .npz sang Flax (chạy trong JAX env)
- `convert_keras_to_flax.py` - Script cũ (chỉ dùng nếu có thể load Model.model trực tiếp)

