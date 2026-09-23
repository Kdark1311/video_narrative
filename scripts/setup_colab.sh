#!/usr/bin/env bash
# Cài môi trường chạy vLLM + Qwen3-VL trên Colab GPU Blackwell (RTX PRO 6000) và tải dữ liệu.
# Dùng: bash scripts/setup_colab.sh [GDRIVE_URL]   rồi   source scripts/env.sh
set -e
pip install -q vllm gdown
# vllm kéo torch cu130 -> torchaudio phải cùng bản CUDA, nếu không transformers sẽ lỗi khi import
pip install -q --no-deps "torchaudio==2.11.0+cu130" --index-url https://download.pytorch.org/whl/cu130
# nvcc hệ thống của Colab là 12.8, GPU SM 12.x cần >= 12.9 để JIT kernel
pip install -q "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-cccl==13.0.*" "nvidia-cuda-crt==13.0.*"
C=$(python -c "import nvidia,os;print(os.path.join(nvidia.__path__[0],'cu13'))")
[ -e $C/lib64 ] || ln -s $C/lib $C/lib64
if [ -n "$1" ]; then
  mkdir -p /content/traffic && cd /content/traffic
  gdown --fuzzy "$1" && unzip -q -o *.zip -d data
fi
