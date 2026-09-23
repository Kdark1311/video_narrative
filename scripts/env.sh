# source scripts/env.sh trước khi chạy src/caption_vllm.py
C=$(python -c "import nvidia,os;print(os.path.join(nvidia.__path__[0],'cu13'))")
export CUDA_HOME=$C PATH=$C/bin:$PATH LD_LIBRARY_PATH=$C/lib:${LD_LIBRARY_PATH}
export VLLM_USE_DEEP_GEMM=0            # DeepGEMM không hỗ trợ SM 12.0
export VLLM_USE_FLASHINFER_SAMPLER=0   # sampler FlashInfer JIT lỗi trên SM 12.0; greedy nên không cần
