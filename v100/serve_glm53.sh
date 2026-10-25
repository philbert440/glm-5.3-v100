#!/bin/bash
# Serve GLM-5.3-Flash EXL3 on 4x V100-32GB, TP4, MTP k=3, FP8 KV, 131K context (the model card's command).
#   GLM53_MODEL   local checkpoint dir (required): hf download philbert440/GLM-5.3-Flash-EXL3 --local-dir ...
#   GLM53_VENV    venv with this repo's wheel (default ./.venv)
#   GLM53_SEQS    --max-num-seqs (default 1; 4 still fits at 131K). Never leave it at vLLM's default.
#   HOST / PORT   bind address (default 127.0.0.1:8000)
# Extra arguments go to the server after these and override them.
set -euo pipefail
model=${GLM53_MODEL:?set GLM53_MODEL to the checkpoint directory}
venv=${GLM53_VENV:-./.venv}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0   # else 1.6 GiB is reserved for graph capture and no KV fits at 131K
export PATH="$venv/bin:$PATH"
exec "$venv/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$model" --served-model-name glm-5.3-flash \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --trust-remote-code --dtype half --tensor-parallel-size 4 \
  --kv-cache-dtype fp8_e4m3 --max-model-len 131072 --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 4096 --max-num-seqs "${GLM53_SEQS:-1}" \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"greedy"}' \
  --reasoning-parser deepseek_r1 --tool-call-parser glm47 --enable-auto-tool-choice "$@"
