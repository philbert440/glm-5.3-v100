# glm-5.3-v100

1Cat-vLLM engine snapshot for serving [GLM-5.3-Flash EXL3](https://huggingface.co/philbert440/GLM-5.3-Flash-EXL3)
(~2.1 bpw, SM70 tile order) on four V100-32GB GPUs, plus a prebuilt SM70 wheel.

The tree is the exact build that the model card's numbers were measured on: upstream 1Cat-vLLM main with the GLM-5.3
SM70 PRs applied. The PRs are drafts under review upstream. This repo pins the combination so you don't have to
rebuild it from the card's cherry-pick recipe.

- Model page: https://philtompkins.com/models/glm-5.3-flash
- Checkpoint: https://huggingface.co/philbert440/GLM-5.3-Flash-EXL3
- Upstream engine: https://github.com/1CatAI/1Cat-vLLM

## Exact base

| | |
|---|---|
| Base | 1Cat-vLLM `main` @ [`ec60883ad7187dc8aa29b3c30cae5a69e1cfa5cd`](https://github.com/1CatAI/1Cat-vLLM/commit/ec60883ad7187dc8aa29b3c30cae5a69e1cfa5cd) (`v1.5.1-1486-gec60883ad`, 2026-10-10) |
| Cherry-picked | [#1125](https://github.com/1CatAI/1Cat-vLLM/pull/1125) at `587fac6eb` + `da13d2d9c` (contains [#1092](https://github.com/1CatAI/1Cat-vLLM/pull/1092)), [#1159](https://github.com/1CatAI/1Cat-vLLM/pull/1159) at `44d292046`, [#1161](https://github.com/1CatAI/1Cat-vLLM/pull/1161) at `a66fcda49`, [#1166](https://github.com/1CatAI/1Cat-vLLM/pull/1166) at `74d2cf6da`, [#1165](https://github.com/1CatAI/1Cat-vLLM/pull/1165) at `60dd820a7` |
| Merged | [#1102](https://github.com/1CatAI/1Cat-vLLM/pull/1102) at `b96d43e5c` (conflicts in `CMakeLists.txt` and `csrc/ops.h`, both sides kept) |
| Already in base | [#1091](https://github.com/1CatAI/1Cat-vLLM/pull/1091), MTP speculator wiring (merged upstream) |
| Snapshot tag | [`glm53-card-20261010`](../../tree/glm53-card-20261010): source tree `b321701ed88fc9882ae7c71cdb5968a6635505ab` |

The history here is linear. The first commit imports the upstream tree at `ec60883ad` unmodified, without upstream
history. Each PR commit after it keeps its original author and message. #1102 is a single squashed commit. The tagged
tree is byte-identical to the card build. To check it against a clone of the upstream repo, run the card's build steps
and compare `git rev-parse HEAD^{tree}` with the hash above. The commit after the tag only adds this README and
`v100/`; the upstream README moved to [README.1Cat-vLLM.md](README.1Cat-vLLM.md).

Why cherry-picks: #1125, #1159, #1161, #1166 and #1165 are rebased onto a newer `main` that fails at MTP draft-model
init. Merging their branches onto the pinned base pulls that history in, so the pin does nothing. Their own commits
apply cleanly on `ec60883ad`. #1102 predates the regression, so it merges.

## Install (prebuilt wheel)

Requirements: Linux x86_64 (glibc 2.39 or newer), Python 3.12, NVIDIA driver with CUDA 12.8 support, the CUDA 12.8
toolkit, a C++ compiler, and `ninja` for the TileLang kernels that still JIT-compile at startup.

```bash
# from https://github.com/philbert440/glm-5.3-v100/releases/tag/glm53-card-20261010
sha256sum -c SHA256SUMS
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python ./WHEEL_NAME ninja --torch-backend=cu128
```

The wheel is built from the tagged tree with `TORCH_CUDA_ARCH_LIST=7.0`, CUDA 12.8.1, gcc 14 and torch 2.10.0+cu128,
on Ubuntu 24.04. It runs on V100 (SM70) only.

Check it:

```bash
.venv/bin/python -c "import vllm, torch, flash_attn_v100; print(vllm.__version__, torch.__version__, torch.cuda.get_device_capability(0))"
```

The kernel tests (`tests/kernels/quantization/test_exl3_moe.py`, `tests/models/glm5next/test_sm70_sparse.py`; 80 pass
on the card build) run from a source build. They need `pytest tblib`, which the build requirements don't install.

## Build from source

```bash
git clone --branch glm53-card-20261010 https://github.com/philbert440/glm-5.3-v100 && cd glm-5.3-v100
uv venv --python 3.12 --seed
uv pip install -r requirements/build/cuda.txt --torch-backend=cu128
TORCH_CUDA_ARCH_LIST=7.0 CMAKE_CUDA_ARCHITECTURES=70 MAX_JOBS=10 \
CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14 \
  uv pip install -e . --torch-backend=cu128 --no-build-isolation
```

The full build takes 40-60 minutes.

## Launch

Four V100-32GB in one box, all visible to one process. NVLink between the four is the measured setup. If your GPUs
span cross-socket PCIe, check that peer-to-peer copies are actually correct. On the test box they silently returned
garbage between non-NVLinked pairs, and `NCCL_P2P_LEVEL=NVL` fixed it.

```bash
hf download philbert440/GLM-5.3-Flash-EXL3 --local-dir ./GLM-5.3-Flash-EXL3
GLM53_MODEL=./GLM-5.3-Flash-EXL3 ./v100/serve_glm53.sh
```

[`v100/serve_glm53.sh`](v100/serve_glm53.sh) is the card's serve command: TP4, MTP with 3 draft tokens, FP8 E4M3 KV,
131,072 context, `--gpu-memory-utilization 0.92`, `--max-num-seqs 1`. It reads `GLM53_MODEL`, `GLM53_VENV`,
`GLM53_SEQS`, `HOST`, `PORT` and `CUDA_VISIBLE_DEVICES`, and passes extra arguments through.

- Keep `--max-num-seqs` set. vLLM's default of 256 makes the MLA prefill workspace too big and the server will not start
  at 131K. `GLM53_SEQS=4` still fits.
- `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` (set by the script) is needed for KV to fit at 131K. Weights take about
  25.4 GiB per GPU and the KV pool about 1.3 GiB.
- The startup log should show `EXL3 routed experts: mcg, K=2 ... colmajor tile order` and
  `GLM-5.3-Flash route: SM70 FP16 sparse MLA with packed E4M3FN KV and FP32-softmax decode/verify`.

On the measured NVLink quad this build does 91.4-91.9 tok/s single stream, 32.4 ms per forward pass (512-token greedy
responses, re-verified from a clean clone). Under MTP, a single prompt's tok/s depends on its text, so compare builds
by milliseconds per pass.

Eight GPUs: TP4 × PP2 (`VLLM_PP_LAYER_PARTITION=24,21`, `NCCL_P2P_LEVEL=NVL`, `--pipeline-parallel-size 2`,
`--max-model-len 1048576`, `--max-num-seqs 16`, no `--speculative-config`) serves 1M context and measured 175.1 tok/s aggregate at eight
streams. MTP is unavailable with PP > 1, so single stream drops to 39.5 tok/s. TP8 doesn't work (the EXL3 MoE needs
per-rank intermediate size divisible by 512), and expert parallelism is refused.

## Credits

- [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM): the SM70 vLLM fork and its GLM-5.3 path, and for
  reviewing and merging #1091. Built on [vLLM](https://github.com/vllm-project/vllm).
- [ai-bond/flash-attention-v100](https://github.com/ai-bond/flash-attention-v100): the base the FP32 sparse MLA kernel was
  forked from.
- turboderp for [exllamav3](https://github.com/turboderp-org/exllamav3) (MIT), and Frenchy2k1 for the SM70 exllamav3 fork
  the conversion started from.
- Z.ai for GLM-5.3-Flash.

## License

Apache-2.0, as upstream: see [LICENSE](LICENSE). Vendored components keep their own licenses and notices.
