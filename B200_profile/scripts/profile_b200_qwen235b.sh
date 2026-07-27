#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="Qwen/Qwen3-235B-A22B"
MODEL_CONFIG="$REPO_ROOT/configs/model/Qwen/Qwen3-235B-A22B.json"
HARDWARE="B200"
TP_DEGREES="1,2,4"
PROFILE_OUT_ROOT="${PROFILE_OUT_ROOT:-$REPO_ROOT/B200_profile/profiles}"
DELIVERY_ROOT="${B200_PROFILE_ROOT:-$REPO_ROOT/B200_profile}"
LOG_ROOT="$DELIVERY_ROOT/logs"
HARDWARE_INFO_ROOT="$DELIVERY_ROOT/hardware_info"
MIN_GPU_MEMORY_MIB="${MIN_GPU_MEMORY_MIB:-175000}"
EXPECTED_VLLM_VERSION="0.19.0"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$PROFILE_OUT_ROOT" "$LOG_ROOT" "$HARDWARE_INFO_ROOT"
cd "$REPO_ROOT"

if [[ ! -s "$MODEL_CONFIG" ]]; then
  echo "Missing model config: $MODEL_CONFIG" >&2
  exit 2
fi

cmd=(
  python3 -m profiler profile "$MODEL"
  --hardware "$HARDWARE"
  --tp "$TP_DEGREES"
  --dtype bfloat16
  --max-num-batched-tokens 2048
  --max-num-seqs 256
  --attention-max-kv 16384
  --attention-chunk-factor 4.0
  --attention-kv-factor 4.0
  --measurement-iterations 3
  --skip-skew
  --out-root "$PROFILE_OUT_ROOT"
)

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

command -v nvidia-smi >/dev/null || {
  echo "nvidia-smi is required inside the GPU container." >&2
  exit 2
}

gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "$gpu_count" != "1" ]]; then
  echo "Expected exactly one visible GPU, found $gpu_count." >&2
  exit 2
fi

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ "$gpu_name" != *B200* ]]; then
  echo "Expected an NVIDIA B200, found: $gpu_name" >&2
  exit 2
fi

memory_total_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if (( memory_total_mib < MIN_GPU_MEMORY_MIB )); then
  echo "Expected a full B200 memory allocation (>=${MIN_GPU_MEMORY_MIB} MiB), found ${memory_total_mib} MiB." >&2
  exit 2
fi

vllm_version="$(python3 -c 'import vllm; print(vllm.__version__)')"
if [[ "$vllm_version" != "$EXPECTED_VLLM_VERSION" ]]; then
  echo "Expected vLLM $EXPECTED_VLLM_VERSION, found $vllm_version." >&2
  exit 2
fi

compute_major="$(python3 -c 'import torch; print(torch.cuda.get_device_capability(0)[0])')"
if (( compute_major < 10 )); then
  echo "Expected Blackwell compute capability (major >= 10), found major=$compute_major." >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="$LOG_ROOT/qwen235b-${timestamp}.log"
nvidia-smi -q >"$HARDWARE_INFO_ROOT/nvidia-smi-q-${timestamp}.txt"
{
  nvidia-smi
  python3 -c 'import torch, vllm; print(f"torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}"); print(f"vllm={vllm.__version__}"); print(f"compute_capability={torch.cuda.get_device_capability(0)}")'
  if command -v nvcc >/dev/null; then
    nvcc --version
  fi
} >"$HARDWARE_INFO_ROOT/environment-${timestamp}.txt" 2>&1

"${cmd[@]}" 2>&1 | tee "$log_file"

profile_dir="$PROFILE_OUT_ROOT/B200/Qwen/Qwen3-235B-A22B/bf16"
python3 "$REPO_ROOT/check_profile.py" "$profile_dir" \
  --model-id "$MODEL" \
  --hardware B200 \
  --tp 1,2,4 \
  --moe \
  --expected-vllm-version "$EXPECTED_VLLM_VERSION" \
  2>&1 | tee -a "$log_file"
