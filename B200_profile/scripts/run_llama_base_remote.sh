#!/usr/bin/env bash
set -euo pipefail

cd /root/LLMServingSim
mkdir -p B200_profile/logs B200_profile/profiles
status_file="B200_profile/logs/llama405b.status"
log_file="B200_profile/logs/llama405b.log"
rm -f "$status_file"

finish() {
  rc=$?
  printf '%s\n' "$rc" >"$status_file"
  printf 'finished_at_utc=%s exit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >>"$log_file"
}
trap finish EXIT

printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$log_file"
python3 -c 'import pandas,torch,vllm; print(f"pandas={pandas.__version__} torch={torch.__version__} torch_cuda={torch.version.cuda} vllm={vllm.__version__} gpu={torch.cuda.get_device_name(0)}")' | tee -a "$log_file"

python3 -m profiler profile meta-llama/Llama-3.1-405B-Instruct \
  --hardware B200 \
  --tp 1,2,4,8 \
  --dtype bfloat16 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 256 \
  --attention-max-kv 16384 \
  --attention-chunk-factor 4.0 \
  --attention-kv-factor 4.0 \
  --measurement-iterations 3 \
  --skip-skew \
  --out-root B200_profile/profiles \
  2>&1 | tee -a "$log_file"

python3 check_profile.py \
  B200_profile/profiles/B200/meta-llama/Llama-3.1-405B-Instruct/bf16 \
  --model-id meta-llama/Llama-3.1-405B-Instruct \
  --hardware B200 \
  --tp 1,2,4,8 \
  --expected-vllm-version 0.19.0 \
  2>&1 | tee -a "$log_file"
