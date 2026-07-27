#!/usr/bin/env bash
set -euo pipefail

cd /root/LLMServingSim
mkdir -p B200_profile/logs B200_profile/profiles
status_file="B200_profile/logs/qwen-skew.status"
log_file="B200_profile/logs/qwen-skew.log"
rm -f "$status_file"

finish() {
  rc=$?
  printf '%s\n' "$rc" >"$status_file"
  printf 'finished_at_utc=%s exit_code=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >>"$log_file"
}
trap finish EXIT

printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$log_file"
printf 'hard_runtime_limit=60m; grid_factors=8.0\n' | tee -a "$log_file"

timeout --signal=TERM --kill-after=30s 60m \
  python3 -m profiler profile Qwen/Qwen3-235B-A22B \
    --hardware B200 \
    --tp 1,2,4 \
    --dtype bfloat16 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 256 \
    --attention-max-kv 16384 \
    --attention-chunk-factor 4.0 \
    --attention-kv-factor 4.0 \
    --measurement-iterations 3 \
    --skew-n-factor 8.0 \
    --skew-pc-factor 8.0 \
    --skew-kp-factor 8.0 \
    --skew-kvs-factor 8.0 \
    --only-skew \
    --out-root B200_profile/profiles \
    2>&1 | tee -a "$log_file"

python3 check_profile.py \
  B200_profile/profiles/B200/Qwen/Qwen3-235B-A22B/bf16 \
  --model-id Qwen/Qwen3-235B-A22B \
  --hardware B200 \
  --tp 1,2,4 \
  --moe \
  --skew \
  --expected-vllm-version 0.19.0 \
  2>&1 | tee -a "$log_file"
