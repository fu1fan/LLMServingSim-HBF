#!/usr/bin/env bash
set -euo pipefail

cd /root/LLMServingSim
llama_status="B200_profile/logs/llama405b.status"
qwen_status="B200_profile/logs/qwen235b.status"
chain_log="B200_profile/logs/chain-qwen.log"

printf 'watch_started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$chain_log"
while [[ ! -f "$llama_status" ]]; do
  sleep 10
done

rc="$(tr -d '[:space:]' <"$llama_status")"
if [[ "$rc" != "0" ]]; then
  printf 'llama_failed_at_utc=%s exit_code=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >>"$chain_log"
  exit 1
fi

if pgrep -f "python3 -m profiler profile Qwen/Qwen3-235B-A22B" >/dev/null; then
  printf 'qwen_already_running_at_utc=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$chain_log"
  exit 0
fi
if [[ -f "$qwen_status" ]] && [[ "$(tr -d '[:space:]' <"$qwen_status")" == "0" ]]; then
  printf 'qwen_already_complete_at_utc=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$chain_log"
  exit 0
fi

printf 'qwen_started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$chain_log"
nohup /root/run_qwen_base_remote.sh >/root/qwen-driver.log 2>&1 &
printf 'qwen_driver_pid=%s\n' "$!" >>"$chain_log"
