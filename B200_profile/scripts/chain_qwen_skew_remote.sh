#!/usr/bin/env bash
set -euo pipefail

cd /root/LLMServingSim
qwen_status="B200_profile/logs/qwen235b.status"
chain_log="B200_profile/logs/chain-qwen-skew.log"
latest_start_epoch="$(date -u -d '2026-07-26 15:35:00' +%s)"

printf 'watch_started_at_utc=%s latest_start_utc=2026-07-26T15:35:00Z\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$chain_log"
while [[ ! -f "$qwen_status" ]]; do
  sleep 10
done

rc="$(tr -d '[:space:]' <"$qwen_status")"
if [[ "$rc" != "0" ]]; then
  printf 'qwen_base_failed_at_utc=%s exit_code=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >>"$chain_log"
  exit 1
fi

now_epoch="$(date -u +%s)"
if (( now_epoch >= latest_start_epoch )); then
  printf 'skew_skipped_deadline_at_utc=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$chain_log"
  exit 0
fi

printf 'qwen_skew_started_at_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$chain_log"
nohup /root/run_qwen_skew_remote.sh >/root/qwen-skew-driver.log 2>&1 &
printf 'qwen_skew_driver_pid=%s\n' "$!" >>"$chain_log"
