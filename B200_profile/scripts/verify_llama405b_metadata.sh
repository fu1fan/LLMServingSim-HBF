#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_ID="meta-llama/Llama-3.1-405B-Instruct"
MODEL_DIR="$REPO_ROOT/models/Llama-3.1-405B-Instruct"
PROFILER_CONFIG="$REPO_ROOT/configs/model/meta-llama/Llama-3.1-405B-Instruct.json"
VERIFICATION_RECORD="$MODEL_DIR/OFFICIAL_VERIFICATION.txt"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required and must have access to $MODEL_ID." >&2
  exit 2
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

for filename in config.json tokenizer_config.json generation_config.json; do
  curl -fsSL \
    -H "Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/$MODEL_ID/resolve/main/$filename" \
    -o "$tmp_dir/$filename"
done

python3 - "$MODEL_DIR/config.json" "$tmp_dir/config.json" <<'PY'
import json
import sys

local_path, official_path = sys.argv[1:]
with open(local_path, encoding="utf-8") as handle:
    local = json.load(handle)
with open(official_path, encoding="utf-8") as handle:
    official = json.load(handle)

required = (
    "architectures",
    "model_type",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "max_position_embeddings",
    "rope_scaling",
)
differences = [
    f"{key}: local={local.get(key)!r}, official={official.get(key)!r}"
    for key in required
    if local.get(key) != official.get(key)
]
local_dtype = local.get("torch_dtype", local.get("dtype"))
official_dtype = official.get("torch_dtype", official.get("dtype"))
if str(local_dtype).lower() not in {"bfloat16", "bf16"}:
    differences.append(f"local dtype is not BF16: {local_dtype!r}")
if str(official_dtype).lower() not in {"bfloat16", "bf16"}:
    differences.append(f"official dtype is not BF16: {official_dtype!r}")

if differences:
    print("Official Llama config does not match the prepared profiling shape:", file=sys.stderr)
    for difference in differences:
        print(f"  - {difference}", file=sys.stderr)
    raise SystemExit(1)
PY

cp "$tmp_dir/config.json" "$MODEL_DIR/config.json"
cp "$tmp_dir/tokenizer_config.json" "$MODEL_DIR/tokenizer_config.json"
cp "$tmp_dir/generation_config.json" "$MODEL_DIR/generation_config.json"
cp "$tmp_dir/config.json" "$PROFILER_CONFIG"

{
  echo "model_id=$MODEL_ID"
  echo "verified_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "source=https://huggingface.co/$MODEL_ID/resolve/main"
  shasum -a 256 \
    "$MODEL_DIR/config.json" \
    "$MODEL_DIR/tokenizer_config.json" \
    "$MODEL_DIR/generation_config.json"
} >"$VERIFICATION_RECORD"

echo "PASS: official gated Llama metadata verified and refreshed."
echo "Record: $VERIFICATION_RECORD"
