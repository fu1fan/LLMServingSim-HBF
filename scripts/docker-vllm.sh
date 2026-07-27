#!/bin/bash

# Launch vLLM Docker for profiler / bench / validate.
#
# Mounts the LLMServingSim repo root as /workspace so the profiler,
# bench, datasets generators, and shared model configs are all visible:
#
#     /workspace/profiler/            profiler package + scripts
#     /workspace/bench/               bench + validate
#     /workspace/workloads/            workload JSONLs and generators
#     /workspace/configs/model/       HF model configs
#
# The working directory defaults to /workspace so any of the modules
# can be run via ``python -m profiler``, ``python -m bench``, etc.
#
# The official vllm/vllm-openai image already provides vllm, pydantic,
# pyyaml, rich, and huggingface_hub. The current profiler also imports
# pandas while writing skew-fit metadata, including skip-skew runs.

set -euo pipefail

# Resolve the repo root regardless of where this script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../scripts
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                    # .../LLMServingSim
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.19.0}"

docker run --name vllm_docker \
  --gpus all \
  -it \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$REPO_ROOT":/workspace \
  --volume "$HOME/.cache/huggingface":/root/.cache/huggingface \
  --shm-size=16g \
  -w /workspace \
  --entrypoint /bin/bash \
  "$VLLM_IMAGE" \
  -c "pip install 'pandas>=2.2,<3' datasets matplotlib && exec bash"
