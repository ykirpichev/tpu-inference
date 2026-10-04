#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Text-only Qwen/Qwen3.8-27B BF16 on one v6e-4 host (tensor parallel 4).
# The weights are about 67 GB. One v6e chip is 32 GB, so v6e-1 cannot hold them.
# This is not the Qwen3.5-397B server: no FP8 KV, no expert parallel, no
# data-parallel attention, no reasoning parser, no MTP, no YaRN.
#
# MODEL_IMPL_TYPE stays at the default "auto". Qwen3_5ForConditionalGeneration
# is in _VLLM_PREFERRED_ARCHITECTURES, so auto selects the vLLM implementation.
set -euo pipefail

PORT="${PORT:-8000}"

args=(
  Qwen/Qwen3.8-27B
  --tensor-parallel-size=4
  --seed=42
  --language-model-only
  --limit-mm-per-prompt='{"image": 0, "video": 0}'
  --max-model-len=4096
  --max-num-seqs=8
  --max-num-batched-tokens=1024
  --gpu-memory-utilization=0.90
  --no-enable-prefix-caching
  --port="${PORT}"
)

vllm serve "${args[@]}"
