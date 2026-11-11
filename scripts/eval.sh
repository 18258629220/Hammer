#!/usr/bin/env bash
# Example: PYTHON_BIN=/path/to/python CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh \
#   --weight /path/to/checkpoint --dataset PIADv1 --setting Seen --save_predictions
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
exec "${PYTHON_BIN:-python}" tools/eval.py "$@"
