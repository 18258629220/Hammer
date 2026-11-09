#!/bin/bash
set -euo pipefail

# Usage: bash scripts/train.sh -d PIADv1 -p Seen -g 4 -b 64 -l 0.0001 -e 30 -n exp1 [-m hammer|partfield|crag3d] [-c CACHE] [-P|-s] [-- train.py args]

# Set the used cuda path; maybe unnecessary
export PATH=$HOME/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN_CODE=train.py
PYTHON_BIN=${PYTHON_BIN:-python}
cd "$ROOT_DIR"

DATASET="PIADv1"  # Options: PIADv1, PIADv2
SETTING="Seen"  # Options: Seen, Unseen_afford, Unseen_obj for PIADv2; Seen, Unseen for PIADv1
QUESTION_TYPE="afford_obj"  # Options: simple, afford, afford_obj

NUM_DEVICES=4
GLOBAL_BATCH_SIZE=64
BATCH_SIZE_PER_DEVICE=1
LR=0.0001
EPOCHS=30
NAME=exp1
MODEL_VARIANT=hammer
PARTFIELD_CACHE_DIR=""
PREFLIGHT=0
SMOKE_CHECK=0

while getopts ":d:p:g:b:l:e:n:m:c:Psh" opt; do
  case $opt in
    d)
      DATASET=$OPTARG
      ;;
    p)
      SETTING=$OPTARG
      ;;
    g)
      NUM_DEVICES=$OPTARG
      ;;
    b)
      GLOBAL_BATCH_SIZE=$OPTARG
      ;;
    l)
      LR=$OPTARG
      ;;
    e)
      EPOCHS=$OPTARG
      ;;
    n)
      NAME=$OPTARG
      ;;
    m)
      MODEL_VARIANT=$OPTARG
      ;;
    c)
      PARTFIELD_CACHE_DIR=$OPTARG
      ;;
    P)
      PREFLIGHT=1
      ;;
    s)
      SMOKE_CHECK=1
      ;;
    h)
      echo "Usage: $0 -d DATASET -p SETTING -g GPUS -b GLOBAL_BATCH -l LR -e EPOCHS -n NAME [-m hammer|partfield|crag3d] [-c CACHE] [-P|-s] [-- train.py args]"
      echo "-P: source/data preflight only; -s: one real training batch forward/backward"
      exit 0
      ;;
    :)
      echo "Missing argument for -$OPTARG" >&2
      exit 2
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 2
      ;;
  esac
done
shift $((OPTIND - 1))

for value in "$NUM_DEVICES" "$GLOBAL_BATCH_SIZE" "$EPOCHS"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "GPU count, global batch size and epochs must be positive integers" >&2
        exit 2
    fi
done
if (( GLOBAL_BATCH_SIZE % (BATCH_SIZE_PER_DEVICE * NUM_DEVICES) != 0 )); then
    echo "Global batch size must be divisible by microbatch size times GPU count" >&2
    exit 2
fi
case "$MODEL_VARIANT" in
    hammer|partfield|crag3d) ;;
    *) echo "Unknown model variant: $MODEL_VARIANT" >&2; exit 2 ;;
esac
case "$DATASET/$SETTING" in
    PIADv1/Seen|PIADv1/Unseen|PIADv2/Seen|PIADv2/Unseen_obj|PIADv2/Unseen_afford) ;;
    *) echo "Unsupported dataset/setting: $DATASET/$SETTING" >&2; exit 2 ;;
esac
if [[ ! "$NAME" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
    echo "Experiment name may only contain letters, digits, underscores, dots and hyphens" >&2
    exit 2
fi

if [ "$DATASET" == "PIADv2" ]; then
    STEPS_PER_EPOCH=$((37120 / GLOBAL_BATCH_SIZE))
    PRINT_FREQ=50
elif [ "$DATASET" == "PIADv1" ]; then
    STEPS_PER_EPOCH=$((12565 / GLOBAL_BATCH_SIZE))
    PRINT_FREQ=20
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi
if (( STEPS_PER_EPOCH < 1 )); then
    echo "Global batch size exceeds the dataset's samples per epoch" >&2
    exit 2
fi

PROJECT_NAME="HAMMER_${DATASET}_$(hostname)"
EXP_NAME="${DATASET}_${SETTING}_${QUESTION_TYPE}_${NAME}"
EXP_DIR=$ROOT_DIR/exp/${EXP_NAME}
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code

echo " =========> CREATE EXP DIR <========="
echo "Experiment dir: $EXP_DIR"
if [ -e "$CODE_DIR/train.py" ]; then
    echo "An experiment snapshot already exists: $CODE_DIR; choose a new -n name. Use --resume with the existing checkpoint in a new experiment." >&2
    exit 2
fi
mkdir -p "$MODEL_DIR" "$CODE_DIR"
cp -r scripts tools src "$CODE_DIR"
cp "$TRAIN_CODE" "$CODE_DIR"

export PYTHONPATH=$CODE_DIR:${PYTHONPATH:-}
echo "Runing code in: $CODE_DIR"

if [ "$NUM_DEVICES" -gt 0 ]; then
    list=$(seq -s, 0 $((NUM_DEVICES-1)))
else
    list=""
fi
INCLUDE="localhost:$list"
echo "Using devices: $INCLUDE"

echo " =========> RUN TASK <========="

# generate random number between 20000 and 25000
MASTER_PORT=$(( ( RANDOM % 5000 )  + 20000 ))

VERSION="Qwen/Qwen2.5-VL-3B-Instruct"
LORA_R=16
LORA_ALPHA=32
DATASET_DIR="$ROOT_DIR/data"

GRAD_ACCUMULATION_STEPS=$(( GLOBAL_BATCH_SIZE / (BATCH_SIZE_PER_DEVICE * NUM_DEVICES) ))
CE_LOSS_WEIGHT=1
MASK_LOSS_WEIGHT=2


TRAIN_ARGS=(
  --version="$VERSION"
  --lora_r="$LORA_R"
  --lora_alpha="$LORA_ALPHA"
  --dataset_dir="$DATASET_DIR"
  --log_base_dir="$ROOT_DIR/exp"
  --project_name="$PROJECT_NAME"
  --exp_name="$EXP_NAME"
  --lr="$LR"
  --epochs="$EPOCHS"
  --batch_size="$BATCH_SIZE_PER_DEVICE"
  --steps_per_epoch="$STEPS_PER_EPOCH"
  --grad_accumulation_steps="$GRAD_ACCUMULATION_STEPS"
  --ce_loss_weight="$CE_LOSS_WEIGHT"
  --mask_loss_weight="$MASK_LOSS_WEIGHT"
  --print_freq="$PRINT_FREQ"
  --dataset="$DATASET"
  --question_type="$QUESTION_TYPE"
  --setting="$SETTING"
  --model_variant="$MODEL_VARIANT"
)
if [ -n "$PARTFIELD_CACHE_DIR" ]; then
    # Resolve before launching code from the experiment snapshot.
    PARTFIELD_CACHE_DIR=$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$PARTFIELD_CACHE_DIR")
    TRAIN_ARGS+=(--partfield_cache_dir="$PARTFIELD_CACHE_DIR")
fi
if (( SMOKE_CHECK )); then
    TRAIN_ARGS+=(--smoke_check --debug)
fi
TRAIN_ARGS+=("$@")
if (( PREFLIGHT )); then
    exec "$PYTHON_BIN" "$CODE_DIR/$TRAIN_CODE" "${TRAIN_ARGS[@]}" --preflight
fi

"$PYTHON_BIN" -m deepspeed.launcher.runner \
  --include "$INCLUDE" \
  --master_port="$MASTER_PORT" "$CODE_DIR/$TRAIN_CODE" \
  "${TRAIN_ARGS[@]}"

