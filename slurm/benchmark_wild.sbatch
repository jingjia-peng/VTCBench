#!/bin/bash
#SBATCH -o logs/%x/%A_%a.out
#SBATCH -e logs/%x/%A_%a.err
#SBATCH -p a100
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -t 30-00:00:00
#SBATCH -J VTCBench-wild
#SBATCH --ntasks=1
#SBATCH --array=0-2

set -x

export PYTHON_UNBUFFERED=1

SPLITS=()
SPLITS+=("Retrieval")
SPLITS+=("Reasoning")
SPLITS+=("Memory")

split=${SPLITS[$SLURM_ARRAY_TASK_ID]}

uv run examples/run_wild.py \
    --model config/model/gemini.json \
    --data.path MLLM-CL/VTCBench \
    --data.split $split \
    --data.pure_text true \
    --run.num_tasks 0 \
    --run.result_dir results_wild/$split \
    --run.api_cache_dir .cache/wild \
    $@
