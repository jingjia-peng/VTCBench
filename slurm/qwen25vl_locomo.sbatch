#!/bin/bash
#SBATCH -o logs/%x/%A_%a.out
#SBATCH -e logs/%x/%A_%a.err
#SBATCH -p a30
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -t 30-00:00:00
#SBATCH -J Qwen2.5-VL-7B-Instruct-LoCoMo
#SBATCH --ntasks=1
#SBATCH --array=0-1%1

set -x

DATASETS=(
    data/LoCoMo/needlesets/1_MultiHop.json
    data/LoCoMo/needlesets/2_Temporal.json
    data/LoCoMo/needlesets/3_OpenDomain.json
    data/LoCoMo/needlesets/4_SingleHop.json
)

# smoke test
if nc -z localhost 1026; then
    curl http://localhost:1026/v1/models
else
    echo 'model server not running, exit'
    exit 1
fi

# if it is ready
export PYTHON_UNBUFFERED=1

export LIBRARY_PATH=$HOME/.local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

MODELID=$SLURM_JOB_NAME

# based on some common configs shared across a family of models
# and modify for specific model
uv run examples/run.py \
    --model config/model/qwen_2.5_vl_7b.json \
    --data config/data/locomo.json \
    --data.needle_set_path ${DATASETS[$SLURM_ARRAY_TASK_ID]} \
    --render config/render/locomo.yml \
