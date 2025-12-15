#!/bin/bash
#SBATCH -o logs/%x/%A_%a.out
#SBATCH -e logs/%x/%A_%a.err
#SBATCH -p a100
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -t 30-00:00:00
#SBATCH -J VTC-Retrieval
#SBATCH --ntasks=1
#SBATCH --array=0-5%1

# get id from slurm, 0-5
# if you are not from slurm, write a script that export it, like
# for i in $(seq 0 5); do
#     export SLURM_ARRAY_TASK_ID=${i}
#     export SLURM_JOB_NAME="Qwen2.5-VL-7B-Instruct"
#     # run in background, and dump stdout/err to files for logging
#     echo "call this script to run in background" > ${SLURM_JOB_NAME}_${i}.out 2> ${SLURM_JOB_NAME}_${i}.err &
# done

# export SLURM_ARRAY_TASK_ID=0
task_id=${SLURM_ARRAY_TASK_ID}

# one to one
LEN_ARR=(
    1000
    2000
    4000
    8000
    16000
    32000
)
RENDER_CONFIG=(
    config/render/default.yml
    config/render/default.yml
    config/render/default.yml
    config/render/default.yml
    config/render/32k.yml
    config/render/32k.yml
)

set -x

# for this job, get length, e.g. 1000 for 1st job
LEN=${LEN_ARR[$task_id]}
RENDER=${RENDER_CONFIG[$task_id]}

# smoke test if the llm server is alive
# if nc -z localhost 1026; then
#     curl http://localhost:1025/v1/models
# else
#     echo 'model server not running, exit'
#     exit 1
# fi

export PYTHON_UNBUFFERED=1

# playwright must have libasound2 before running
# this is a fix if libasound.so.2 not in system path, but $HOME/.local/lib/libasound.so.2
# avoids `sudo apt install libasound2`
# we dont have videos in our data, so a fake file is fine
export LIBRARY_PATH=$HOME/.local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

# we specified `#SBATCH -J Qwen2.5-VL-72B-Instruct` at line9
# so this is `MODELID=Qwen2.5-VL-72B-Instruct`
MODELID=$SLURM_JOB_NAME

# based on some common configs shared across a family of models
# and modify for specific model
# make sure these files exist
# - config/model/qwen_2.5_vl_7b.json
# - config/data/nolima.json
# - config/render/default.yml
NEEDLES=(
    "data/RULER/needlesets/s_niah.json"
    "data/RULER/needlesets/mk_niah.json"
    "data/RULER/needlesets/mv_niah.json"
    "data/RULER/needlesets/mq_niah.json"
)

for NEEDLE_PATH in "${NEEDLES[@]}"; do
    uv run examples/run.py \
    --model config/model/qwen_2.5_vl_72b.json \
        --data config/data/ruler.json \
        --data.needle_set_path "[$NEEDLE_PATH]" \
        --render ${RENDER} \
        --data.context_length ${LEN} \
        --run.num_tasks 10 \
        $@
done

# you can specify needle set from cmdline, e.g.
# sbatch xxx.sbatch --data.needle_set_path data/RULER/needlesets/mv_niah.json \
