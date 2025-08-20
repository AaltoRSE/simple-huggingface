#!/bin/bash -e
# Make sure GPUs are up
echo "Rank $SLURM_PROCID - $(taskset -p $$) $ROCR_VISIBLE_DEVICES"
# Make sure GPUs are up, this seems to sometimes be necessary on lumi... 
if [ $SLURM_LOCALID -eq 0 ] ; then
    rocm-smi 
fi
sleep 2

export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

# Set MIOpen cache to a temporary folder.
if [ $SLURM_LOCALID -eq 0 ] ; then
    rm -rf $MIOPEN_USER_DB_PATH
    mkdir -p $MIOPEN_USER_DB_PATH
fi
sleep 2

# Report affinity
echo "Rank $SLURM_PROCID --> $(taskset -p $$)"

# ROCm/HIP environment setup
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HSA_ENABLE_SDMA=0
# export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb

# RCCL/NCCL configuration for AMD GPUs
export RCCL_DEBUG=WARN
export NCCL_DEBUG=WARN
export RCCL_ENABLE_P2P=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=PHB
export RCCL_MSCCL_ENABLE=0

# Set NCCL timeout and debugging variables - INCREASED TIMEOUT
export NCCL_TIMEOUT=1800  # 1 hour instead of 30 minutes
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1  # Changed to 1 for better error handling
# export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600  # Add heartbeat timeout

# # Add RNG synchronization fix
# export PYTHONHASHSEED=42  # Fixed hash seed for reproducibility
# export TORCH_NCCL_TRACE_BUFFER_SIZE=2048  # Enable trace buffer for debugging

# Set interfaces to be used by RCCL.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3

# Remove CUDA-specific variables that can cause issues with ROCm
unset CUDA_LAUNCH_BLOCKING
unset HIP_LAUNCH_BLOCKING

export LOGLEVEL=INFO

# Set environment for the app
# Get the master node address from SLURM_NODELIST
export MASTER_ADDR=$(/runscripts/get-master "$SLURM_NODELIST")
export MASTER_PORT=6001
export WORLD_SIZE=$SLURM_NPROCS
export RUNID="34567"
export RANK=$SLURM_PROCID

echo "Master address: $MASTER_ADDR"
echo "Rank: $RANK"
echo "World size: $WORLD_SIZE"
echo "Nodes: $SLURM_NODELIST"
echo "Visible devices: $ROCR_VISIBLE_DEVICES"

# use the project directory as Huggingface cache folder
export HF_HOME=/workdir

# export TRANSFORMERS_OFFLINE=0
# export HF_DATASETS_OFFLINE=0
# echo 
set -x
export OMP_NUM_THREADS=8
echo "SLURM Process ID: $SLURM_PROCID"

# Set Accelerate configuration file
export ACCELERATE_CONFIG_FILE="/workdir/fsdp_config.yaml"

NUM_PROCESSES=$(expr $SLURM_NNODES \* 8)

export LAUNCHER="accelerate launch \
    --config_file $ACCELERATE_CONFIG_FILE \
    --machine_rank $SLURM_NODEID \
    --num_machines $SLURM_NNODES \
    --num_processes $NUM_PROCESSES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT"

# Training script command
export CMD="/workdir/main.py \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --dataset_name smangrul/code-chat-assistant-v1 \
    --dataset_splits train,test \
    --batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --warmup_ratio 0.1 \
    --max_length 1024 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --mixed_precision bf16 \
    --logging_steps 25 \
    --save_steps 250 \
    --output_dir /workdir/outputs/llama2-7b-code-chat-lora \
    --max_grad_norm 0.3"

echo "Launching FSDP training with Accelerate..."
echo "Master address: $MASTER_ADDR"
echo "Command: $LAUNCHER $CMD"

$LAUNCHER $CMD 2>&1