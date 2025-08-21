#!/bin/bash
#SBATCH --job-name=fsdp-training
#SBATCH --account=project_462000365
#SBATCH --time=04:00:00
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=7
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --partition=standard-g
#SBATCH -o fsdp_training_%j.out
#SBATCH -e fsdp_training_%j.err

set -o pipefail
wd=$(realpath .)
module load CrayEnv
module load PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250404

export SINGULARITYENV_NCCL_DEBUG=INFO

cd $wd

echo "Starting FSDP training job with:"
echo "Num of nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Working directory: $wd"

srun --jobid $SLURM_JOBID\
     --nodes $SLURM_NNODES \
     --ntasks $SLURM_NNODES \
     --cpus-per-task $((8*7)) \
     --gpus-per-task 8 \
     singularity exec \
        --bind "$wd:/workdir" \
        $SIFPYTORCH /workdir/run.sh