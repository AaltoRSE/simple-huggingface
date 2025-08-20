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

Nodes=$SLURM_NNODES
c=fe
export MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

cd $wd

echo "Starting FSDP training job with:"
echo "  Nodes: $Nodes"
echo "  GPUs per node: $SLURM_GPUS_PER_NODE"
echo "  Working directory: $wd"

srun --jobid $SLURM_JOBID\
     -N $((Nodes)) \
     -n $((Nodes)) \
     -c $((8*7)) \
    --cpu-bind=mask_cpu:$MYMASKS \
    --gpus $(($Nodes*8)) \
    singularity exec \
        -B "$wd:/workdir" \
        $SIFPYTORCH /workdir/run.sh