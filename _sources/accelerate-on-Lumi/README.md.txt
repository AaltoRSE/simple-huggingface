# LLM Training Environment Setup on LUMI Supercomputer

This guide provides step-by-step instructions for setting up and running Large Language Model (LLM) training on the LUMI supercomputer using Accelerate with Singularity containers.

## Prerequisites

- Access to LUMI supercomputer
- Basic familiarity with SLURM job scheduler
- Understanding of containerized environments

## Getting Started

### Step 1: Set Up Singularity Container with EasyBuild

#### 1.1 Configure Installation Paths
Set up the installation paths for EasyBuild by following the [LUMI EasyBuild preparation guide](https://docs.lumi-supercomputer.eu/software/installing/easybuild/#preparation-set-the-location-for-your-easybuild-installation) to specify where container and module files will be saved.

#### 1.2 Load Required Modules
Load the necessary modules for EasyBuild and Singularity:

```bash
module load LUMI partition/container EasyBuild-user
```

#### 1.3 Install PyTorch Container
Install the PyTorch Singularity container:

```bash
eb PyTorch-2.2.0-rocm-5.6.1-python-3.10-singularity-20240315.eb
```

> **Note**: Additional container options with different PyTorch, ROCm, or Python versions are available in the [LUMI PyTorch documentation](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/#singularity-containers-with-modules-for-binding-and-extras).

### Step 2: Install Additional Python Libraries (Optional)

If you need extra Python packages, extend the container with virtual environment support following the [container extension guide](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/#extending-the-containers-with-virtual-environment-support).

### Step 3: Submit Training Job

Submit your training job using the provided SLURM script:

```bash
sbatch submit.sh
```

## Files

This directory contains the following key files:

- **`submit.sh`** - SLURM job submission script
- **`run.sh`** - Script to set up distributed environment on LUMI and launch training
- **`main.py`** - Main entry point for model training
- **`utils.py`** - Helper functions and utilities
- **`fsdp_config.yaml`** - FSDP configuration for Hugging Face Accelerate

## Monitoring and Debugging

### Step 4: Monitor GPU Usage (Optional)

To monitor GPU utilization during training (useful for debugging performance issues):

1. Connect to your running job:
   ```bash
   srun --overlap --pty --jobid=YOUR_JOB_ID bash
   ```

2. Monitor GPU status in real-time:
   ```bash
   watch rocm-smi
   ```


