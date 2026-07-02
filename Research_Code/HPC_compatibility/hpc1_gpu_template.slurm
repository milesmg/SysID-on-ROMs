#!/bin/bash
# This template requests a GPU node, but the current Julia training code is CPU/SciML based.
# Use it only after checking hpc1's exact GPU partition/GRES names and after porting the RHS/AD path to CUDA arrays.
#
#SBATCH -J ac_gpu_template
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --cpus-per-task=18
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -t 23:59:00
#SBATCH -o Research_Code/HPC_compatibility/logs/%x-%j.out
#SBATCH -e Research_Code/HPC_compatibility/logs/%x-%j.err

set -euo pipefail

echo "This is a GPU allocation template only."
echo "Current run_fom_hpc.jl and run_rom_hpc.jl do not move arrays/models to CUDA, so a GPU will not speed them up yet."
echo "Prefer hpc1_run_fom.slurm or hpc1_run_rom.slurm on CPU nodes for the current code."
