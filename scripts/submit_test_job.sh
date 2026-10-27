#!/bin/bash
#SBATCH --job-name=injections
#SBATCH --account=ctb-vkaspi
#SBATCH --array=0-1                   #len pointings // 50 = 2 batches
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G              
#SBATCH --time=00:10:00
#SBATCH --output=logs/inject_%A_%a.out
#SBATCH --error=logs/inject_%A_%a.err

mkdir -p logs
module load apptainer
apptainer run /project/ctb-vkaspi/champss/apptainer/champss_software_20261001.sif python3 run_stack_injections.py
