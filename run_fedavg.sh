#!/bin/bash
#SBATCH --job-name=fedavg_1000
#SBATCH --output=logs/fedavg_%j.out
#SBATCH --error=logs/fedavg_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

module load python/3.12
module load cuda/12.6
conda activate mosaic

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd ~/mosaic
python experiments/run_fedavg.py

echo "Job finished: $(date)"
