#!/bin/bash
#SBATCH --job-name=fedavg_lowlr_1000
#SBATCH --output=logs/fedavg_lowlr_%j.out
#SBATCH --error=logs/fedavg_lowlr_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=all-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

module load python/3.12
module load cuda/12.6
conda activate mosaic

echo "Job started: $(date)"
echo "EXPERIMENT: FedAvg | LR=0.001 | Seed=$1"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd ~/mosaic
python experiments/run_fedavg.py --seed $1 --lr 0.001

echo "Job finished: $(date)"
