#!/bin/bash
#SBATCH --job-name=ablation_a_vanilla_fedavg_1000
#SBATCH --output=logs/ablation_a_vanilla_fedavg_%j.out
#SBATCH --error=logs/ablation_a_vanilla_fedavg_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=all-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

module load python/3.12
module load cuda/12.6
conda activate mosaic

echo "Job started: $(date)"
echo "EXPERIMENT: Ablation A - Vanilla FedAvg, random selection, no anchor loss (lambda_anchor=0.0) | Seed=$1"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd ~/mosaic
python experiments/run_fedavg.py --seed $1 --lambda_anchor 0.0 --lr 0.001

echo "Job finished: $(date)"
