#!/bin/bash
#SBATCH --job-name=ablation_b_no_anchor
#SBATCH --output=logs/ablation_b_no_anchor_%j.out
#SBATCH --error=logs/ablation_b_no_anchor_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=all-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

module load python/3.12
module load cuda/12.6
conda activate mosaic

echo "Job started: $(date)"
echo "EXPERIMENT: Ablation B - MCAT without CrossModalAnchor (lambda_anchor=0.0) | Seed=$1"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd ~/mosaic
python experiments/run_mosaic.py --seed $1 --lambda_anchor 0.0

echo "Job finished: $(date)"
