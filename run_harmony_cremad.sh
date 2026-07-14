#!/bin/bash
#SBATCH --job-name=harmony_cremad
#SBATCH --output=logs/harmony_cremad_%j.out
#SBATCH --error=logs/harmony_cremad_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=all-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
echo "Job started: $(date)"
echo "EXPERIMENT: Harmony CREMA-D | LR=0.001 | Seed=$1 | Fold=1"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd ~/mosaic
/mmfs2/home/jacks.local/joy.okolo/mosaic/mosaic_env/bin/python experiments/run_harmony_cremad.py --seed $1 --lr 0.001 --fold 1
echo "Job finished: $(date)"
