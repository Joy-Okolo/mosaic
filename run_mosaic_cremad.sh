#!/bin/bash
#SBATCH --job-name=mosaic_cremad
#SBATCH --output=logs/mosaic_cremad_%j.out
#SBATCH --error=logs/mosaic_cremad_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=all-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
echo "Job started: $(date)"
echo "EXPERIMENT: MOSAIC CREMA-D | LR=0.001 | Seed=$1 | Fold=1"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd ~/mosaic
/mmfs2/home/jacks.local/joy.okolo/mosaic/mosaic_env/bin/python experiments/run_mosaic_cremad.py --seed $1 --lr 0.001 --fold 1
echo "Job finished: $(date)"
