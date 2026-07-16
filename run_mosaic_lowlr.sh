#!/bin/bash
#SBATCH --job-name=mosaic_with_dutycycle_lowlr
#SBATCH --output=logs/mosaic_with_dutycycle_lowlr_%j.out
#SBATCH --error=logs/mosaic_with_dutycycle_lowlr_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=all-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

module load python/3.12
module load cuda/12.6

echo "Job started: $(date)"
echo "EXPERIMENT: MOSAIC+MCAT | LR=0.001 | Seed=$1"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd ~/mosaic

/mmfs2/home/jacks.local/joy.okolo/mosaic/mosaic_env/bin/python experiments/run_mosaic.py --seed $1 --lr 0.001


echo "Job finished: $(date)"
