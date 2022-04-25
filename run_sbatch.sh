#!/usr/bin/env bash
#SBATCH --mem=45G
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -t 1-00:00:00
#SBATCH -o slurm_logs/slurm-%j.out

source conv-qa/bin/activate
wandb agent shriya/adversarial/<sweep-name>