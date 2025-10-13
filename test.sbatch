#!/bin/bash
#SBATCH -n 16
#SBATCH -p tupi
#SBATCH --time=20:00:00
WORKDIR=/home/users/grkremer/perf_2026


source ~/miniconda3/bin/activate
conda env list
conda env create -f $WORKDIR/environment.yml -n perf_2026
conda activate perf_2026

python train_and_eval.py