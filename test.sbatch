#!/bin/bash
#SBATCH -n 48
#SBATCH -p tupi
WORKDIR=/home/grkremer/perf_2026


source /home/grkremer/miniconda3/bin/activate
conda env list
conda activate perf_2026

python train_and_eval.py