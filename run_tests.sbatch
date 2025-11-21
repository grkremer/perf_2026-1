#!/bin/bash
#SBATCH -n 2
#SBATCH -p tupi
#SBATCH --time=24:00:00
OUTPUT=/home/users/grkremer/perf_2026/out/slurm-%j.out
#SBATCH -o $OUTPUT
OUT_FILE=$1
WORKDIR=/home/users/grkremer/perf_2026


source ~/miniconda3/bin/activate
conda env list
conda env create -f $WORKDIR/environment.yml -n perf_2026
conda activate perf_2026

python train_and_eval.py $OUT_FILE