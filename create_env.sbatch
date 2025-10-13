#!/bin/bash
#SBATCH -n 32
#SBATCH -p tupi
WORKDIR=/home/grkremer/

##wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
##bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/grkremer/miniconda3

export PATH="/home/grkremer/miniconda3/bin:$PATH"

conda env create -f environment.yml -n pymoo_test
