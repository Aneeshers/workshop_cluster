#!/bin/bash
#SBATCH -c 1               # Number of cores (-c)
#SBATCH -t 0-01:10          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu   # Partition to submit to
#SBATCH --account=hankyang_lab
#SBATCH --mem=3000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# load modules
module load python/3.10.9-fasrc01
conda activate torch
cd /n/home04/amuppidi/workshop

# run code with passed arguments
~/.conda/envs/torch/bin/python mnist.py "$@"
