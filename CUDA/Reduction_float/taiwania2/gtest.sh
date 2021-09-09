#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --nodes=1
#SBATCH -p gtest
#SBATCH -t 00:5:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --account=MST108480
#SBATCH --output=out4d.txt
#SBATCH --error=err.txt
stdbuf -o0 -e0 ./a.out

module load cuda/11.3
./reduction.sh