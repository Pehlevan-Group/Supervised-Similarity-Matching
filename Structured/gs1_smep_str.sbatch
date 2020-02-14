#!/bin/bash
#SBATCH -J gs1_str_smep
#SBATCH -p pehlevan_gpu
#SBATCH -n 2 # Number of cores/tasks
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 3-00:00:00 # Runtime in D-HH:MM:SS
#SBATCH --mem=6000
#SBATCH --gres=gpu:1 # Number of GPUsi, removed 2 lines
#SBATCH -o GS_Run50ep/gs_%A_%a.o
#SBATCH -e GS_Run50ep/gs_%A_%a.e

module load Anaconda/5.0.1-fasrc02
module load cuda/9.0-fasrc02 cudnn/7.0_cuda9.0-fasrc01
source activate theano_env 

THEANO_FLAGS="device=cuda, floatX=float32, gcc.cxxflags='-march=core2'" python grid_search_str.py ${SLURM_ARRAY_TASK_ID}
