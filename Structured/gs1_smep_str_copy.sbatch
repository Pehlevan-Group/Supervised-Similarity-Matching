#!/bin/bash
#SBATCH -J lfs_gs1_smep_str
#SBATCH -p pehlevan_gpu
#SBATCH -n 2 # Number of cores/tasks
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 3-00:00:00 # Runtime in D-HH:MM:SS
#SBATCH --mem=6000
#SBATCH --gres=gpu:1 # Number of GPUsi, removed 2 lines
#SBATCH -o /n/scratchlfs02/pehlevan_lab/nmudur_smep/Structured/smep_s2_r4_nps20/gs_%A_%a.o
#SBATCH -e /n/scratchlfs02/pehlevan_lab/nmudur_smep/Structured/smep_s2_r4_nps20/gs_%A_%a.e

module load Anaconda/5.0.1-fasrc02
module load cuda/9.0-fasrc02 cudnn/7.0_cuda9.0-fasrc01
source activate theano_env 

THEANO_FLAGS="device=cuda, floatX=float32, gcc.cxxflags='-march=core2'" python grid_search_str_copy.py ${SLURM_ARRAY_TASK_ID}
