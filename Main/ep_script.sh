#!/bin/bash
#SBATCH -J ep
#SBATCH -p pehlevan_gpu
#SBATCH -n 1 # Number of cores/tasks
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 0-06:00:00 # Runtime in D-HH:MM:SS
#SBATCH --mem=6000
#SBATCH --gres=gpu:1 # Number of GPUsi, removed 2 lines
#SBATCH -o Recheck/n3_eplat_%A.o
#SBATCH -e Recheck/n3_eplat_%A.e

module load Anaconda/5.0.1-fasrc02
module load cuda/9.0-fasrc02 cudnn/7.0_cuda9.0-fasrc01
source activate theano_env 

#THEANO_FLAGS="device=cuda, floatX=float32, gcc.cxxflags='-march=core2'" python train_model.py 
THEANO_FLAGS="device=cuda, floatX=float32, gcc.cxxflags='-march=core2'" python train_model_wlat_ep.py 

