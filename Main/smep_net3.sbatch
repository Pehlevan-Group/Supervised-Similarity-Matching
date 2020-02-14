#!/bin/bash
#SBATCH -J recheck_adaptrerr
#SBATCH -p pehlevan_gpu
#SBATCH -n 1 # Number of cores/tasks
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 1-00:00:00 # Runtime in D-HH:MM:SS
#SBATCH --mem=6000
#SBATCH --gres=gpu:1 # Number of GPUsi, removed 2 lines
#SBATCH -o Recheck/smep_net3_%A_%a.o
#SBATCH -e Recheck/smep_net3_%A_%a.e

module load Anaconda/5.0.1-fasrc02
module load cuda/9.0-fasrc02 cudnn/7.0_cuda9.0-fasrc01
source activate theano_env 

#THEANO_FLAGS="device=cuda, floatX=float32, gcc.cxxflags='-march=core2'" python mod_exp_smep_tmp2.py ${SLURM_ARRAY_TASK_ID} 'new' 'smep' 'mnist'
#THEANO_FLAGS="device=cuda, floatX=float32, gcc.cxxflags='-march=core2'" python mod_exp_smep_tmp2.py 'constant_net1'
THEANO_FLAGS="device=cuda, floatX=float32, gcc.cxxflags='-march=core2'" python mod_exp_smep_tmp2.py 'constant_net3'

