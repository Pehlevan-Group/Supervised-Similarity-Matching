#!/bin/bash
#SBATCH -J recheck_adaptrerr
#SBATCH -p <insert_partition>
#SBATCH -n 1 # Number of cores/tasks
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 2-00:00:00 # Runtime in D-HH:MM:SS
#SBATCH --mem=6000
#SBATCH --gres=gpu:1 # Number of GPUsi, removed 2 lines
#SBATCH -o Recheck/adap_trerr_%A_%a.o
#SBATCH -e Recheck/adap_trerr_%A_%a.e

module load <insert Anaconda module name>
module load <insert cuda module name>
source activate theano_env 

THEANO_FLAGS="device=cuda, floatX=float32, gcc.cxxflags='-march=core2'" python mod_exp_smep_tmp2.py ${SLURM_ARRAY_TASK_ID} 'new' 'smep' 'mnist'
#THEANO_FLAGS="device=cuda, floatX=float32, gcc.cxxflags='-march=core2'" python mod_exp_smep_tmp2.py 'constant_net1'
#THEANO_FLAGS="device=cuda, floatX=float32, gcc.cxxflags='-march=core2'" python mod_exp_smep_tmp2.py 'constant_net3'

