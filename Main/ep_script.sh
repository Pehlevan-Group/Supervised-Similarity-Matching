#!/bin/bash
#SBATCH -J ep
#SBATCH -p <insert_partition>
#SBATCH -n 1 # Number of cores/tasks
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 0-01:00:00 # Runtime in D-HH:MM:SS
#SBATCH --mem=6000
#SBATCH --gres=gpu:1 # Number of GPUsi, removed 2 lines
#SBATCH -o Recheck/n1_betapos_run2_%A.o
#SBATCH -e Recheck/n1_betapos_run2_%A.e

module load <insert Anaconda module name>
module load <insert cuda module name>
source activate theano_env

THEANO_FLAGS="device=cuda, floatX=float32, gcc.cxxflags='-march=core2'" python train_model.py 
#THEANO_FLAGS="device=cuda, floatX=float32, gcc.cxxflags='-march=core2'" python train_model_wlat_ep.py 

