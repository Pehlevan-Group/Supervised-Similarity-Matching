#!/bin/bash
#SBATCH -J final_str
#SBATCH -p <insert_partition_name>
#SBATCH -n 1 # Number of cores/tasks
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 2-00:00:00 # Runtime in D-HH:MM:SS
#SBATCH --mem=6000
#SBATCH --gres=gpu:1 # Number of GPUsi, removed 2 lines
#SBATCH -o Recheck/str_net3_%J.o
#SBATCH -e Recheck/str_net3_%J.e

module load <insert Anaconda module name>
module load <insert cuda module name>
source activate theano_env

THEANO_FLAGS="device=cuda, floatX=float32, gcc.cxxflags='-march=core2'" python run_str_fixedlrs.py "final_net3" "mnist" 4
