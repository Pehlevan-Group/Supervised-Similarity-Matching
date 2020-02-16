To run Structured connectivity runs for fixed parameters: 

  * Net1: `THEANO_FLAGS="device=cuda, floatX=float32, gcc.cxxflags='-march=core2'" python run_str_fixedlrs.py "final_net1" "mnist" 4 `
  
  * Net3: `THEANO_FLAGS="device=cuda, floatX=float32, gcc.cxxflags='-march=core2'" python run_str_fixedlrs.py "final_net3" "mnist" 4`

Using Slurm Script:
Edit `final_str.sh` to add module, directory, and partition, and the appropriate run (finalnet1 or finalnet3). Run `'sbatch final_str.sh'`.


