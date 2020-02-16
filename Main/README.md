Net1:
  * EP, Betasigned / Betapositive: Pass appropriate model to train_net in the main body of train_model.py. Uncomment line running train_model.py in ep_script.sh. Run 'sbatch ep_script.sh' in terminal.
  * EP+Lateral: Pass appropriate model to train_net in the main body of train_model_wlat_ep.py. Uncomment line running train_model_wlat_ep.py in ep_script.sh. Run 'sbatch ep_script.sh' in terminal.
  * SMEP: Uncomment line in smep.sbatch with 'constant_net1'. Run 'sbatch --array=0 smep.sbatch'.

Net3:
  * EP, Betasigned / Betapositive: Pass appropriate model to train_net in the main body of train_model.py. Uncomment line running train_model.py in ep_script.sh. Run 'sbatch ep_script.sh' in terminal.
  * EP+Lateral: Pass appropriate model to train_net in the main body of train_model_wlat_ep.py. Uncomment line running train_model_wlat_ep.py in ep_script.sh. Run 'sbatch ep_script.sh' in terminal.
  * Adaptrerr: Uncomment the line in smep.sbatch with 4 args. Run 'sbatch --array=4 smep.sbatch' in terminal.
  * SMEP: Uncomment line in smep.sbatch with 'constant_net3'. Run 'sbatch --array=0 smep.sbatch' in terminal.
