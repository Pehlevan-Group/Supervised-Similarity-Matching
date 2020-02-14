import numpy as np
import sys

from train_model_smep_str import train_net as train_net_smep_str
from model_smep_str import Network as Network_Lateral_SMEP_Str
from Code_Variants.train_model_smep_str_dense import train_net as train_net_smep_str_dense
from Code_Variants.model_smep_str_dense import Network as Network_Lateral_SMEP_Str_Dense
def create_hyp_param_combination(  #
    nps= [20],
    stride= [1, 2],
    radius= [4],
    dataset= "mnist",
    n_epochs= 50,
    batch_size= 20,
    n_it_neg= 20,
    n_it_pos= 4,
    epsilon= np.float32(.5),
    beta  = np.float32(1.),
    alphas_fwd= [np.float32(0.5), np.float32(.25)],
    alphas_lat= [np.float32(0.75)],
    variant= "normal",  # Non-Linearity type: Clipped between -1, 1, or 0, 1
    alpha_tdep_type= "constant",
    beta_reg_bool= False):

    hp_dict =  {  #
        "nps": nps,
        "stride": stride,
        "radius": radius,
        "dataset": dataset,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "n_it_neg": n_it_neg,
        "n_it_pos": n_it_pos,
        "epsilon": epsilon,
        "beta"  : beta,
        "alphas_fwd": alphas_fwd,
        "alphas_lat": alphas_lat,
        "variant": variant,  # Non-Linearity type: Clipped between -1, 1, or 0, 1
        "alpha_tdep_type": alpha_tdep_type,
        "beta_reg_bool": beta_reg_bool
    }
    return hp_dict
        
        

if __name__=='__main__':
    lr_all = np.load('lh_grid1.npy')
    alphaw1, alphaw2, alphal = lr_all[int(sys.argv[1]), 0], lr_all[int(sys.argv[1]), 1], lr_all[int(sys.argv[1]), 2]
    alphas_fwd = list(np.array([alphaw1, alphaw2], dtype=np.float32))
    alphas_lat = list(np.array([alphal], dtype=np.float32))
    
    hp_dict = create_hyp_param_combination(alphas_fwd=alphas_fwd, alphas_lat = alphas_lat)
    dirname = '/n/scratchlfs02/pehlevan_lab/nmudur_smep/Structured/smep_s2_r4_nps20/'
    name = dirname+'str_{}'.format(int(sys.argv[1]))
    train_net_smep_str(Network_Lateral_SMEP_Str(name, hp_dict))
    #train_net_smep_str_dense(Network_Lateral_SMEP_Str_Dense(name, hp_dict))
    
    '''
    #PART1: Code for 1 HL, single run, specifying hyperparams
    dirname='Net1_Repr/' #CHANGE DIRECTORY CHOICE HERE, .save file will be stored in this directory
    name = dirname + 'smep'
    hpd1 = create_hyp_param_combination(hidden_sizes = [500],
        n_epochs=100,
        batch_size=20,
        n_it_neg=20,
        n_it_pos=4,
        epsilon=np.float32(.5),
        beta=np.float32(1.),
        beta_reg_bool=False,
        alpha_tdep_type='constant',
        dataset="mnist",
        variant="normal",
        alphas_fwd= [np.float32(0.5), np.float32(0.375)], 
        alphas_lat=[np.float32(0.001)], #CHANGE ALPHA LATERAL HERE 
    )
    train_net_smep_mod(Network_SMEP_Mod(name, hpd1))
    '''
    
    #PART2: Code for 3HL, different variants, predecided learning rate combinations
    #run_experiment('alpha_segmented_repr')
    #run_experiment('alpha_segmented_old_40')
    #run_experiment('alpha_segmented')
    #run_experiment('cont_alphasmall')
    #run_experiment('alphatd2')
    #run_experiment('alphatd1')
    #run_experiment('alphadiff')


