import numpy as np
import sys

from train_model_smep_str import train_net as train_net_smep_str
from model_smep_str import Network as Network_Lateral_SMEP_Str
#from Code_Variants.train_model_smep_str_dense import train_net as train_net_smep_str_sdhybrid
#from Code_Variants.model_smep_str_dense import Network as Network_Lateral_SMEP_Str_SDHybrid
#from Code_Variants.train_model_smep_fast import train_net as train_net_smep_fast
#from Code_Variants.model_smep_fast import Network as Network_Lateral_SMEP_Fast



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
    r_1, r_2, r_3 = 8, 12, 24
    dataset=sys.argv[2]  #"mnist_reduced" 
    nps_input = int(sys.argv[3])

    if (sys.argv[1]=='final_net1'): #Run 1HL Structured for fixed LRs
        #alphaw1, alphaw2, alphal = lr_all[int(sys.argv[2]), 0], lr_all[int(sys.argv[2]), 1], lr_all[int(sys.argv[2]), 2]
        alphaw1, alphaw2, alphal = 0.5, 0.375, 0.01
        alphas_fwd = list(np.asarray([alphaw1, alphaw2], dtype=np.float32))
        alphas_lat = list(np.asarray([alphal], dtype=np.float32))
        hp_dict = create_hyp_param_combination(alphas_fwd=alphas_fwd, alphas_lat = alphas_lat, nps=[nps_input], stride = [1, 2], dataset=dataset, n_epochs = 100, radius= [r_1])
        '''
        #for single layer case
        if (dataset=="mnist") and (nps_input==4): 
            dirname = '/n/scratchlfs02/pehlevan_lab/nmudur_smep/Structured/smep_s2_r{}/'.format(r_1)
        if (dataset=="mnist_reduced") and (nps_input==4): 
            dirname = '/n/scratchlfs02/pehlevan_lab/nmudur_smep/Structured/smep_s2_r{}/'.format(r_1)
        if (dataset=="mnist") and (nps_input==20): 
            dirname = '/n/scratchlfs02/pehlevan_lab/nmudur_smep/Structured/smep_s2_r{}/'.format(r_1)
        '''
        dirname = 'Recheck/'
        name = dirname+'str_net1_r{}_n{}_{}'.format(r_1, nps_input, 'bfit')
        train_net_smep_str(Network_Lateral_SMEP_Str(name, hp_dict))
    
    if (sys.argv[1]=='final_net3'): #Run 3HL Structured for fixed LRs
        lr_all = np.load('lh_grid3.npy')
        hpdict = create_hyp_param_combination{alphas_lat=list(np.asarray([0.96, 0.24, 0.06], dtype=np.float32)), n_it_neg=500, stride=list([1, 2, 4, 8]), epsilon=0.5, variant='normal', batch_size =20, n_epochs= 555550, beta_reg_bool=False, beta=1.0, alphas_fwd=list(np.asarray([0.64, 0.16, 0.04, 0.01], dtype=np.float32)), radius= list([8, 12, 24]), n_it_pos=8, dataset='mnist', alpha_tdep_type='constant', nps=list([4, 4, 4])}

        #alphas_fwd = list(np.asarray(lr_all[int(sys.argv[2]), :4], dtype=np.float32)) REPLACE WITH FIXED LRS FOR BFIT
        #alphas_lat = list(np.asarray(lr_all[int(sys.argv[2]), 4:], dtype=np.float32)) 
        dirname = 'Recheck/'
        name = dirname+'str_net3_r{}_n{}_{}'.format(r_1, nps_input, 'bfit')
        train_net_smep_str(Network_Lateral_SMEP_Str(name, hp_dict))
    '''    
    if len(sys.argv)==2: #Default case, GS1 code. Run with GS2
        hp_dict = create_hyp_param_combination(alphas_fwd=alphas_fwd, alphas_lat = alphas_lat, nps=[4])
        dirname = '/n/scratchlfs02/pehlevan_lab/nmudur_smep/Structured/smep_s2_r4_nps4/Full28x28/'
        name = dirname+'str_nps4_{}'.format(int(sys.argv[1]))
        train_net_smep_str(Network_Lateral_SMEP_Str(name, hp_dict))

    if len(sys.argv)==4: #Reduced, and / or nps not 20, Command Line Args: GS array value, dataset, nps. Run with gs3
        dataset=sys.argv[2]  #"mnist_reduced"
        nps_input = int(sys.argv[3])
        hp_dict = create_hyp_param_combination(alphas_fwd=alphas_fwd, alphas_lat = alphas_lat, nps=[nps_input, nps_input, nps_input], stride = [1, 2, 4, 8], dataset=dataset, n_epochs = 1000, radius= [r_1, r_2, r_3])
        
        #for single layer case
        #if (dataset=="mnist") and (nps_input==4): 
        #    dirname = '/n/scratchlfs02/pehlevan_lab/nmudur_smep/Structured/smep_s2_r{}/'.format(r_1)
        #if (dataset=="mnist_reduced") and (nps_input==4): 
        #    dirname = '/n/scratchlfs02/pehlevan_lab/nmudur_smep/Structured/smep_s2_r{}/'.format(r_1)
        #if (dataset=="mnist") and (nps_input==20): 
        #    dirname = '/n/scratchlfs02/pehlevan_lab/nmudur_smep/Structured/smep_s2_r{}/'.format(r_1)
        #
        dirname = 'Structured_Net3/GS1/'
        name = dirname+'str_nps{}_gs{}_1000ep_fast'.format(nps_input, int(sys.argv[1]))
        train_net_smep_str(Network_Lateral_SMEP_Str(name, hp_dict))
    
    if len(sys.argv)==5: #Reduced, and / or nps not 20, Command Line Args: GS array value, dataset, nps, Plus using SDHybrid code versions. Run with GS4
        dataset=sys.argv[2]  #"mnist_reduced"
        key="full"
        if dataset == "mnist_reduced":
            key="red"
        nps_input = int(sys.argv[3])
        code_version = sys.argv[4]
        hp_dict = create_hyp_param_combination(alphas_fwd=alphas_fwd, alphas_lat = alphas_lat, nps=[nps_input], dataset=dataset, n_epochs = 100, radius = [r_1])
        if code_version=="fast": 
            hp_dict = create_hyp_param_combination(alphas_fwd=alphas_fwd, alphas_lat = alphas_lat, nps=[nps_input], dataset=dataset, n_epochs = 200, radius = [r_1])
            dirname = '/n/scratchlfs02/pehlevan_lab/nmudur_smep/Structured/large_nps/'.format(r_1)
            name = dirname+'s{}_r{}_nps{}_{}_{}_gs{}'.format(2, r_1, nps_input, key, code_version, int(sys.argv[1]))
            train_net_smep_fast(Network_Lateral_SMEP_Fast(name, hp_dict))
        if code_version=="faster": 
            hp_dict = create_hyp_param_combination(alphas_fwd=alphas_fwd, alphas_lat = alphas_lat, nps=[nps_input], dataset=dataset, n_epochs = 200, radius = [r_1])
            dirname = '/n/scratchlfs02/pehlevan_lab/nmudur_smep/Structured/large_nps_trunc/'.format(r_1)
            name = dirname+'s{}_r{}_nps{}_{}_{}_gs{}'.format(2, r_1, nps_input, key, code_version, int(sys.argv[1]))
            train_net_smep_fast(Network_Lateral_SMEP_Fast(name, hp_dict))
        if code_version=="sdhybrid":
            if (dataset=="mnist") and (nps_input==4): 
                dirname = '/n/scratchlfs02/pehlevan_lab/nmudur_smep/Structured/smep_s2_r{}/'.format(r_1)
            if (dataset=="mnist_reduced") and (nps_input==4): 
                dirname = '/n/scratchlfs02/pehlevan_lab/nmudur_smep/Structured/smep_s2_r{}/'.format(r_1)
            if (dataset=="mnist") and (nps_input==20): 
                dirname = '/n/scratchlfs02/pehlevan_lab/nmudur_smep/Structured/smep_s2_r{}/'.format(r_1)
            if (dataset=="mnist_reduced") and (nps_input==20): 
                dirname = '/n/scratchlfs02/pehlevan_lab/nmudur_smep/Structured/smep_s2_r{}/'.format(r_1)

            name = dirname+'s{}_r{}_nps{}_{}_{}_gs{}'.format(2, r_1, nps_input, key, code_version, int(sys.argv[1]))
            train_net_smep_str_sdhybrid(Network_Lateral_SMEP_Str_SDHybrid(name, hp_dict))
        else: #sparse
            if (dataset=="mnist") and (nps_input==4): 
                dirname = '/n/scratchlfs02/pehlevan_lab/nmudur_smep/Structured/smep_s2_r4_nps4/Full28x28/'
            if (dataset=="mnist_reduced") and (nps_input==4): 
                dirname = '/n/scratchlfs02/pehlevan_lab/nmudur_smep/Structured/smep_s2_r4_nps4/Reduced20x20/'
            if (dataset=="mnist") and (nps_input==20): 
                dirname = '/n/scratchlfs02/pehlevan_lab/nmudur_smep/Structured/smep_s2_r4_nps20/Reduced20x20/'
            name = dirname+'str_mode_{}_nps{}_gs{}'.format(code_version[:3], nps_input, int(sys.argv[1]))
            train_net_smep_str(Network_Lateral_SMEP_Str(name, hp_dict))
    '''    

