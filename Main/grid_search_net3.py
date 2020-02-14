import numpy as np
import sys
from train_model_wlat_smep import train_net as train_net_lat_smep
from model_wlat_smep import Network as Network_Lateral_SMEP
from train_model_wlat_ep import train_net as train_net_lat
from model_wlat_ep import Network as Network_Lateral
from train_model import train_net as train_net
from model import Network as Network
from train_model_smep_mod import train_net as train_net_smep_mod
from model_wlat_smep_mod import Network as Network_SMEP_Mod
'''
Restrictions: THIS CODE IS ONLY SUITABLE FOR RUNNING NET3 (ALPHAS CONSTANT) FOR EPNOLAT, EPLAT, OR SMEP, WITH ONLY MNIST. Not to be run with SMEP Mod, since that has fewer constraints, and guessed hyperparameter combinations as opposed to a grid search.
One command line argument: set of learning rates to be chosen
'''

def create_hyp_param_combination_for_eplat(hidden_sizes=[500, 500, 500],
    n_epochs=250,
    batch_size=20,
    n_it_neg=500,
    n_it_pos=8,
    epsilon=np.float32(.5),
    beta=np.float32(1.),
    alphas_fwd=[np.float32(0.5), np.float32(.125), np.float32(0.0625), np.float32(0.0312)],
    alphas_lat=[np.float32(0.75), np.float32(0.1875), np.float32(0.046)],
    beta_reg_bool=False,
    alpha_tdep_bool=False,
    dataset="mnist"
):
    #Layers
    hp_dict  = {
    "hidden_sizes": hidden_sizes,
    "n_epochs": n_epochs,
    "batch_size": batch_size,
    "n_it_neg": n_it_neg,
    "n_it_pos": n_it_pos,
    "epsilon": epsilon,
    "beta": beta,
    "alphas_fwd": alphas_fwd,
    "alphas_lat": alphas_lat,
    "beta_reg_bool": beta_reg_bool,
    "alpha_tdep_bool": alpha_tdep_bool}
    return hp_dict

def create_hyp_param_combination(hidden_sizes=[500, 500, 500],
    n_epochs=250,
    batch_size=20,
    n_it_neg=500,
    n_it_pos=8,
    epsilon=np.float32(.5),
    beta=np.float32(1.),
    alphas_fwd=[np.float32(0.5), np.float32(.125), np.float32(0.0625), np.float32(0.0312)],
    alphas_lat=[np.float32(0.75), np.float32(0.1875), np.float32(0.046)],
    beta_reg_bool=False,
    alpha_tdep_bool=False,
    dataset="mnist"
):
    #Layers
    if alphas_lat is not None:
        if (alpha_tdep_bool==False):
            alpha_tdep_type='normal'
        hp_dict  = {
        "hidden_sizes": hidden_sizes,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "n_it_neg": n_it_neg,
        "n_it_pos": n_it_pos,
        "epsilon": epsilon,
        "beta": beta,
        "alphas_fwd": alphas_fwd,
        "alphas_lat": alphas_lat,
        "beta_reg_bool": beta_reg_bool,
        "alpha_tdep_type": alpha_tdep_type,
        "variant": "constant",
        "dataset": dataset}
    else:#EP_NOLateral  
        hp_dict  = {
        "hidden_sizes": hidden_sizes,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "n_it_neg": n_it_neg,
        "n_it_pos": n_it_pos,
        "epsilon": epsilon,
        "beta": beta,
        "alphas": alphas_fwd,
        "beta_reg_bool": beta_reg_bool,
        "alpha_tdep_bool": alpha_tdep_bool,
        "dataset": dataset}
    return hp_dict

def grid_search_over_params(dirname, alphasgrid, ind, mode='smep', L=3, dataset='mnist', beta_reg_bool=False):
    if (mode=='eplat') or (mode=='smep'): #alphaw1, tauw, taul
        alphas_hp=alphasgrid[ind]
        alphaw1, fac2, facl=alphas_hp[0], alphas_hp[1], alphas_hp[2]
        name = dirname+mode+'_alphas_{}'.format(ind)
        alphas_w=np.zeros(L+1)
        alphas_w[0]=alphaw1
        for l in range(1, L+1):
            alphas_w[l]=alphas_w[l-1]*fac2
        alphas_w=list(np.array(alphas_w, dtype=np.float32))
        
        alphas_l=np.zeros(L)
        for l in range(L):
            alphas_l[l]=alphas_w[l]*facl
        alphas_l=list(np.array(alphas_l, dtype=np.float32))
        if L==3:
            hp_dict = create_hyp_param_combination(alphas_fwd=alphas_w, alphas_lat=alphas_l, dataset=dataset, beta_reg_bool=beta_reg_bool)
        if mode=='smep':
            train_net_lat_smep(Network_Lateral_SMEP(name, hp_dict))
        else:
            train_net_lat_ep(Network_Lateral(name, hp_dict))
    elif (mode=='nolat'):
        alphas=alphasgrid[ind]
        alphaw1, fac2=alphas[0], alphas[1]
        alphas_w=np.zeros(L+1)
        name = dirname
        alphas_w[0]=alphaw1
        for l in range(1, L+1):
            alphas_w[l]=alphas_w[l-1]*fac2
        alphas_w=list(np.array(alphas_w, dtype=np.float32))
        hp_dict = create_hyp_param_combination(alphas_fwd=alphas_w, alphas_lat=None, beta_reg_bool=beta_reg_bool) 
        train_net(Network(name, hp_dict))
    elif (mode=='smep_fast'): 
        alphas_fwd = list(lr_all[ind, :4])
        alphas_lat = list(lr_all[ind, 4:])
        name = dirname+'gs_fast_{}'.format(ind)
        hp_dict = create_hyp_param_combination(alphas_fwd=alphas_fwd, alphas_lat=alphas_lat, beta_reg_bool=beta_reg_bool) 
        train_net_smep_mod(Network_SMEP_Mod(name, hp_dict))
    elif (mode=='eplat_fast'): 
        alphas_fwd = list(lr_all[ind, :4])
        alphas_lat = list(lr_all[ind, 4:])
        name = dirname+'gs_fast_{}'.format(ind)
        hp_dict = create_hyp_param_combination_for_eplat(alphas_fwd=alphas_fwd, alphas_lat=alphas_lat, beta_reg_bool=beta_reg_bool) 
        train_net_lat(Network_Lateral(name, hp_dict)) #CHANGE THIS
    elif (mode=='smep_norm'):
        alphas_fwd = list(lr_all[ind, :2]) 
        alphas_lat = list(lr_all[ind, 2:])
        name = dirname+'norm_{}'.format(ind)
        hp_dict = create_hyp_param_combination(hidden_sizes=[500], n_epochs=50, batch_size=20, n_it_neg=20, n_it_pos=4, epsilon=np.float32(.5), beta=np.float32(1.), alpha_tdep_bool=False, dataset='mnist_norm', alphas_fwd=alphas_fwd, alphas_lat=alphas_lat, beta_reg_bool=beta_reg_bool) 
        train_net_smep_mod(Network_SMEP_Mod(name, hp_dict))
    else:
        print 'Error in mode'
    return

if __name__=='__main__':
    #Net1 Case
    #Parameters
    #dirname='Cluster_runs/EP_woLat/Net3/GS_alphaconst_25ep/'
    #dirname ='Cluster_runs/SMEP/Net3/GS_runs/'
    #dirname='Cluster_runs/EPonly_Lat/Net3/GS_alphaconst_25ep/'
    
    #dataset='mnist'
    #dirname='Normalized_runs/'
    #lr_all = np.asarray(np.load('Structured/lh_grid1.npy'), dtype=np.float32)
    
    '''
    alpha_w1_range = np.array([0.5, 0.55, 0.6], dtype=np.float32)
    alpha_w2_factor = np.array([0.6, 0.75, 0.8], dtype=np.float32)
    alpha_l_factor= np.array([0.8, 1.0, 1.2], dtype=np.float32)
    
    hpgrid=[]
    for aw1 in alpha_w1_range:
        for aw2 in alpha_w2_factor:
            for al in alpha_l_factor:
                hpgrid.append([aw1, aw2, al])
    '''
    #hpgrid_repr=[[np.float32(0.128), np.float32(0.25)]] #For NoLat
    #grid_search_over_params(dirname, lr_all, int(sys.argv[1]), mode='smep_norm', L=1, dataset=dataset, beta_reg_bool=False) #For nolat, slurm array value not factored in
    
    
    #Net3 Case 
    
    dataset = 'mnist'
    dirname= 'Fast_Net3_reruns/'
    lr_all = np.asarray(np.load('Fast_Net3_reruns/lh_grid1.npy'), dtype=np.float32)

    grid_search_over_params(dirname, lr_all, int(sys.argv[1]), mode='smep_fast', L=3, dataset=dataset, beta_reg_bool=False)
    #alphaw1, alphaw2, alphal = lr_all[int(sys.argv[1]), 0], lr_all[int(sys.argv[1]), 1], lr_all[int(sys.argv[1]), 2]
    
    #alphas_fwd = list(np.array([alphaw1, alphaw2], dtype=np.float32))
    #alphas_lat = list(np.array([alphal], dtype=np.float32))
    
