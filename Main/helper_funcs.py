import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import numpy as np
import os


# Visualization
def plot_del_si(layers):
    nit_tot, num_layers = len(layers[0]), len(layers)


def analyze(names, lnames, title):
    colorlist = ['g', 'b', 'k', 'r', 'm']
    plt.figure()
    for n, name in enumerate(names):
        sav_log = np.load(name)
        if 'wolat' in name:
            tr_err, val_err = np.array(sav_log[3]['training error']), np.array(sav_log[3]['validation error'])

        else:
            tr_err, val_err = np.array(sav_log[4]['training error']), np.array(sav_log[4]['validation error'])
        print (name, tr_err[-1], val_err[-1])
        plt.plot(np.arange(len(tr_err)), tr_err, linestyle='--', color=colorlist[n])
        plt.plot(np.arange(len(val_err)), val_err, label=lnames[n], linestyle='-', color=colorlist[n])
    plt.ylabel('Error (in %)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title(title)
    plt.show()
    return


def gs_analyze_3p_triple(dirname, hpgrid, mode='plotly'):
    if mode == 'plotly':
        fig = make_subplots()
        for in1, alphaw1 in enumerate(hpgrid[0]):
            for in2, afw2 in enumerate(hpgrid[1]):
                for in3, alf1 in enumerate(hpgrid[2]):
                    alphaw2 = alphaw1 * afw2
                    alphal1 = alphaw1 * alf1
                    svfile = np.load(dirname + '{}_{}_{}.save'.format(in1, in2, in3))
                    tr_err, val_err = np.array(svfile[4]['training error']), np.array(svfile[4]['validation error'])
                    fig.add_trace(
                        dict(x=np.arange(len(val_err)), y=val_err, name='{}, {}, {}'.format(alphaw1, alphaw2, alphal1)),
                        1, 1)
                    print (alphaw1, alphaw2, alphal1, tr_err[-1], val_err[-1])
        fig.show()

    else:
        plt.figure()
        for in1, alphaw1 in enumerate(hpgrid[0]):
            for in2, afw2 in enumerate(hpgrid[1]):
                for in3, alf1 in enumerate(hpgrid[2]):
                    alphaw2 = alphaw1 * afw2
                    alphal1 = alphaw1 * alf1
                    svfile = np.load(dirname + '{}_{}_{}.save'.format(in1, in2, in3))
                    tr_err, val_err = np.array(svfile[4]['training error']), np.array(svfile[4]['validation error'])
                    plt.plot(np.arange(len(val_err)), val_err, label='{}, {}, {}'.format(alphaw1, alphaw2, alphal1))
                    print (alphaw1, alphaw2, alphal1, tr_err[-1], val_err[-1])
                    # plt.plot(np.arange(len(val_err)), val_err, label='{}, {}, {}'.format(alphaw1, alphaw2, alphal1))
        plt.ylabel('Error (in %)')
        plt.xlabel('Epoch')
        plt.legend()
        plt.title('GS, EP Only, Alphas: Training Error')
        plt.show()
    return


def gs_analyze_3p(dirname, hpgrid, mode='plotly'):
    if mode == 'plotly':
        fig = make_subplots()
        figv = make_subplots()
        ctr = 0
        for in1, alphaw1 in enumerate(hpgrid[0]):
            for in2, afw2 in enumerate(hpgrid[1]):
                for in3, alf1 in enumerate(hpgrid[2]):
                    if ctr == 23:
                        print (alphaw1, afw2, alf1)
                        ctr = ctr + 1
                        continue
                    alphaw2 = alphaw1 * afw2
                    # alphal1=alphaw1*alf1
                    alphal1 = alf1
                    svfile = np.load(dirname + '{}.save'.format(ctr))
                    tr_err, val_err = np.array(svfile[4]['training error']), np.array(svfile[4]['validation error'])
                    fig.add_trace(
                        dict(x=np.arange(len(tr_err)), y=tr_err, name='{}, {}, {}'.format(alphaw1, alphaw2, alphal1)),
                        1, 1)
                    figv.add_trace(
                        dict(x=np.arange(len(val_err)), y=val_err, name='{}, {}, {}'.format(alphaw1, alphaw2, alphal1)),
                        1, 1)
                    print (alphaw1, alphaw2, alphal1, tr_err[-1], val_err[-1])
                    ctr = ctr + 1

        fig.update_layout(title_text='Training Error')
        figv.update_layout(title_text='Validation Error')
        fig.show()
        figv.show()

    else:
        plt.figure()
        ctr = 0
        for in1, alphaw1 in enumerate(hpgrid[0]):
            for in2, afw2 in enumerate(hpgrid[1]):
                for in3, alf1 in enumerate(hpgrid[2]):
                    alphaw2 = alphaw1 * afw2
                    alphal1 = alphaw1 * alf1
                    svfile = np.load(dirname + '{}.save'.format(ctr))
                    tr_err, val_err = np.array(svfile[4]['training error']), np.array(svfile[4]['validation error'])
                    plt.plot(np.arange(len(val_err)), val_err, label='{}, {}, {}'.format(alphaw1, alphaw2, alphal1))
                    print (alphaw1, alphaw2, alphal1, tr_err[-1], val_err[-1])
                    # plt.plot(np.arange(len(val_err)), val_err, label='{}, {}, {}'.format(alphaw1, alphaw2, alphal1))
                    ctr = ctr + 1
        plt.ylabel('Error (in %)')
        plt.xlabel('Epoch')
        plt.legend()
        plt.title('GS, EP Only, Alphas: Training Error')
        plt.show()
    return


def gs_analyze_2p(dirname, hpgrid, mode='plotly'):
    if mode == 'plotly':
        fig = make_subplots()
        figv = make_subplots()
        ctr = 0
        for in1, alphaw1 in enumerate(hpgrid[0]):
            for in2, afw2 in enumerate(hpgrid[1]):
                alphaw2 = alphaw1 * afw2
                svfile = np.load(dirname + '{}.save'.format(ctr))
                tr_err, val_err = np.array(svfile[3]['training error']), np.array(svfile[3]['validation error'])
                fig.add_trace(
                    dict(x=np.arange(len(tr_err)), y=tr_err, name='{}, {}'.format(alphaw1, alphaw2)), 1, 1)
                figv.add_trace(
                    dict(x=np.arange(len(val_err)), y=val_err, name='{}, {}'.format(alphaw1, alphaw2)), 1,
                    1)
                print (alphaw1, alphaw2, tr_err[-1], val_err[-1])
                ctr = ctr + 1

        fig.update_layout(title_text='Training Error')
        figv.update_layout(title_text='Validation Error')
        fig.show()
        figv.show()
    else:
        print 'Uncoded'
    return


def gs_analyze(dirname, hpgrid, numparams, mode):
    if numparams == 3:
        # gs_analyze_3p_triple(dirname, hpgrid, mode)
        gs_analyze_3p(dirname, hpgrid, mode)
    if numparams == 2:
        gs_analyze_2p(dirname, hpgrid, mode)
    return


def plot_fid(fnamelist, mode='plotly'):
    if mode == 'plotly':
        fig = make_subplots()
        figv = make_subplots()
        ctr = 0

    for fname in fnamelist:
        if '.save' in fname:
            statefile = np.load(fname)
            tr_err, val_err = statefile[-1]['training error'], statefile[-1]['validation error']
            print (fname, tr_err[-1], val_err[-1])
            index = fname[fname.rindex('/') + 1:]
            index = index.replace('.save', '')
            fig.add_trace(dict(x=np.arange(len(tr_err)), y=tr_err, name=index),
                          1, 1)
            figv.add_trace(
                dict(x=np.arange(len(val_err)), y=val_err, name=index), 1, 1)
    fig.update_layout(title_text='Training Error')
    figv.update_layout(title_text='Validation Error')
    fig.show()
    figv.show()


# Hyperparameter Grids
grid_epwolat = {
    'w1lr': np.array([0.1, 0.25, 0.5, 0.8]),
    'w2f': np.array([0.25, 0.5, 0.75])
}

grid_eplat = {
    'w1lr': np.array([0.4, 0.5, 0.8]),
    'w2f': np.array([0.25, 0.5]),
    'lf': np.array([1.5, 2.0, 2.5])
}

grid_smep = {
    'w1lr': np.array([0.1, 0.5]),
    'w2f': np.array([0.25, 0.5, 0.75]),
    'llr': np.array([0.001, 0.005, 0.01, 0.05, 0.75])
}

if __name__ == '__main__':
    '''
    #EP_wolat
    hpgrid = [grid_epwolat['w1lr'], grid_epwolat['w2f']]
    gs_analyze('Cluster_runs/EP_woLat/GS_alphaconst_25ep/net1_alphas_', hpgrid, 2, 'plotly')'''
    '''
    #SMEP
    hpgrid = [grid_smep['w1lr'], grid_smep['w2f'], grid_smep['llr']]
    gs_analyze('Cluster_runs/SMEP/Net1_gs/GS2_alphaconst_25ep/net1_alphas_', hpgrid, 3, 'plotly')'''

    # Comparative Net1
    '''
    dir_prefix='Comparisons/Net1/'
    savfilenames=['epwolat_betapos.save', 'epwolat_betasigned.save', 'eplat_betapos.save', 'smep_betapos.save']
    savfilenames=[dir_prefix+sav for sav in savfilenames]
    analyze(savfilenames, ['EP NoLat', 'EP NoLat Beta regularized', 'EP Lateral', 'SMEP'], 'Comparison Network1')
    print 4
    '''
    # Plotting all save files in a directory
    # Comparison Net3
    '''
    fnamelist=['Cluster_runs/SMEP/Net3/'+name for name in os.listdir('Cluster_runs/SMEP/Net3')]
    plot_fid(fnamelist, 'plotly')

    fnamelist=['Cluster_runs/EPonly_Lat/Net3/'+name for name in os.listdir('Cluster_runs/EPonly_Lat/Net3')]
    plot_fid(fnamelist, 'plotly')'''


    comdirname = 'Comparisons/FinalNet1/'
    fnamelist = os.listdir(comdirname)
    fnames = []
    for name in fnamelist:
        if '.save' in name:
            fnames.append(comdirname + name)
    plot_fid(fnames, 'plotly')
    '''
    comdirname = 'Normalized_runs/'
    fnamelist = os.listdir(comdirname)
    for name in fnamelist:
        if '.save' in name:
            fnames.append(comdirname + name)
    
    
    comdirname = 'Comparisons/Net3/'
    fnamelist = os.listdir(comdirname)
    fnames = []
    for name in fnamelist:
        if '.save' in name:
            fnames.append(comdirname + name)
    comdirname = 'Normalized_runs/'
    fnamelist = os.listdir(comdirname)
    for name in fnamelist:
        if '.save' in name:
            fnames.append(comdirname + name)
    # fnames.append('Mod_Experiments/Best_runs/smep_4_post70_cdiff_alphasmall.save')
    # fnames.append('Mod_Experiments/Best_runs/smep_4_post70_cdiff_alphasmall.save')
    plot_fid(fnames, 'plotly')
    '''
    # gs_analyze()
    # analyze('Cluster_runs/EPonly_Lat/Net1/net1_lat_betasigned_gradsub.save', r'#L=1, $\beta$ signed, gradient subtracted')

    # dirname='Cluster_runs/EPonly_Lat/GS_alphaconst_25ep/net1_alphas_'
    # gs_analyze(dirname, [w1lr, w2f, lf], 3, 'plotly')'''
