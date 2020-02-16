Net1:
  * EP, Betasigned / Betapositive: Pass appropriate model to train_net in the main body of train_model.py. Run THEANO_FLAGS="device=cuda, floatX=float32, gcc.cxxflags='-march=core2'" python train_model.py in terminal.
  * EP+Lateral: Pass appropriate model to train_net in the main body of train_model_wlat_ep.py. Run THEANO_FLAGS="device=cuda, floatX=float32, gcc.cxxflags='-march=core2'" python train_model_wlat_ep.py in terminal.
  * SMEP: THEANO_FLAGS="device=cuda, floatX=float32, gcc.cxxflags='-march=core2'" python mod_exp_smep_tmp2.py 'constant_net1'

Net3:
  * EP, Betasigned / Betapositive: Pass appropriate model to train_net in the main body of train_model.py. Run THEANO_FLAGS="device=cuda, floatX=float32, gcc.cxxflags='-march=core2'" python train_model.py in terminal.
   * EP+Lateral: Pass appropriate model to train_net in the main body of train_model_wlat_ep.py. Run THEANO_FLAGS="device=cuda, floatX=float32, gcc.cxxflags='-march=core2'" python train_model_wlat_ep.py in terminal.
   * Adaptrerr: THEANO_FLAGS="device=cuda, floatX=float32, gcc.cxxflags='-march=core2'" python mod_exp_smep_tmp2.py 4 'new' 'smep' 'mnist'
  * SMEP: THEANO_FLAGS="device=cuda, floatX=float32, gcc.cxxflags='-march=core2'" python mod_exp_smep_tmp2.py 'constant_net3'
