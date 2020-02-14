import cPickle
import numpy as np
import os
import theano
import theano.tensor as T
from itertools import chain, izip_longest
from theano.ifelse import ifelse
from scipy.sparse import csc_matrix
import theano.sparse as sp
import theano.sparse.basic as sp_b

from external_world_str import External_World, External_World_Reduced
from structure_funcs import *

def rho(s):
    return T.clip(s, 0., 1.)
    # return T.nnet.sigmoid(4.*s-2.)
def rho_diff(s):
    return T.clip(s, -1., 1.)


class Network(object):

    def __init__(self, name, hyperparameters=dict()):

        self.path = name + ".save"

        # LOAD/INITIALIZE PARAMETERS
        # Layers
        self.struct_matrices, self.biases, self.weights, self.hyperparameters, self.training_curves = self.__load_params(hyperparameters)
        #struct_matrices: 1-> L

        #Precalculate number of non zero elements
        nzelems_fwd=[np.count_nonzero(str_fwd) for ind, str_fwd in enumerate(self.struct_matrices['fwd'])]
        nzelems_fwd.append(self.weights['fwd'][-1].eval().data.size)
        nzelems_lat = [np.count_nonzero(str_lat) for ind, str_lat in enumerate(self.struct_matrices['lat'])]
        self.nzelems = dict({'fwd': nzelems_fwd, 'lat': nzelems_lat}) #fwd: 1->L+1, lat: 1-> L
        
        # Inp diM
        # LOAD EXTERNAL WORLD (=DATA)
        if self.hyperparameters["dataset"] == "mnist":
            input_dim = 28 #DEBUG
            self.external_world = External_World()
        if self.hyperparameters["dataset"] == "cifar10":
            input_dim = 32
            self.external_world = External_World() #CIF
        if self.hyperparameters["dataset"] == "mnist_reduced":
            input_dim = 20
            self.external_world = External_World_Reduced()



        # INITIALIZE PERSISTENT PARTICLES
        dataset_size = self.external_world.size_dataset #50k
        layer_dims = get_layerdims_from_stride(self.hyperparameters["stride"], input_dim) #dl
        #adapt to nps being list
        values = [np.zeros((dataset_size, int(self.hyperparameters["nps"][l_ind]* (layer_dim**2))), dtype=theano.config.floatX) for l_ind, layer_dim in enumerate(layer_dims[1:(-1)])] #Flat choice, values of the nodes, HL=1...L, bs*Vl
        values.append(np.zeros((dataset_size, 1*1*10), dtype=theano.config.floatX))
        self.persistent_particles = [theano.shared(value, borrow=True) for value in values] #1HL-OP

        # LAYERS = MINI-BACTHES OF DATA + MINI-BACTHES OF PERSISTENT PARTICLES
        batch_size = self.hyperparameters["batch_size"]
        self.index = theano.shared(np.int32(0), name='index')  # index of a mini-batch

        self.x_data = self.external_world.x[self.index * batch_size: (self.index + 1) * batch_size]
        self.y_data = self.external_world.y[self.index * batch_size: (self.index + 1) * batch_size]
        self.y_data_one_hot = T.extra_ops.to_one_hot(self.y_data, 10) #Check if reshape needed here
        self.layers = [self.x_data] + [particle[self.index * batch_size: (self.index + 1) * batch_size] for particle in
                                       self.persistent_particles] #

        # BUILD THEANO FUNCTIONS
        self.rho = self.__build_nonlinfunc()
        self.change_mini_batch_index = self.__build_change_mini_batch_index()
        self.measure = self.__build_measure()
        self.measure_verbose = self.__build_measure_verbose()
        self.free_phase = self.__build_free_phase()
        self.weakly_clamped_phase = self.__build_weakly_clamped_phase()



    def save_params(self): #ADAPT FOR STR
        f = file(self.path, 'wb')
        struct_fwd_matrices, struct_lat_matrices = [s.get_value() for s in self.struct_matrices['fwd']], [s.get_value() for s in self.struct_matrices['lat']]
        biases_fwd_values = [b.get_value() for b in self.biases['fwd']]
        weights_fwd_values, weights_lat_values = [W.get_value() for W in self.weights['fwd']], [W.get_value() for W in self.weights['lat']]
        to_dump = struct_fwd_matrices, struct_lat_matrices, biases_fwd_values, weights_fwd_values, weights_lat_values, self.hyperparameters, self.training_curves
        cPickle.dump(to_dump, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        
    def __build_nonlinfunc(self):
        var=self.hyperparameters["variant"]
        if var=='clipdiff':
            print "Clipdiff initialization"
            return rho_diff
        else:
            return rho
        
            

    def __load_params(self, hyperparameters): #R, ADAPT FOR STR, USE STRUC_FUNCS HERE
        hyper = hyperparameters

        # Glorot/Bengio weight initialization

        def initialize_layer(nzelems, d_next, nps_prev, nrow, ncol, pd_bool=False): #pd_bool: True if pos def, DEBUG
            if pd_bool==False:
                rng = np.random.RandomState()
                '''
                W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                        high=np.sqrt(6. / (n_in + n_out)),
                        size=(nrow, ncol)
                    ),
                    dtype=theano.config.floatX
                )'''
                W_values=np.asarray(np.random.normal(0, 1, size=(nrow*ncol)), dtype=theano.config.floatX).reshape(nrow, ncol)
                factor= np.sqrt(nzelems / ((d_next**2.0) * nps_prev))
                W_values= W_values / (factor*10.0)

            else: #n_in = n_out since lateral
                rng=np.random.RandomState()
                '''
                W_values = np.asarray(
                    rng.uniform(
                        low=-np.sqrt(6. / (n_in + n_out)),
                        high=np.sqrt(6. / (n_in + n_out)),
                        size=(nrow, ncol)
                    ),
                    dtype=theano.config.floatX
                )
                '''
                #W_values = np.asarray(np.random.normal(0, 1, size=(nrow * ncol)), dtype=theano.config.floatX).reshape(nrow,
                #                                                                                              ncol)
                #W_values = np.matmul(W_values, W_values.transpose())
                #W_values=np.identity(nrow, dtype=theano.config.floatX)
                W_values = np.asarray(np.random.normal(0, 1, size=(nrow * ncol)), dtype=theano.config.floatX).reshape(
                    nrow,
                    ncol)
                W_values = np.matmul(W_values, W_values.transpose())
                norm_val = np.linalg.norm(W_values)
                W_values = W_values / norm_val

            return W_values


        if os.path.isfile(self.path): #loads weights from the .save file for all subsequent epochs
            f = file(self.path, 'rb')
            struct_fwd_matrices, struct_lat_matrices, biases_fwd_values, weights_fwd_str, weights_lat_str, hyperparameters, training_curves = cPickle.load(f) #assuming saved as sparse matrices
            f.close()
            for k, v in hyper.iteritems(): #replaces the loaded (old) hyperparams from the .save file with the new hyperparameters arg
                hyperparameters[k] = v
            print ('Reading in weights from stored network file')
        else: #CHECK
            stride, radius, nps=hyperparameters["stride"], hyperparameters["radius"], hyperparameters["nps"]
            L=len(radius)
            input_dim = 28 #DEBUG
            if hyperparameters["dataset"] == "cifar10":
                input_dim = 32
            if hyperparameters["dataset"] == "mnist_reduced":
                input_dim = 20
            dl_list=get_layerdims_from_stride(stride, input_dim) #0->L+1
            #Total num nodes
            Vl_list=np.zeros(L+2, dtype=np.int32) #0->L+1
            Vl_list[-1]=10
            Vl_list[:-1] = (np.array(dl_list[:-1]) ** 2) * np.array(
                [1] + nps)  # total number of nodes in each layer, eg: [4096, 3072...]
            s_fwd_list, s_lat_list=get_structured_matices(stride, nps, radius, input_dim) #not sparse
            # Adapted Glorot Bengio init?
            '''
            n_in = np.square(
                2 * np.array(radius, dtype=theano.config.floatX) / np.array(stride[:(-1)], dtype=theano.config.floatX))
            n_out = np.square(
                2 * np.array(radius, dtype=theano.config.floatX) / np.array(stride[1:], dtype=theano.config.floatX))
            n_in = list(np.array(n_in, dtype=np.int32)) + [Vl_list[-2]]
            n_out = list(np.array(n_out, dtype=np.int32)) + [Vl_list[-1]]
            
            weights_fwd_fc = [initialize_layer(size_pre, size_post, nrow, ncol) for size_pre, size_post, nrow, ncol in
                              zip(n_in, n_out, Vl_list[:(-1)], Vl_list[1:])]  # 1->L+1
            weights_fwd_str = [np.multiply(w, s) for w, s in zip(weights_fwd_fc[:-1], s_fwd_list)]
            weights_fwd_str.append(weights_fwd_fc[-1])

            weights_lat_fc = [initialize_layer(size, size, nrow, nrow, pd_bool=True) for size, nrow in
                              zip(nps, Vl_list[1:(-1)])]  # recurrent in all but input and output
            weights_lat_str = [np.multiply(l, str) for l, str in zip(weights_lat_fc, s_lat_list)]'''

            # DSM Initialization, #debug
            #fwd
            nzelems = [np.count_nonzero(s_l) for s_l in s_fwd_list]  # 1->L
            d_next = dl_list[1:(-1)]
            nps_prev = [1] + nps[:(-1)]
            weights_fwd_fc = [initialize_layer(nznum, dn, npsp, nrow, ncol) for nznum, dn, npsp, nrow, ncol in
                              zip(nzelems, d_next, nps_prev, Vl_list[:(-2)], Vl_list[1:(-1)])]  # 1->L
            weights_fwd_str = [np.multiply(w, s) for w, s in zip(weights_fwd_fc, s_fwd_list)] #1->L

            fcmat = np.asarray(np.random.normal(0, 1, size=(Vl_list[-2] * Vl_list[-1])),
                               dtype=theano.config.floatX).reshape(Vl_list[-2], Vl_list[-1])
            fcmat = fcmat / (dl_list[-2] ** 2.0)  # Final fwd matrix
            weights_fwd_str.append(fcmat)
            #lat
            nzelems = [np.count_nonzero(s_l) for s_l in s_lat_list]

            weights_lat_fc = [initialize_layer(nznum, dn, npsp, nrow, nrow, pd_bool=True) for nznum, dn, npsp, nrow in
                              zip(nzelems, d_next, nps, Vl_list[1:(-1)])]  # recurrent in all but input and output
            weights_lat_str = [np.multiply(l, str) for l, str in zip(weights_lat_fc, s_lat_list)]

            #DEBUG
            # Make sparse
            weights_fwd_str = [csc_matrix(w, dtype=theano.config.floatX) for w in weights_fwd_str]
            weights_lat_str = [csc_matrix(l, dtype=theano.config.floatX) for l in weights_lat_str]
            biases_fwd_values = [np.zeros((size,), dtype=theano.config.floatX) for size in Vl_list[1:]]

            print ('W norm: ', np.linalg.norm(np.array(weights_fwd_str[0].todense())), np.linalg.norm(np.array(weights_fwd_str[1].todense())), ' W norm perelem: ', np.linalg.norm(np.array(weights_fwd_str[0].todense()))/(Vl_list[0]*Vl_list[1]), np.linalg.norm(np.array(weights_fwd_str[1].todense()))/(Vl_list[1]*Vl_list[2]))
            print ('M norm: ', np.linalg.norm(np.array(weights_lat_str[0].todense())), np.linalg.norm(np.array(weights_lat_str[0].todense()))/(Vl_list[1]**2.0))
            training_curves = dict()
            training_curves["training error"] = list()
            training_curves["validation error"] = list()
        weights_all = dict({'fwd': [theano.shared(value=value, borrow=True) for value in weights_fwd_str],
                            'lat': [theano.shared(value=value, borrow=True) for value in weights_lat_str]})
        biases_all = dict({'fwd': [theano.shared(value=value, borrow=True) for value in biases_fwd_values]})
        struct_all= dict({'fwd': [theano.shared(value=value, borrow=True) for value in s_fwd_list],
                          'lat': [theano.shared(value=value, borrow=True) for value in s_lat_list]})

        return struct_all, biases_all, weights_all, hyperparameters, training_curves

    # SET INDEX OF THE MINI BATCH
    def __build_change_mini_batch_index(self):

        index_new = T.iscalar("index_new")

        change_mini_batch_index = theano.function(
            inputs=[index_new],
            outputs=[],
            updates=[(self.index, index_new)]
        )

        return change_mini_batch_index

    # ENERGY FUNCTION, DENOTED BY E
    def __energy(self, layers, longbool=0): #Eqn 1
        #theano.shared(scipy.sparse.csc_matrix(lay0.eval()), borrow=True)=>Code is not
        squared_norm = sum([T.batched_dot(self.rho(layer), self.rho(layer)) for layer in layers]) / 2.
        linear_terms = - sum([T.dot(self.rho(layer), b) for layer, b in zip(layers[1:], self.biases['fwd'])]) #R
        #linear_terms -= sum([T.dot(rho(layer), b) for layer, b in zip(layers[1:(-1)], self.biases['lat'])])
        quadratic_terms = - sum(
            [sparse_batched_dot(sp_b.structured_dot(self.rho(pre), W), self.rho(post)) for pre, W, post in
             zip(layers[:-1], self.weights['fwd'], layers[1:])])  # STRCHANGE
        #theano.sparse.basic.structured_dot(sp0, net.weights['fwd'][0]).eval()

        quad_lat =  sum([sparse_batched_dot(sp_b.structured_dot(self.rho(node), W), self.rho(node)) for W, node in
                                 zip(self.weights['lat'], layers[1:(-1)])]) / 2.0
            #CHECK IF ONE SHOULD BE THE TRANSPOSE, AND IF CONSTANT FACTOR OF 0.5
        if longbool==0:
            return squared_norm + linear_terms + quadratic_terms +quad_lat
        else:
            return (squared_norm + linear_terms + quadratic_terms + quad_lat), squared_norm, linear_terms, quadratic_terms, quad_lat

    # COST FUNCTION, DENOTED BY C
    def __cost(self, layers):
        return ((layers[-1] - self.y_data_one_hot) ** 2).sum(axis=1)

    # TOTAL ENERGY FUNCTION, DENOTED BY F
    def __total_energy(self, layers, beta):
        return self.__energy(layers) + beta * self.__cost(layers)

    # MEASURES THE ENERGY, THE COST AND THE MISCLASSIFICATION ERROR FOR THE CURRENT STATE OF THE NETWORK
    def __build_measure(self):

        E = T.mean(self.__energy(self.layers))
        C = T.mean(self.__cost(self.layers))
        y_prediction = T.argmax(self.layers[-1], axis=1)
        error = T.mean(T.neq(y_prediction, self.y_data))

        measure = theano.function(
            inputs=[],
            outputs=[E, C, error]
        )

        return measure

    def __build_measure_verbose(self):
        e, sq_n, lin_term, quad_term, quad_lat = self.__energy(self.layers, 1)
        E = T.mean(e)
        C = T.mean(self.__cost(self.layers))
        y_prediction = T.argmax(self.layers[-1], axis=1)
        error = T.mean(T.neq(y_prediction, self.y_data))
        Sq_n = T.mean(sq_n)
        Lin_term = T.mean(lin_term)
        Quad_term = T.mean(quad_term)
        Quad_lat = T.mean(quad_lat)

        measure = theano.function(inputs=[], outputs=[E, C, error, Sq_n, Lin_term, Quad_term, Quad_lat]) #STRCHANGE, Unrelated to str, removed on_unused_input warning

        return measure

    def __build_free_phase(self):

        n_iterations = T.iscalar('n_iterations')
        epsilon = T.fscalar('epsilon')

        def step(*layers): #Eq8, beta=0
            E_sum = T.sum(self.__energy(layers))
            layers_dot = T.grad(-E_sum, list(layers))  # temporal derivative of the state (free trajectory)
            layers_new = [layers[0]] + [self.rho(layer+epsilon*dot) for layer, dot in
                                        zip(layers, layers_dot)][1:]
            return layers_new
        #Neural Dynamics
        (layers, updates) = theano.scan(
            step,
            outputs_info=self.layers, #initial output
            n_steps=n_iterations #number of iterative updates of si
        )
        layers_end = [layer[-1] for layer in layers] #si after updating

        for particles, layer, layer_end in zip(self.persistent_particles, self.layers[1:], layers_end[1:]):
            updates[particles] = T.set_subtensor(layer, layer_end) #replaces old subtensors of layer with layer_end, assigns to updates[particles]

        free_phase = theano.function( #SV Inputs: particles (node values), the particles are updated according to updates calculated from theano scan
            inputs=[n_iterations, epsilon],
            outputs=[],
            updates=updates
        )

        return free_phase


    def __build_weakly_clamped_phase(self):
        eps_norm=np.float32(np.power(10.0, -6.))
        n_iterations = T.iscalar('n_iterations')
        epsilon = T.fscalar('epsilon')
        beta = T.fscalar('beta')
        alphas = [T.fscalar("alpha_W" + str(r + 1)) for r in range(len(self.weights['fwd']))] + [
            T.fscalar("alpha_L" + str(r + 1)) for r in range(len(self.weights['lat']))]

        alphas_fwd = alphas[:len(self.weights['fwd'])]
        alphas_lat = alphas[len(self.weights['fwd']):]

        # Neural Dynamics
        def step(*layers): #Eq42
            F_sum = T.sum(self.__total_energy(layers, beta))
            layers_dot = T.grad(-F_sum, list(layers))  # temporal derivative of the state (weakly clamped trajectory)
            layers_new = [layers[0]] + [self.rho(layer + epsilon * dot) for layer, dot in
                                        zip(layers, layers_dot)][1:]
            return layers_new

        (layers, updates) = theano.scan(
            step,
            outputs_info=self.layers,
            n_steps=n_iterations
        )
        layers_weakly_clamped = [layer[-1] for layer in layers]

        #STRCHANGE
        e_free, squared_norm, linear_terms, quadratic_terms, quad_lat = self.__energy(self.layers, 1) #These are the layers post Free relaxation
        E_mean_free = T.mean(e_free)
        e_wc, squared_norm_wc, linear_terms_wc, quadratic_terms_wc, quad_lat_wc  = self.__energy(layers_weakly_clamped, 1)
        E_mean_weakly_clamped = T.mean(e_wc)
        biases_fwd_dot = T.grad((E_mean_weakly_clamped - E_mean_free) / beta, self.biases['fwd'],
                            consider_constant=layers_weakly_clamped) #Eq10
        #biases_lat_dot = T.grad((E_mean_weakly_clamped - E_mean_free) / beta, self.biases['lat'],
        #                    consider_constant=layers_weakly_clamped) #Eq10
        weights_fwd_dot = T.grad((E_mean_weakly_clamped - E_mean_free) / beta, self.weights['fwd'],
                             consider_constant=layers_weakly_clamped) #NEED TO ENSURE ONLY CALCULATING NON ZERO ELEMENT GRADS
        weights_lat_dot = T.grad(E_mean_weakly_clamped / beta, self.weights['lat'],
                                 consider_constant=layers_weakly_clamped) #SMEP


        #biases_lat_new = [b - alpha * dot for b, alpha, dot in zip(self.biases['lat'], alphas_lat, biases_lat_dot)]
        biases_fwd_new = [b - alpha * dot for b, alpha, dot in zip(self.biases['fwd'], alphas_fwd, biases_fwd_dot)]
        weights_fwd_new = [W - alpha * dot for W, alpha, dot in zip(self.weights['fwd'], alphas_fwd, weights_fwd_dot)] #Eq13
        norm_factor=1.0/(2.0*np.abs(beta))
        weights_lat_new = [W + alpha * (dot - (norm_factor*W)) for W, alpha, dot in
                           zip(self.weights['lat'], alphas_lat, weights_lat_dot)]  #SMEP

        #Converting to dense for delta logging
        w_new_dense, l_new_dense=[theano.sparse.basic.csm_data(w) for w in weights_fwd_new], [theano.sparse.basic.csm_data(l) for l in weights_lat_new]
        w_old_dense, l_old_dense=[theano.sparse.basic.csm_data(w) for w in self.weights['fwd']], [theano.sparse.basic.csm_data(l) for l in self.weights['lat']]

        Delta_log = [T.sqrt( ((W_new - W) ** 2).mean() ) / T.sqrt( (W ** 2).mean() ) for W,W_new in zip(w_old_dense, w_new_dense)]+ [T.sqrt( ((W_new - W) ** 2).mean() ) / T.sqrt( (W ** 2).mean() ) for W,W_new in zip(l_old_dense, l_new_dense)]
        #STRCHANGE
        #Check that weights were not changed to sparse in place, using debug
        print 42
        #Add nz_check here


        #Delta_log = [T.sqrt( ((W_new - W) ** 2).mean() ) / T.sqrt( (W ** 2).mean() ) for W,W_new in zip(self.weights['fwd'],weights_fwd_new)]+ [T.sqrt( ((W_new - W) ** 2).mean() ) / T.sqrt( (W ** 2).mean() ) for W,W_new in zip(self.weights['lat'],weights_lat_new)]+[T.sqrt( ((W_new - W) ** 2).mean() ) / T.sqrt( (W ** 2).mean()+eps_norm ) for W,W_new in zip(self.weights['lat'],weights_lat_new)]
        #Signed_delta_log = [((W_new - W)/np.abs(W)).mean() for W, W_new in zip(self.weights['fwd'], weights_fwd_new)]+[((W_new - W)/np.abs(W)).mean() for W, W_new in zip(self.weights['lat'], weights_lat_new)]

        #ENSURE UPDATES IS IN A FORM WHERE CAN BE INDEXED IN BELOW FASHION
        #CHECK BELOW

        for bias, bias_new in zip(self.biases['fwd'], biases_fwd_new):
            updates[bias] = bias_new

        for weight, weight_new in zip(self.weights['fwd'], weights_fwd_new):
            updates[weight] = weight_new

        for weight, weight_new in zip(self.weights['lat'], weights_lat_new):
            updates[weight] = weight_new

        #out= T.as_tensor_variable([Delta_log])
        #out_b = T.as_tensor_variable([biases_fwd_new])
        #out_wf = T.as_tensor_variable([weights_fwd_new])
        #out_wl = T.as_tensor_variable([weights_lat_new])

        weakly_clamped_phase = theano.function( #SV Inputs: network weights, the updates are applied to the weights
            inputs=[n_iterations, epsilon, beta]+alphas,
            outputs=Delta_log, #DEBUG
            #outputs=[Delta_log[0], Delta_log[1], Delta_log[2], biases_fwd_dot[0], biases_fwd_dot[1], weights_fwd_dot[0], weights_fwd_dot[1], weights_lat_dot[0], E_mean_free, E_mean_weakly_clamped,
            #         squared_norm, squared_norm_wc, linear_terms, linear_terms_wc, quadratic_terms, quadratic_terms_wc, quad_lat, quad_lat_wc],
            updates=updates
        )

        return weakly_clamped_phase

