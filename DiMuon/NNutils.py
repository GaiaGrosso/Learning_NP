import sys
import tensorflow as tf
import h5py
import os
from tensorflow import keras
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import metrics, losses, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow import Variable
from tensorflow import linalg as la
from tensorflow.keras import initializers
import numpy as np
import logging

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range                                                                                   \
                                                                                                                                                              
    '''
    def __init__(self, c=2):
        self.c = c
    def __call__(self, p):
        return tf.clip_by_value(p, clip_value_min=-self.c, clip_value_max=self.c)
    def get_config(self):
        return {'name': self.__class__.__name__, 'c': self.c}
      
      
      class BSMfinder(Model):
    def __init__(self,input_shape, architecture=[1, 4, 1], weight_clipping=None, activation='sigmoid', trainable=True, initializer=None, name=None, **kwargs)\
:
        # default initializer                                                                                                                                 
        kernel_initializer = "glorot_uniform"
        bias_initializer   = "zeros"
        if not initializer==None:
            kernel_initializer = initializer
            bias_initializer = initializer
        super().__init__(name=name, **kwargs)
        if weight_clipping ==None:
            self.hidden_layers = [Dense(architecture[i+1], input_shape=(architecture[i],), activation=activation, trainable=trainable,
                                        kernel_initializer=initializer, bias_initializer=initializer) for i in range(len(architecture)-2)]
            self.output_layer  = Dense(architecture[-1], input_shape=(architecture[-2],), activation='linear', trainable=trainable,
                                       kernel_initializer=initializer, bias_initializer=initializer)
        else:
            self.hidden_layers = [Dense(architecture[i+1], input_shape=(architecture[i],), activation=activation, trainable=trainable,
                                        kernel_constraint = WeightClip(weight_clipping),
                                        kernel_initializer=initializer, bias_initializer=initializer) for i in range(len(architecture)-2)]
            self.output_layer  = Dense(architecture[-1], input_shape=(architecture[-2],), activation='linear', trainable=trainable,
                                       kernel_constraint = WeightClip(weight_clipping),
                                       kernel_initializer=initializer, bias_initializer=initializer)
        self.build(input_shape)

    def call(self, x):
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x)
        x = self.output_layer(x)
        return x
      
      class NPLM_imperfect(Model):
    def __init__(self, input_shape, NU_S, NUR_S, NU0_S, SIGMA_S, NU_N, NUR_N, NU0_N, SIGMA_N,
                 BSMarchitecture=[1, 10, 1], BSMweight_clipping=1., correction='',
                 shape_dictionary_list=[None],
                 train_nu=True, train_f=True, name='NPLM', **kwargs):
        '''                                                                                                                                                   
        input_shape: (None, D)  #D is the number of input variables                                                                                           
        NU_S: set of shape nuisance parameters (fit to the data)                                                                                              
        NU_SR: set of shape nuisance parameters (reference, usually 0)                                                                                        
        NU_S0: set of shape nuisance parameters (value from auxiliary measures)                                                                               
        SIGMA_S: set of shape errors associated to each scale nuisance parameter                                                                              
        NU_N: set of normalization nuisance parameters (fit to the data)                                                                                     \
                                                                                                                                                              
        NU_NR: set of normalization nuisance parameters (reference, usually 0)                                                                               \
                                                                                                                                                              
        NU_N0: set of normalization nuisance parameters (value from auxiliary measures)                                                                      \
                                                                                                                                                              
        SIGMA_N: set of normalization errors associated to each normalization nuisance parameter                                                              
        N_Bkg: set of expected yelds under the reference hypothesis                                                                                           
        '''
        super().__init__(name=name, **kwargs)
        if correction not in ['SHAPE', 'NORM', '']:
            logging.error("value %s for binning is not valid. Choose between 'NN', 'BIN_POLY', 'BIN_POINTS'. ")
            exit()
        if correction=='SHAPE' and shape_dictionary_list[0]==None:
            logging.error("Missing argument 'shape_dictionary_list', required in the 'NN' correction mode")
            exit()
        if correction=='SHAPE':
            self.deltas_NN  = [] #                                                                                                                            
            self.deltas_std = [] #shape scale                                                                                                                 
            for j in range(len(shape_dictionary_list)):
                shape_dict = shape_dictionary_list[j]
                delta_poly_degree = shape_dict['poly_degree']
                delta_architectures = shape_dict['architectures']
                delta_inputsize  = delta_architectures[0][0]
                delta_input      = (None, delta_inputsize)
                delta_activation = shape_dict['activation']
                delta_wclips     = shape_dict['wclips']
                delta_std        = shape_dict['shape_std']
                delta_weights    = shape_dict['weights_file']
                self.deltas_std.append(delta_std)
                NN = ParametricNet(delta_input, delta_architectures, delta_wclips, delta_activation, degree=delta_poly_degree)
                NN.load_weights(delta_weights)
                #don't want to train Delta                                                                                                                    
                for module in NN.layers:
                    for layer in module.layers:
                        layer.trainable = False
                self.deltas_NN.append(NN)
            self.nu_s   = Variable(initial_value=NU_S,         dtype="float32", trainable=train_nu,  name='nu_s')
            self.nuR_s  = Variable(initial_value=NUR_S,        dtype="float32", trainable=False,     name='nuR_s')
            self.nu0_s  = Variable(initial_value=NU0_S,        dtype="float32", trainable=False,     name='nu0_s')
            self.sig_s  = Variable(initial_value=SIGMA_S,      dtype="float32", trainable=False,     name='sigma_s')
            self.nu_n   = Variable(initial_value=NU_N,         dtype="float32", trainable=train_nu,  name='nu_n')
            self.nuR_n  = Variable(initial_value=NUR_N,        dtype="float32", trainable=False,     name='nuR_n')
            self.nu0_n  = Variable(initial_value=NU0_N,        dtype="float32", trainable=False,     name='nu0_n')
            self.sig_n  = Variable(initial_value=SIGMA_N,      dtype="float32", trainable=False,     name='sigma_n')

        if correction == 'NORM':
            self.nu_n   = Variable(initial_value=NU_N,         dtype="float32", trainable=train_nu,  name='nu_n')
            self.nuR_n  = Variable(initial_value=NUR_N,        dtype="float32", trainable=False,     name='nuR_n')
            self.nu0_n  = Variable(initial_value=NU0_N,        dtype="float32", trainable=False,     name='nu0_n')
            self.sig_n  = Variable(initial_value=SIGMA_N,      dtype="float32", trainable=False,     name='sigma_n')
			
