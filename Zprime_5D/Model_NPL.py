import numpy as np
import os
import h5py
import time
import datetime
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import model_from_json
import tensorflow as tf
from scipy.stats import norm, expon, chi2, uniform
from scipy.stats import chisquare
from keras.constraints import Constraint, max_norm
from keras import callbacks
from keras import metrics, losses, optimizers
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Input, Conv1D, Flatten, Dropout, LeakyReLU, Layer                                                                                                                          


class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range                                                                                                                                            
    '''
    def __init__(self, c=2):
        self.c = c
    def __call__(self, p):
        return K.clip(p, -self.c, self.c)
    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}

def NPL_Model(nInput, latentsize=5, layers=3):
    """
    nInput: number of input features
    latentsize: number of neurons in each latent layer
    layers: number of layers
    """
    inputs = Input(shape=(nInput, ))
    dense  = Dense(latentsize, input_shape=(nInput,), activation='sigmoid', W_constraint = WeightClip(weight_clipping))(inputs)
    for l in range(layers-1):
        dense  = Dense(latentsize, input_shape=(latentsize,), activation='sigmoid', W_constraint = WeightClip(weight_clipping))(dense)
    output = Dense(1, input_shape=(latentsize,), activation='linear', W_constraint = WeightClip(weight_clipping))(dense)
    model = Model(inputs=[inputs], outputs=[output])
    return model

def NPL_Model_v2(nInput, layers=[5, 5, 5]):
    """
    nInput: number of input features
    layers: list of integers descibing the length of each latent layer
    """
    inputs = Input(shape=(nInput, ))
    dense  = Dense(layers[0], input_shape=(nInput,), activation='relu', W_constraint = WeightClip(weight_clipping))(inputs)
    for l in range(len(layers)-1):
        dense  = Dense(layers[l+1], input_shape=(layers[l],), activation='relu', W_constraint = WeightClip(weight_clipping))(dense)
    output = Dense(1, input_shape=(layers[-1],), activation='linear', W_constraint = WeightClip(weight_clipping))(dense)
    model = Model(inputs=[inputs], outputs=[output])
    return model

#Loss function definition                                                                                                                                                                                             
def Loss(yTrue, yPred, N_D, N_R):
    return K.sum(-1.*yTrue*(yPred) + (1-yTrue)*N_D/float(N_R)*(K.exp(yPred)-1))
