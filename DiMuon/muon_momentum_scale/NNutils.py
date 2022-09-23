import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import math, os, h5py
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import metrics, losses, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow import Variable
from tensorflow import linalg as la
from tensorflow.keras import initializers

def ParametricLinearLoss_c(true, pred):
    a1 = pred[:, 0]
    y  = true[:, 0]
    w  = true[:, 1]
    nu = true[:, 2]
    f  = tf.multiply(a1, nu)
    c  = 1./(1+tf.exp(f))
    return tf.reduce_mean(y*w*c**2 + (1-y)*w*(1-c)**2)

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range                                                                                                                  
    '''
    def __init__(self, c=2):
        self.c = c
    def __call__(self, p):
        return tf.clip_by_value(p, clip_value_min=-self.c, clip_value_max=self.c)
    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}

class BSMfinder(Model):
    def __init__(self,input_shape, architecture=[1, 4, 1], weight_clipping=1.0, activation='sigmoid', trainable=True, initializer=None, name=None, **kwargs):
        kernel_initializer="glorot_uniform"
        bias_initializer="zeros"
        if not initializer==None:
            kernel_initializer = initializer
            bias_initializer = initializer
        super().__init__(name=name, **kwargs)
        kernel_constraint = None
        if not weight_clipping==None: kernel_constraint = WeightClip(weight_clipping)
        self.hidden_layers = [Dense(architecture[i+1], input_shape=(architecture[i],), activation=activation, trainable=trainable,
                                    kernel_constraint=kernel_constraint, kernel_initializer=initializer, bias_initializer=initializer) for i in range(len(architecture)-2)]
        self.output_layer  = Dense(architecture[-1], input_shape=(architecture[-2],), activation='linear', trainable=trainable,
                                     kernel_constraint=kernel_constraint, kernel_initializer=initializer, bias_initializer=initializer)
        self.build(input_shape)

    def call(self, x):
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x)
        x = self.output_layer(x)
        return x
class BSMfinder_c(Model):
    def __init__(self,input_shape, architecture=[1, 4, 1], activation='sigmoid', l2=None, trainable=True, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        kernel_regularizer = None
        bias_regularizer   = None
        if not l2==None:
            kernel_regularizer = tf.keras.regularizers.L2(l2=l2, **kwargs)
            bias_regularizer = tf.keras.regularizers.L2(l2=l2, **kwargs)
        self.hidden_layers = [Dense(architecture[i+1], input_shape=(architecture[i],), activation=activation,
                                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                    trainable=trainable) for i in range(len(architecture)-2)]
        self.output_layer  =  Dense(architecture[-1], input_shape=(architecture[-2],), activation='linear',
                                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                    trainable=trainable)
        self.build(input_shape)

    def call(self, x):
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x)
        x = self.output_layer(x)
        return x


class ParametricNet(Model):
    def __init__(self, input_shape, architecture=[1, 10, 1], activation='sigmoid', l2=None, poly_degree=1,
                 initial_model=None, train_coeffs=True, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.poly_degree = poly_degree
        if not isinstance(train_coeffs, list): self.train_coeffs = [train_coeffs for _ in range(self.poly_degree)]
        else: self.train_coeffs = train_coeffs

        self.coeffs = [
            BSMfinder_c(input_shape, architecture, activation=activation, trainable=self.train_coeffs[i]) for i in range(self.poly_degree)
        ]
        self.build(input_shape)
        if not initial_model == None:
            self.load_weights(initial_model, by_name=True)

    def call(self, x):
        out = []
        for i in range(self.poly_degree):
            out.append(self.coeffs[i](x))
        if self.poly_degree == 1:
            return out[0]
        else:
            return tf.keras.layers.Concatenate(axis=1)(out)
          
def ParametricLoss_c(true, pred):
    f = pred[:, 0]
    y = true[:, 0]
    w = true[:, 1]
    c = 1./(1+tf.exp(f))
    return 100*tf.reduce_mean(y*w*c**2 + (1-y)*w*(1-c)**2)

def ParametricLoss_poly(true, pred):
    y = true[:, 0]
    w = true[:, 1]
    nu= true[:, 2]
    f = tf.zeros_like(y)
    for i in range(pred.shape[1]):
        f += pred[:, i]*(nu**(i+1))
    c = 1./(1+tf.exp(f))
    return 100*tf.reduce_mean(y*w*c**2 + (1-y)*w*(1-c)**2)

def Delta_poly(true, pred):
    y = true[:, 0]
    w = true[:, 1]
    nu= true[:, 2]
    f = tf.zeros_like(y)
    for i in range(pred.shape[1]):
        f += pred[:, i]*(nu**(i+1))
    return f.numpy()

