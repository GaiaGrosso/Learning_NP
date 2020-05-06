import numpy as np
import os
import h5py
import time
import datetime
import tensorflow as tf
import sys
import keras.backend as K
from keras.constraints import Constraint, max_norm
from keras import metrics, losses, optimizers
from keras.models import Model, Sequential
from keras.activations import relu
from keras.layers import Dense, Activation, Input, Conv1D, Flatten, Dropout, LeakyReLU, Layer, ReLU
from keras.optimizers import Adam 

#
#
#
###################### PARAMETERS set up ##########################
seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
np.random.seed(seed)
print('Random seed:'+str(seed))
N_Bkg           = 5000
N_D             = N_Bkg
N_Bkg_P         = np.random.poisson(lam=N_Bkg, size=1)
N_Bkg_p         = N_Bkg_P[0]
DIM             = int(sys.argv[2])
N_ref           = int(sys.argv[1])#500000
N_R             = N_ref
total_epochs    = int(sys.argv[4])
scale_REF       = 1.
scale_DATA      = 1.
architecture    = sys.argv[3]
layers          = architecture.split('_')[:-1]
layers          = [int(l) for l in layers]
weight_clipping = 0.08
memory_limit    = int(sys.argv[5])
file_log        = sys.argv[6]
print('N_Bkg: '+str(N_Bkg))
print('N_Bkg_Pois: '+str(N_Bkg_p))
print('Model architecture: '+architecture)

#
#
#
###################### GPU setting ##########################

print('Memory limit: %i'%(memory_limit))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
        
#
#
#
####################### Def functions #############################
def Loss(yTrue, yPred):
    return K.sum(-1.*yTrue*(yPred) + (1-yTrue)*N_D/float(N_R)*(K.exp(yPred)-1))

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
    
def ModelSigmoid(nInput, layers, weight_clipping):
    inputs = Input(shape=(nInput, ))
    dense  = Dense(layers[0], input_shape=(nInput,),
                   activation='sigmoid',
                   #W_constraint = WeightClip(weight_clipping)                                                 \
                                                                                                                
                  )(inputs)
    for l in range(len(layers)-1):
        dense  = Dense(layers[l+1], input_shape=(layers[l],),
                       activation='sigmoid',
                       W_constraint = WeightClip(weight_clipping))(dense)
    output = Dense(1, input_shape=(layers[-1],), activation='linear',
                   W_constraint = WeightClip(weight_clipping))(dense)
    model = Model(inputs=[inputs], outputs=[output])
    return model

def Build_Data_Expo(DIM, scale, N_ref, N_Bkg):
    feature = np.random.exponential(scale=scale, size=(N_ref+N_Bkg, DIM))
    feature_REF  = feature[:N_ref, :]
    feature_DATA = feature[N_ref:, :]
    target_REF   = np.zeros(N_ref)
    target_DATA  = np.ones(N_Bkg)
    return feature_REF, feature_DATA, target_REF, target_DATA

def Apply_nu(DIM, feature_DATA, nu):
    feature_DATA_i = np.copy(feature_DATA)
    feature_DATA_i[:, DIM] = nu*feature_DATA_i[:, DIM]
    return feature_DATA_i

#
#
#
######################### Trining ##################################
t0  = datetime.datetime.now()
dt1 = 0
print('time 0: '+str(datetime.datetime.now()))
with tf.device('/device:GPU:0'):
    print('\n generate data')
    #generate data
    feature_REF, feature_DATA, target_REF, target_DATA = Build_Data_Expo(DIM, 1., N_ref, N_Bkg_p)
    feature_REF  = Apply_nu(DIM-1, feature_REF, scale_REF)
    feature_DATA = Apply_nu(DIM-1, feature_DATA, scale_DATA)
    mean         = np.mean(feature_REF, axis=0)
    for i in range(feature_REF.shape[1]):
        feature_REF[:, i]  = feature_REF[:, i]*1./mean[i]
        feature_DATA[:, i] = feature_DATA[:, i]*1./mean[i]
    feature      = np.concatenate((feature_REF, feature_DATA), axis=0)
    target       = np.concatenate((target_REF, target_DATA), axis=0)

    print('\n model compile')
    # model compile
    BSMfinder = ModelSigmoid(DIM, layers=layers, weight_clipping=weight_clipping)
    BSMfinder.compile(loss = Loss,  optimizer = 'adam')
    
    t1 = datetime.datetime.now()
    print('time 1/2: '+str(datetime.datetime.now()))
    dt1 = t1-t0
    print('\nmodel fit')
    #model fit
    batch_size = feature.shape[0]
    hist       = BSMfinder.fit(feature, target, 
                               batch_size=batch_size, epochs=total_epochs, 
                               shuffle=False, verbose=0)

t2 = datetime.datetime.now()
print('time 1: '+str(datetime.datetime.now()))
print('Training time: '+str(t2-t1))
dt2 = t2-t1

#Log file
f = open(file_log, 'a')
f.write('Nref%i,DIM%i,ARC%s,Epochs%i,Tgen%isec%imicrosec,Tfit%isec:%imicrosec, Mem%i\n'%(N_ref, DIM, architecture, total_epochs, dt1.seconds, dt1.microseconds, dt2.seconds, dt2.microseconds, memory_limit))
f.close()

