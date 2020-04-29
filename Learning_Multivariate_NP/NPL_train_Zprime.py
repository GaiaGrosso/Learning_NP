#python code to run 5D test: Zprime vs. Zmumu
#from __future__ import division                                                                                                                                                  
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
from keras.utils import plot_model

from Data_Reader import BuildSample_DY
from NPL_Model import NPL_Model

#ARGS:
#output path
OUTPUT_PATH = sys.argv[1]
#toy
toy = sys.argv[2]
#signal path
INPUT_PATH_SIG = sys.argv[3] #EX: '/eos/project/d/dshep/BSM_Detection/DiLepton_Zprime300'

#background path                                                                                                                                           
INPUT_PATH_BKG = '/eos/project/d/dshep/BSM_Detection/DiLepton_SM/'

#random seed
seed=datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
np.random.seed(seed)
print('Random seed:'+str(seed))

N_Sig = 0
N_Bkg = 20000
N_ref = 100000

N_Sig_P = np.random.poisson(lam=N_Sig, size=1)
N_Sig_p = N_Sig_P[0]
print('N_Sig: '+str(N_Sig))
print('N_Sig_Pois: '+str(N_Sig_p))

N_Bkg_P = np.random.poisson(lam=N_Bkg, size=1)
N_Bkg_p = N_Bkg_P[0]
print('N_Bkg: '+str(N_Bkg))
print('N_Bkg_Pois: '+str(N_Bkg_p))

total_epochs = 200000
latentsize = 5 # number of nodes in each hidden layer
layers = 3 #number of hidden layers

patience = 5000 # number of epochs between two consecutives saving points
nfile_REF = 66 #number of files in REF repository
nfile_SIG = 1 #number of files in SIG repository

#GLOBAL VARIABLES:
weight_clipping = 2.15
N_D = N_Bkg
N_R = N_ref

# define output path
ID = '/Z_5D_patience'+str(patience)+'_ref'+str(N_ref)+'_bkg'+str(N_Bkg)+'_sig'+str(N_Sig)+'_epochs'+str(total_epochs)+'_latent'+str(latentsize)+'_wclip'+str(weight_clipping)
OUTPUT_PATH = OUTPUT_PATH+ID
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
OUTPUT_FILE_ID = '/Toy5D_patience'+str(patience)+'_'+str(N_ref)+'ref_'+str(N_Bkg)+'_'+str(N_Sig)+'_'+toy

#check if the toy label has already been used. If it is so, exit the program without repeating.
if os.path.isfile("%s/%s_t.txt" %(OUTPUT_PATH, OUTPUT_FILE_ID)):
    exit()

print('Start analyzing Toy '+toy)
#---------------------------------------

#Read Data

#BACKGROUND+REFERENCE
HLF_REF = BuildSample_DY(N_Events=N_ref+N_Bkg_p, INPUT_PATH=INPUT_PATH_BKG, seed=seed, nfiles=nfile_REF)
print(HLF_REF.shape)

#SIGNAL                                                                                                                                                                                                                
INPUT_PATH_SIG = INPUT_PATH_SIG + "/"
print("Z' input path: "+INPUT_PATH_SIG)
HLF_SIG = BuildSample_DY(N_Events=N_Sig_p, INPUT_PATH=INPUT_PATH_SIG, seed=seed, nfiles=nfile_SIG)
print(HLF_SIG.shape)

#TARGETS
target_REF = np.zeros(N_ref)
target_DATA = np.ones(N_Bkg_p+N_Sig_p)
target = np.append(target_REF, target_DATA)
target = np.expand_dims(target, axis=1)

feature = np.concatenate((HLF_REF, HLF_SIG), axis=0)
feature = np.concatenate((feature, target), axis=1)
np.random.shuffle(feature)
print('feature shape ')
print(feature.shape)
target = feature[:, -1]
feature = feature[:, :-1]

#remove MASS from the input features
feature = feature[:, :-1]

#standardize dataset                                                                                                                                                                                                  
for j in range(feature.shape[1]):
    vec = feature[:, j]
    mean = np.mean(vec)
    std = np.std(vec)
    if np.min(vec) < 0:
        vec = vec- mean
	vec = vec *1./ std
    elif np.max(vec) > 1.0:# Assume data is exponential -- just set mean to 1.                                                                                                                                        
        vec = vec *1./ mean
    feature[:, j] = vec

print('Target: ')
print(target.shape) 
print('Features: ')
print(feature.shape)
print('Start training Toy '+toy)
#--------------------------------------------------------------------

# training
batch_size=feature.shape[0]
BSMfinder = NPL_Model(feature.shape[1], latentsize, layers)
BSMfinder.compile(loss = Loss,  optimizer = 'adam')
hist = BSMfinder.fit(feature, target, batch_size=batch_size, epochs=total_epochs, 
                     verbose=0)
print('Finish training Toy '+toy)

Data = feature[target==1]
Reference = feature[target!=1]
print(Data.shape, Reference.shape)                                                                                                                               

# inference
f_Data = BSMfinder.predict(Data, batch_size=batch_size)
f_Reference = BSMfinder.predict(Reference, batch_size=batch_size)
f_All = BSMfinder.predict(feature, batch_size=batch_size)

# metrics                                                                                                                                                           
loss = np.array(hist.history['loss'])

# test statistic                                                                                                                                              
final_loss=loss[-1]
t_e_OBS = -2*final_loss

# save t                                                                                                                                                            
log_t = OUTPUT_PATH+OUTPUT_FILE_ID+'_t.txt'
out = open(log_t,'w')
out.write("%f\n" %(t_e_OBS))
out.close()

# write the loss history                                                                                                     
log_history =OUTPUT_PATH+OUTPUT_FILE_ID+'_history'+str(patience)+'.h5'
f = h5py.File(log_history,"w")
keepEpoch = np.array(range(total_epochs))
keepEpoch = keepEpoch % patience == 0
f.create_dataset('loss', data=loss[keepEpoch], compression='gzip')
f.close()

# save the model                                                                                                                                                  
log_model =OUTPUT_PATH+OUTPUT_FILE_ID+'_seed'+str(seed)+'_model.json'
log_weights =OUTPUT_PATH+OUTPUT_FILE_ID+'_seed'+str(seed)+'_weights.h5'
model_json = BSMfinder.to_json()
with open(log_model, "w") as json_file:
    json_file.write(model_json)

BSMfinder.save_weights(log_weights)
'''
# save outputs                                                                                                                                                      
log_predictions =OUTPUT_PATH+OUTPUT_FILE_ID+'_predictions.h5'
f = h5py.File(log_predictions,"w")
f.create_dataset('feature', data=f_All, compression='gzip')
f.create_dataset('target', data=target, compression='gzip')
f.close()
'''

print('Output saved for Toy '+toy)
print('----------------------------\n')
