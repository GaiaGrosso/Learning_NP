import sys
import numpy as np
import os
import h5py
import time
import datetime
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import model_from_json
import tensorflow as tf
from scipy.stats import norm, expon, chi2, uniform
from scipy.stats import chisquare
from keras.constraints import Constraint                                                                                                

from keras import metrics, losses, optimizers
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Input, Layer

from Utils import *

# toy  
toy  = sys.argv[2]  

seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
np.random.seed(seed)
print('Random seed: '+str(seed))


# statistics                                                                                                                                        
N_ref      = 200000
N_Bkg      = 2000
N_R        = N_ref
N_D        = N_Bkg

# nuisance                                                                                                                                          
scale_sigma = 0.3
norm_sigma  = 0.3
scale = 1 + 0 * scale_sigma
norm  = 1 + 0.5 * norm_sigma
N_Bkg_Pois = np.random.poisson(lam=N_Bkg*norm, size=1)[0]

# training time                                                                                                                                     
total_epochs = 300000
patience     = 10000

# architecture                                                                                                                                      
latentsize      = 4
layers          = 1
weight_clipping = 8

# define output path                                                                                                                                
OUTPUT_PATH = sys.argv[1]
ID          = '/EXP_1D_patience'+str(patience)+'_epochs'+str(total_epochs)+'_ref'+str(N_ref)+'_bkg'+str(N_Bkg)
ID         += '_scale'+str(scale)+'_norm'+str(norm)+'_Ssigma'+str(scale_sigma)+'_Nsigma'+str(norm_sigma)
ID         += '_latent'+str(latentsize)+'_layers'+str(layers)+'_wclip'+str(weight_clipping)
OUTPUT_PATH = OUTPUT_PATH+ID
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
OUTPUT_FILE_ID = '/EXP1D_patience'+str(patience)+'_ref'+str(N_ref)+'_bkg'+str(N_Bkg)+'_seed'+str(seed)+'_toy'+toy

# do not run the job if the toy label is already in the folder                                                                                     
if os.path.isfile("%s/%s_t.txt" %(OUTPUT_PATH, OUTPUT_FILE_ID)):
    exit()
#############################################                                                                                                       
print('Start analyzing Toy '+toy)
# data                                                                                                                                              
featureData = np.random.exponential(scale=1./8*scale, size=(N_Bkg_Pois, 1))
featureRef  = np.random.exponential(scale=1./8, size=(N_ref, 1))
feature     = np.concatenate((featureData, featureRef), axis=0)

# standardize                                                                                                                                       
mean_ref    = 1./8
feature    /= mean_ref

# target                                                                                                                                            
targetData  = np.ones_like(featureData)
targetRef   = np.zeros_like(featureRef)
target      = np.concatenate((targetData, targetRef), axis=0)

# weights                                                                                                                                           
weightData = np.ones_like(featureData)
weightRef  = np.ones_like(featureRef)*N_D*1./N_R
weights    = np.concatenate((weightData, weightRef))

target     = np.concatenate((target, weights), axis=1)
q_SCALE, m_SCALE, c_SCALE, bins, nu_fitSCALE = Read_FitBins('/eos/user/g/ggrosso/PhD/BSM/Sistematiche/Expo1D/Expo1D_BinFitSCALE.h5')
q_NORM,  m_NORM,  c_NORM,  bins, nu_fitNORM  = Read_FitBins('/eos/user/g/ggrosso/PhD/BSM/Sistematiche/Expo1D/Expo1D_BinFitNORM.h5')
Mmatrix      = np.concatenate((m_SCALE.reshape(1,-1), m_NORM.reshape(1,-1)), axis=0)
Qmatrix      = np.concatenate((q_SCALE.reshape(1,-1), q_NORM.reshape(1,-1)), axis=0)
NUmatrix     = np.concatenate((nu_fitSCALE[6:7].reshape(1,-1), nu_fitNORM[6:7].reshape(1,-1)), axis=0)
NURmatrix    = np.concatenate((nu_fitSCALE[5:6].reshape(1,-1), nu_fitNORM[5:6].reshape(1,-1)), axis=0)
NU0matrix    = np.array([[np.random.normal(loc=scale, scale=scale_sigma,size=1)[0], np.random.normal(loc=norm, scale=norm_sigma,size=1)[0]]])
SIGMAmatrix  = np.array([[scale_sigma, norm_sigma]])

batch_size   = feature.shape[0]
inputsize    = feature.shape[1]
architecture = [inputsize]
for _ in range(layers):
    architecture.append(latentsize)
architecture.append(1)
print('architecture:')
print(architecture)
input_shape = (None, inputsize)
model       = BSMfinderUpgrade(input_shape=input_shape,
                               edgebinlist=bins, mean_ref, mean_ref,
                               A1matrix=Mmatrix, A0matrix=Qmatrix,
                               NUmatrix=NUmatrix, NURmatrix=NURmatrix, NU0matrix=NU0matrix, SIGMAmatrix=SIGMAmatrix,
                               architecture=architecture, weight_clipping=weight_clipping)

model.compile(loss=NPLLoss_New,  optimizer='adam')
#print(model.summary())                                                                                                                             
hist        = model.fit(feature, target, batch_size=batch_size, epochs=total_epochs, verbose=False)
print('Finish training Toy '+toy)

################################################     
# metrics                                                                                                                                           
loss = np.array(hist.history['loss'])

# test statistic                                                                                                                                   
final_loss = loss[-1]
t_e_OBS    = -2*final_loss

# save t                                                                                                                                          
log_t = OUTPUT_PATH+OUTPUT_FILE_ID+'_t.txt'
out   = open(log_t,'w')
out.write("%f\n" %(t_e_OBS))
out.close()

# write the loss history                                                                                                                           
log_history = OUTPUT_PATH+OUTPUT_FILE_ID+'_history'+str(patience)+'.h5'
f           = h5py.File(log_history,"w")
keepEpoch   = np.array(range(total_epochs))
keepEpoch   = keepEpoch % patience == 0
f.create_dataset('loss', data=loss[keepEpoch], compression='gzip')
f.close()

# save the model                                                                 
log_weights = OUTPUT_PATH+OUTPUT_FILE_ID+'_weights.h5'                                                                                                                   
model.save_weights(log_weights)
##############################################        
