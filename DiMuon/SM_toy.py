#from __future__ import division                                                                                                                              
import json
import numpy as np
import os, sys
import h5py
import time
import datetime
import sys
import matplotlib as mpl
mpl.use('Agg') # avoid pyplot show interface
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import metrics, losses, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Layer

from DATAutils import *
from NNutils import *

parser = argparse.ArgumentParser()                                                  
parser.add_argument('-j', '--jsonfile'  , type=str, help="json file", required=True)
args = parser.parse_args()

# random seed ###############################
seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
np.random.seed(seed)
print('Random seed:'+str(seed))

# experiment set up ########################
with open(args.jsonfile, 'r') as jsonfile:
        config_json = json.load(jsonfile)

columns_training = ['leadmupt', 'subleadmupt', 'leadmueta', 'subleadmueta', 'delta_phi', # 5D analysis
                    'mass', # just used for cutting
                    'weight'
                   ]
N_Bkg            = config_json["N_Bkg"]
N_D              = N_Bkg
mass_cut         = config_json["Mcut"]

#### Architecture: #########################                                                                                                                  
BSM_architecture = config_json["BSMarchitecture"]
BSM_wclip        = config_json["BSMweight_clipping"]
patience         = config_json["patience"]
total_epochs     = config_json["epochs"]

##### define output path ######################                                                                                                               
OUTPUT_PATH    = config_json["output_directory"]
OUTPUT_FILE_ID = '/Toy5D_seed'+str(seed)+'_patience'+str(patience)

# do not run the job if the toy label is already in the folder                                                                                                
if os.path.isfile("%s/%s_t.txt" %(OUTPUT_PATH, OUTPUT_FILE_ID)):
    exit()

# nuisance values ###########################
# correction mode:
correction       = config_json["correction"] 
# shape effects:                                                                                                                                               
shape_generation = np.array(config_json["shape_nuisances_generation"]) # units of sigma                                                                       
shape_reference  = np.array(config_json["shape_nuisances_reference"])
shape_sigma      = np.array(config_json["shape_nuisances_sigma"])
shape_auxiliary  = []
for i in range(len(shape_sigma)):
    if shape_sigma[i]:
        shape_auxiliary.append(np.random.normal(shape_generation[i], shape_sigma[i], size=(1,))[0])
    else:
        shape_auxiliary.append(0)
shape_auxiliary = np.array(shape_auxiliary)
shape_dictionary_list = config_json["shape_dictionary_list"]

# global normalization                                                                                                                                        
norm_generation = config_json["norm_nuisances_generation"] # units of sigma                                                                                  \
                                                                                                                                                              
norm_reference  = config_json["norm_nuisances_reference"]
norm_sigma      = config_json["norm_nuisances_sigma"]
if norm_sigma:
        norm_auxiliary  = np.random.normal(norm_generation, norm_sigma, size=(1,))[0]
else:
        norm_auxiliary = 0

N_Bkg_P = np.random.poisson(lam=N_Bkg*np.exp(norm_generation*norm_sigma), size=1)
N_Bkg_p = N_Bkg_P[0]
print('N_Bkg: '+str(N_Bkg))
print('N_Bkg_Pois: '+str(N_Bkg_p))

# cross section normalization                                                                                                                                 
csec_nuisances_data       = config_json["csec_nuisances_data"]
csec_nuisances_reference  = config_json["csec_nuisances_reference"]
csec_nuisances_sigma      = config_json["csec_nuisances_sigma"]

##### Read and build training data ###############################                                                                                                               
INPUT_PATH_REF = '.../MC/'
DATA   = np.array([])
REF    = np.array([])
W_DATA = np.array([])
W_REF  = np.array([])
Y_DATA = np.array([])
Y_REF  = np.array([])
M_DATA = np.array([])
M_REF  = np.array([])

read_file_total_R = np.array([])
read_file_total_D = np.array([])
# build sample for reference
for process in trim_list:
        #cross section uncertainty factor to generate the reference (exponential parametrization, usually 1)                                                  
        cross_sx_nu_R = np.exp(csec_nuisances_reference[process]*csec_nuisances_sigma[process])
        f = h5py.File(INPUT_PATH_REF+process+'.h5', 'r')
        read_file = np.array([])
        for p in columns_training:
                col = np.array(f.get(p))
                if p =='weights':
                        col = col*cross_sx_nu_R
                col = np.expand_dims(col, axis=1)
                if read_file.shape[0]==0:
                        read_file = col
                else:
                        read_file = np.concatenate((read_file, col), axis=1)
        f.close()
        if read_file_total_R.shape[0]==0:
                read_file_total_R = read_file
        else:
                read_file_total_R = np.concatenate((read_file_total_R, read_file), axis=0)
        #print('process: %s --> number of simulations: %i, yield: %f'%(process, read_file.shape[0], np.sum(read_file[:, -1])))
        
# build sample for SM-like data
for process in trim_list:
        #cross section uncertainty factor to generate the data sample (exponential parametrization)                                                           
        cross_sx_nu_D =np.exp(csec_nuisances_data[process]*csec_nuisances_sigma[process])
        f = h5py.File(INPUT_PATH_REF+process+'.h5', 'r')
        read_file = np.array([])
        for p in columns_training:
                col = np.array(f.get(p))
                if p =='weights':
                        col = col*cross_sx_nu_D                                                                                                              \

                col = np.expand_dims(col, axis=1)
                if read_file.shape[0]==0:
                        read_file = col
                else:
                        read_file = np.concatenate((read_file, col), axis=1)
        f.close()
        if read_file_total_D.shape[0]==0:
                read_file_total_D = read_file
        else:
                read_file_total_D = np.concatenate((read_file_total_D, read_file), axis=0)
read_file_total = np.concatenate((read_file_total_R, read_file_total_D), axis=1)

# shuffle in to randomize the unweighting procedure                                                                                                           
np.random.shuffle(read_file_total)
read_file_total_R = read_file_total[:, :len(columns_training)]
read_file_total_D = read_file_total[:, len(columns_training):]

# unweighting events for SM-like DATA                                                                                                                         
N_file = read_file_total_D.shape[0]
REF    = read_file_total_D[:, :-2]
W_REF  = read_file_total_D[:, -1:]
M_REF  = read_file_total_D[:, -2]
mask   = M_REF>mass_cut
REF    = REF[mask]
W_REF  = W_REF[mask]
# consider only events with weights in [0, 1] (this can be done if the rejected events are a negligible fraction)                                             
mask   = (W_REF[:, 0]>0)*(W_REF[:, 0]<1)
REF    = REF[mask]
W_REF  = W_REF[mask]
f_MAX  = np.max(W_REF)
W_REF_TOT = np.sum(W_REF)
print('f_MAX: %f'%(f_MAX))
DATA_idx = np.array([])
N_REF  = REF.shape[0]
N_DATA = N_Bkg_p
if N_REF<N_DATA:
    print('Cannot produce %i events; only %i available'%(N_DATA, N_REF))
    exit()
else:
        i = 0
        while DATA.shape[0]<N_DATA:
                x = REF[i:i+1, :]
                f = W_REF[i, :]
                if f<0:
                        print('f<0')
                        DATA_idx = np.append(DATA_idx, i) #neglect the negative weights both in the DATA and the REF (then they will be considered when train\
ing against Real Data)                                                                                                                                        
                        i+=1
                        continue
                r = f/f_MAX
                if r>=1:
                        if DATA.shape[0]==0:
                                DATA = x
                                DATA_idx = np.append(DATA_idx, i)
                        else:
                                DATA = np.concatenate((DATA, x), axis=0)
                                DATA_idx = np.append(DATA_idx, i)
                else:
                        u = np.random.uniform(size=1)
                        if u<= r:
                                if DATA.shape[0]==0:
                                        DATA = x
                                        DATA_idx = np.append(DATA_idx, i)
                                else:
                                        DATA = np.concatenate((DATA, x), axis=0)
                                        DATA_idx = np.append(DATA_idx, i)
                i+=1
                if i>=REF.shape[0]:
                        print('End of file')
                        N_DATA = DATA.shape[0]
                        break
# remove from REF the events selected as DATA                                                                                                                 
REF    = read_file_total_R[:, :-2]
W_REF  = read_file_total_R[:, -1:]
M_REF  = read_file_total_R[:, -2]
mask   = (M_REF>mass_cut)*(W_REF[:, 0]>0)*(W_REF[:, 0]<1)
REF    = REF[mask]
W_REF  = W_REF[mask]
REF    = np.delete(REF, DATA_idx, 0)
W_REF  = np.delete(W_REF, DATA_idx, 0)

N_REF  = REF.shape[0]
W_REF_TOT = np.sum(W_REF)

Y_REF  = np.zeros_like(REF[:, 0:1])
W_REF  = W_REF*(N_DATA*1./W_REF_TOT)
Y_DATA = np.ones_like(DATA[:, 0:1])
W_DATA = np.ones_like(DATA[:, 0:1])

feature = np.concatenate((REF, DATA), axis=0)
weights = np.concatenate((W_REF, W_DATA), axis=0)
target  = np.concatenate((Y_REF, Y_DATA), axis=0)

print('N(R): %f'%(np.sum(W_DATA)))
print('N_ref: %f'%(np.sum(W_REF)))
print('Nsim(R): %f'%(W_DATA.shape[0]))
print('Nsim_Ref: %f'%(W_REF.shape[0]))

target  = np.concatenate((target, weights), axis=1)

'''                                                                                                                                                           
i=0                                                                                                                                                           
bins_dict = {                                                                                                                                                 
    'leadmupt': np.arange(0, 1200, 50),                                                                                                                       
    'subleadmupt': np.arange(0, 1200, 50),                                                                                                                    
    'leadmueta': np.arange(-2, 2, 0.2),                                                                                                                       
    'subleadmueta': np.arange(-2, 2, 0.2),                                                                                                                    
    'delta_phi': np.arange(-3.5, 3.5, 0.2),                                                                                                                   
    'mass': np.arange(500, 3000, 50)                                                                                                                          
}                                                                                                                                                             
for key in columns_training[:-1]:                                                                                                                             
        bins = bins_dict[key]                                                                                                                                 
        plt.subplots(1, 2, figsize=(18, 9))                                                                                                                   
        plt.subplot(1, 2, 1)                                                                                                                                  
        mask = target[:, 0]==1                                                                                                                                
        DATA = feature[mask]                                                                                                                                  
        REF  = feature[~mask]                                                                                                                                 
        W_DATA = weights[mask]                                                                                                                                
        W_REF  = weights[~mask]                                                                                                                               
        hD   = plt.hist(DATA[:, i], weights=W_DATA[:, 0], bins=bins, label='DATA')                                                                            
        hR   = plt.hist(REF[:, i], weights=W_REF[:, 0], histtype='step', bins=bins, label='REFERENCE', lw=3)                                                  
        hD1  = plt.hist(DATA[:, i], weights=np.ones_like(W_DATA[:, 0]), bins=bins,  alpha=0.)                                                                 
        hR1  = plt.hist(REF[:, i], weights=np.ones_like(W_REF[:, 0]), bins=bins, alpha=0.)                                                                    
        plt.legend(fontsize=18)                                                                                                                               
        plt.xlabel(key, fontsize=18)                                                                                                                          
        plt.yscale('log')                                                                                                                                     
        plt.ylim(0, 1.1*np.max(hD[0]))                                                                                                                        
        plt.subplot(1, 2, 2)                                                                                                                                  
        x = 0.5*(bins[1:]+bins[:-1])                                                                                                                          
        plt.errorbar(x, hD[0]/hR[0], yerr=hD[0]/hR[0]*np.sqrt(1/hD1[0]+1/hR1[0]), ls='', marker='o', label ='DATA/REF')                                       
        plt.plot(x, np.ones_like(x), ls='--')                                                                                                                 
        plt.legend(fontsize=18)                                                                                                                               
        plt.xlabel(key, fontsize=18)                                                                                                                          
        plt.plot()                                                                                                                                            
        plt.show()                                                                                                                                            
        i+=1                                                                                                                                                  
exit()                                                                                                                                                        
'''
#standardize dataset #######################                                                                                                                  
for j in range(feature.shape[1]):
    vec  = feature[:, j]
    mean = np.mean(vec)
    std  = np.std(vec)
    if np.min(vec) < 0:
        vec = vec- mean
        vec = vec *1./ std
    elif np.max(vec) > 1.0:# Assume data is exponential -- just set mean to 1.                                                                                
        vec = vec *1./ mean
    feature[:, j] = vec

#### training tau ###############################                                                                                                             
batch_size = feature.shape[0]
input_shape = (None, BSM_architecture[0])
BSMfinder  = NPLM_imperfect(input_shape,
                            NU_S=shape_reference, NUR_S=shape_reference, NU0_S=shape_auxiliary, SIGMA_S=shape_sigma,
                            NU_N=norm_reference, NUR_N=norm_reference, NU0_N=norm_auxiliary, SIGMA_N=norm_sigma,
                            shape_dictionary_list=shape_dictionary_list,
                            BSMarchitecture=BSM_architecture, BSMweight_clipping=BSM_wclip, correction=correction,
                            train_nu=True, train_f=True)

BSMfinder.compile(loss = NPLM_Imperfect_Loss,  optimizer = 'adam')
hist = BSMfinder.fit(feature, target, batch_size=batch_size, epochs=total_epochs, verbose=False)

##### OUTPUT ################################                                                                                                                 
# test statistic                                                                                                                                              
loss = np.array(hist.history['loss'])
final_loss = loss[-1]
t_OBS      = -2*final_loss
print('tau: %f'%(t_OBS))
# save t                                                                                                                                                      
log_t = OUTPUT_PATH+OUTPUT_FILE_ID+'_tau.txt'
out   = open(log_t,'w')
out.write("%f\n" %(t_OBS))
out.close()
# write the loss history                                                                                                                                      
log_history = OUTPUT_PATH+OUTPUT_FILE_ID+'_TAU_history'+str(patience)+'.h5'
f           = h5py.File(log_history,"w")
epoch       = np.array(range(total_epochs))
keepEpoch   = epoch % patience == 0
f.create_dataset('epoch', data=epoch[keepEpoch], compression='gzip')
for key in list(hist.history.keys()):
    monitored = np.array(hist.history[key])
    print('%s: %f'%(key, monitored[-1]))
    f.create_dataset(key, data=monitored[keepEpoch],   compression='gzip')
f.close()
# save the model                                                                                                                                                                                                                                                                
log_weights = OUTPUT_PATH+OUTPUT_FILE_ID+'_TAU_weights.h5'
BSMfinder.save_weights(log_weights)

####################################################
#### training Delta ################################                                                                                                          
total_epochs_d = 2*total_epochs
batch_size = feature.shape[0]
input_shape = (None, BSM_architecture[0])
BSMfinder  = NPLM_imperfect(input_shape,
                            NU_S=shape_reference, NUR_S=shape_reference, NU0_S=shape_auxiliary, SIGMA_S=shape_sigma,
                            NU_N=norm_reference, NUR_N=norm_reference, NU0_N=norm_auxiliary, SIGMA_N=norm_sigma,
                            shape_dictionary_list=shape_dictionary_list,
                            BSMarchitecture=BSM_architecture, BSMweight_clipping=BSM_wclip, correction=correction,
                            train_nu=True, train_f=False)

opt  = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00000001)
BSMfinder.compile(loss = NPLM_Imperfect_Loss,  optimizer = opt)
hist = BSMfinder.fit(feature, target, batch_size=batch_size, epochs=total_epochs_d, verbose=False)
##### OUTPUT ################################                                                                                                                                                                                                                                                            
# test statistic                                                                                                                                                                                                                                                                                              
loss = np.array(hist.history['loss'])
final_loss = loss[-1]
t_OBS      = -2*final_loss
print('Delta: %f'%(t_OBS))
# save t                                                                                                                                                                                                                                                                                     
log_t = OUTPUT_PATH+OUTPUT_FILE_ID+'_Delta.txt'
out   = open(log_t,'w')
out.write("%f\n" %(t_OBS))
out.close()
# write the loss history                                                                                                                                                                                                                                                             
log_history = OUTPUT_PATH+OUTPUT_FILE_ID+'_DELTA_history'+str(patience)+'.h5'
f           = h5py.File(log_history,"w")
epoch       = np.array(range(total_epochs_d))
keepEpoch   = epoch % patience == 0
f.create_dataset('epoch', data=epoch[keepEpoch], compression='gzip')
for key in list(hist.history.keys()):
    monitored =np.array(hist.history[key])
    print('%s: %f'%(key, monitored[-1]))
    f.create_dataset(key, data=monitored[keepEpoch],   compression='gzip')
f.close()
# save the model                                                                                                                                                                                                                                                              
log_weights = OUTPUT_PATH+OUTPUT_FILE_ID+'_DELTA_weights.h5'
BSMfinder.save_weights(log_weights)

# END ##########################################
      
