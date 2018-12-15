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
#from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, ProgbarLogger                                                                                      
from keras.utils import plot_model

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


class SaturationCounting(callbacks.Callback):
    """Count number of parameters that have achieved the WeightClipping threshold
    """
    def __init__(self, patience, Wclipping, path, save_weights):
        super(SaturationCounting, self).__init__()
        self.patience = patience
        self.Wclipping = Wclipping
        self.count = 0
        self.path = path
        self.save_weights = save_weights

    def on_train_begin(self, logs={}):
        self.counts = []
        self.epochs = []
        return
    
    def on_epoch_end(self, epoch, logs={}):
        if not epoch%self.patience:
            # layer 1
            W = self.model.get_weights()[0]
            W_saturated=np.sum(np.all(W==self.Wclipping, axis=1).astype(int))+np.sum(np.all(W==-self.Wclipping, axis=1).astype(int))
            self.count = self.count+W_saturated
            # layer 2
            W = self.model.get_weights()[2]
            W_saturated=np.sum(np.all(W==self.Wclipping, axis=0).astype(int))+np.sum(np.all(W==-self.Wclipping, axis=0).astype(int))
            self.count = self.count+W_saturated
            self.epochs.append(epoch)
            self.counts.append(self.count)
            if self.save_weights:
                self.model.save_weights(self.path+'_weights'+str(epoch)+'.h5')
            return
        else:
            return
        
    def on_train_end(self, logs={}):
        self.epochs=np.array(self.epochs)
        self.counts=np.array(self.counts)
        if self.save_weights:
            f = h5py.File(self.path+'_SaturationCount.h5',"w")
            f.create_dataset('epoch', data=self.epochs, compression='gzip')
            f.create_dataset('counts', data=self.counts, compression='gzip')
            f.close()
        return


print('STARTING BINNED TOYS TRAINING')
#toy
toy = sys.argv[2]
#feature label
#feature_label = int(sys.argv[3])

#N_D = 10000
N_Sig = 0
N_Bkg = 5000#N_D - N_Sig
N_D = N_Bkg + N_Sig
#N_bins = 1000
N_ref = 50000
N_R = N_ref
total_epochs=200000
latentsize=5

patience=1000
weight_clipping=10

seed=datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
np.random.seed(seed)
print('Random seed:'+str(seed))

feature_name=['Lep1Px', 'lep1Pz', 'Lep1IsoCh', 'Lep1IsoGamma', 'Lep1IsoNeu',
       'Lep1Charge', 'Lep1IsEle', 'Lep2Px', 'Lep2Py', 'Lep2Pz',
       'Lep2IsoCh', 'Lep2IsoGamma', 'Lep2IsoNeu', 'Lep2Charge',
       'Lep2IsEle', 'Jet1Px', 'Jet1Py', 'Jet1Pz', 'Jet1Mass', 'Jet1Btag',
       'Jet2Px', 'Jet2Py', 'Jet2Pz', 'Jet2Mass', 'Jet2BTag', 'HT', 'METx',
       'METy', 'nJets', 'nBjets', 'Mll', 'MT_l1MET', 'MT_l2MET',
       'MT_llMET', 'Mjj', 'Mjjll', 'MT_j1MET', 'MT_j2MET', 'MT_jjMET']

#INPUT_PATH_REF = '/eos/user/g/ggrosso/BINNED/Zsamples/Zref/'
#INPUT_PATH_REF = '/eos/project/d/dshep/BSM_Detection/Zref/'
#INPUT_PATH_BKG = '/eos/project/d/dshep/BSM_Detection/Zmumu_lepFilter_13TeV'

def MyModel(nInput, latentsize):
    inputs = Input(shape=(nInput, ))
    dense  = Dense(latentsize, input_shape=(nInput,), activation='sigmoid', W_constraint = WeightClip(weight_clipping))(inputs)
    output = Dense(1, input_shape=(latentsize,), activation='linear', W_constraint = WeightClip(weight_clipping))(dense)
    model = Model(inputs=[inputs], outputs=[output])
    return model
#Loss function definition                                                                                                                      
def Loss(yTrue,yPred):
    return K.sum(-1.*yTrue*(yPred) + (1-yTrue)*N_D/float(N_R)*(K.exp(yPred)-1))
'''
def Loss(yTrue,yPred):
    Zeros=tf.zeros_like(yTrue)
    Ones=tf.ones_like(yTrue)
    D_label= tf.where(tf.equal(yTrue, Ones), Ones, Zeros)
    R_label = tf.where(tf.equal(yTrue, Ones), Zeros, Ones)
    return K.sum(-1.*D_label*(yPred) + R_label*(N_D)*(K.exp(yPred)-1))

def binLossR(yTrue,yPred):
    Zeros=tf.zeros_like(yTrue)
    Ones=tf.ones_like(yTrue)
    R_label = tf.where(tf.equal(yTrue, Ones), Zeros, yTrue)
    return R_label*(N_D)*(K.exp(yPred)-1)

def binLossD(yTrue,yPred):
    Zeros=tf.zeros_like(yTrue)
    Ones=tf.ones_like(yTrue)
    D_label= tf.where(tf.equal(yTrue, Ones), Ones, Zeros)
    return -1.*D_label*(yPred)
'''
# define input path                                                                                                                                                                                                                  
#INPUT_PATH = '/eos/user/g/ggrosso/BINNED/Zsamples'
#ID='/Z_bins'+str(N_bins)+'_bkg'+str(N_Bkg)+'_sig'+str(N_Sig)
#INPUT_PATH = INPUT_PATH + ID
#INPUT_FILE_ID = ID+'_toy'+toy

# define output path
OUTPUT_PATH = sys.argv[1]
#f_id=feature_name[feature_label]
ID='/Z_5D_patience'+str(patience)+'_ref'+str(N_ref)+'_bkg'+str(N_Bkg)+'_sig'+str(N_Sig)+'_epochs'+str(total_epochs)+'_latent'+str(latentsize)+'_wclip'+str(weight_clipping)
OUTPUT_PATH = OUTPUT_PATH+ID
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
OUTPUT_FILE_ID = '/Toy5D_patience'+str(patience)+'_'+str(N_ref)+'ref_'+str(N_Bkg)+'_'+str(N_Sig)+'_'+toy
print('Start analyzing Toy '+sys.argv[2])

#random integer to select Zprime file between 0 and 9                                                                                                       
u = np.random.randint(10, size=10)
u1 = u[0]
v = np.random.randint(1000, size=1000)
v1 = v[0]

#reference
INPUT_PATH_BKG = '/eos/project/d/dshep/BSM_Detection/Zmumu_lepFilter_13TeV/'

HLF_REF = np.array([])
HLF_name = ''

#for fileIN in glob.glob("%s/*.h5" %INPUT_PATH_BKG):
i=0
for v_i in v:
#    print(str(v_i))
    f = h5py.File(INPUT_PATH_BKG+'Zmumu_13TeV_20PU_'+str(v_i)+'.h5')
#    f = h5py.File(fileIN)
    hlf = np.array(f.get('HLF'))
    print(hlf.shape)
    hlf_names = f.get('HLF_names')
    if not hlf_names:
        continue
    #n_events=np.min([hlf.shape(0), N_ref+N_Bkg])
    cols = [0, 1, 7, 8, 9]
    if i==0:
        HLF_REF=hlf[:, cols]
        i=i+1
        #HLF_REF=np.expand_dims(HLF_REF, axis=1)
    else:
        HLF_REF=np.concatenate((HLF_REF, hlf[:, cols]), axis=0)
    f.close()
    print(HLF_REF.shape)
    if HLF_REF.shape[0]>=N_ref+N_Bkg:
        HLF_REF=HLF_REF[:N_ref+N_Bkg, :]
        break
print('HLF_REF+BKG shape')
print(HLF_REF.shape)
#sig+bkg
INPUT_PATH_SIG = '/eos/project/d/dshep/BSM_Detection/Zprime_lepFilter_13TeV/'

#HLF_BKG = HLF[N_ref:, :]
#HLF_REF = HLF[:N_ref, :]

HLF_SIG = np.array([])
HLF_SIG_name = ''
i=0
#for fileIN in glob.glob("%s/*.h5" %INPUT_PATH_SIG):
for u_i in u:
    f = h5py.File(INPUT_PATH_SIG+'Zprime_lepFilter_13TeV_'+str(u_i)+'.h5')
 #   f = h5py.File(fileIN)
    hlf = np.array(f.get('HLF'))
    hlf_names = f.get('HLF_names')
    if not hlf_names:
        continue
#    n_events=np.max([hlf.shape(0), N_Sig])
    cols=[0, 1, 7, 8, 9]
    if i==0:
        HLF_SIG=hlf[:, cols]
        i=i+1
    else:
        HLF_SIG=np.concatenate((HLF_SIG, hlf[:, cols]), axis=0)
    f.close()
    if HLF_SIG.shape[0]>=N_Sig :
        HLF_SIG=HLF_SIG[:N_Sig, :]
        break
print('HLF_SIG shape')
print(HLF_SIG.shape)
#HLF_DATA=np.concatenate((HLF_BKG, HLF_SIG), axis=0)

# datasets
#f = h5py.File(INPUT_PATH+INPUT_FILE_ID+'.h5')
#feature = f.get("feature")
target_REF=np.zeros(N_ref)
print('target_REF shape ')
print(target_REF.shape)
target_DATA=np.ones(N_Bkg+N_Sig)
print('target_DATA shape ')
print(target_DATA.shape)
target = np.append(target_REF, target_DATA)
target = np.expand_dims(target, axis=1)
print('target shape ')
print(target.shape)
feature=np.concatenate((HLF_REF, HLF_SIG), axis=0)

#5D features construction
p1 = np.sqrt(np.multiply(feature[:, 0], feature[:, 0])+np.multiply(feature[:, 1], feature[:, 1]))
pt1 = feature[:, 0]
pt2 = np.sqrt(np.multiply(feature[:, 2], feature[:, 2])+np.multiply(feature[:, 3], feature[:, 3]))
p2 = np.sqrt(np.multiply(pt2, pt2)+np.multiply(feature[:, 4], feature[:, 4]))
eta1 = np.arctanh(np.divide(feature[:, 1], p1))
eta2 = np.arctanh(np.divide(feature[:, 4], p2))
delta_phi = np.arctan(np.divide(feature[:, 3], feature[:, 2]))+np.pi*0.5*(1-np.sign(feature[:, 2]))

pt1=np.expand_dims(pt1, axis=1)
pt2=np.expand_dims(pt2, axis=1)
eta1=np.expand_dims(eta1, axis=1)
eta2=np.expand_dims(eta2, axis=1)
delta_phi=np.expand_dims(delta_phi, axis=1)

final_features=np.concatenate((pt1, pt2), axis=1)
final_features=np.concatenate((final_features, eta1), axis=1)
final_features=np.concatenate((final_features, eta2), axis=1)
final_features=np.concatenate((final_features, delta_phi), axis=1)
print('final_features shape ')
print(final_features.shape)

feature=np.concatenate((final_features, target), axis=1)
np.random.shuffle(feature)
print('feature shape ')
print(feature.shape)
target=feature[:, -1]
feature=feature[:, :-1]
'''
#standardize dataset
file_name = 'Zreference_'+str(N_bins)+'BINS_'+feature_name[feature_label]+'.h5'
f_ref = h5py.File(INPUT_PATH_REF+file_name)
Xf_ref = f_ref.get('x_binned')
yf_ref = f_ref.get('probability_binned')
Xf_ref = np.array(Xf_ref)
yf_ref = np.array(yf_ref)
f_ref.close()

mean=np.average(Xf_ref, weights=yf_ref)
std=np.sqrt(np.average((Xf_ref-mean)*(Xf_ref-mean), weights=yf_ref))

vec = feature
if np.min(vec) < 0:
    vec = vec- mean
    vec = vec / std
elif np.max(vec) > 1.0:# Assume data is exponential -- just set mean to 1.                                                                      
    vec = vec / mean
feature = vec

print(target.shape, feature.shape)
'''
# training
batch_size=feature.shape[0]
BSMfinder = MyModel(feature.shape[1], latentsize)
BSMfinder.compile(loss = Loss,  optimizer = 'adam')#, metrics=[binLossR, binLossD])
hist = BSMfinder.fit(feature, target, batch_size=batch_size, epochs=total_epochs, 
                     #callbacks = [SaturationCounting(patience=patience, Wclipping=weight_clipping, path=OUTPUT_PATH+OUTPUT_FILE_ID, save_weights=False)],
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
#lossR = np.array(hist.history['binLossR'])
#lossD = np.array(hist.history['binLossD'])

# test statistic                                                                                                                                              
final_loss=loss[-1]
#final_lossR=lossR[-1]
#final_lossD=lossD[-1]
t_e_OBS = -2*final_loss
#t_e_R = -2*feature.shape[0]*final_lossR
#t_e_D = -2*feature.shape[0]*final_lossD

# save t                                                                                                                                                            
log_t = OUTPUT_PATH+OUTPUT_FILE_ID+'_t.txt'
#log_tD =OUTPUT_PATH+OUTPUT_FILE_ID+'_tD.txt'
#log_tR =OUTPUT_PATH+OUTPUT_FILE_ID+'_tR.txt'
out = open(log_t,'w')
out.write("%f\n" %(t_e_OBS))
out.close()
#out = open(log_tD,'w')
#out.write("%f\n" %(t_e_D))
#out.close()
#out = open(log_tR,'w')
#out.write("%f\n" %(t_e_R))
#out.close()

# write the loss history                                                                                                     
log_history =OUTPUT_PATH+OUTPUT_FILE_ID+'_history'+str(patience)+'.h5'
f = h5py.File(log_history,"w")
keepEpoch = np.array(range(total_epochs))
keepEpoch = keepEpoch % patience == 0
f.create_dataset('loss', data=loss[keepEpoch], compression='gzip')
#f.create_dataset('lossR', data=lossR[keepEpoch], compression='gzip')
#f.create_dataset('lossD', data=lossD[keepEpoch], compression='gzip')
f.close()

"""
# save the model                                                                                                                                                  
log_model =OUTPUT_PATH+OUTPUT_FILE_ID+'_model.json'
log_weights =OUTPUT_PATH+OUTPUT_FILE_ID+'_weights.h5'
model_json = BSMfinder.to_json()
with open(log_model, "w") as json_file:
    json_file.write(model_json)

BSMfinder.save_weights(log_weights)

# save outputs                                                                                                                                                      
log_predictions =OUTPUT_PATH+OUTPUT_FILE_ID+'_predictions.h5'
f = h5py.File(log_predictions,"w")
f.create_dataset('feature', data=f_All, compression='gzip')
f.create_dataset('target', data=target, compression='gzip')
f.close()
"""
print('Output saved for Toy '+toy)
print('----------------------------\n')
