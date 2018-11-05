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
print('STARTING BINNED TOYS TRAINING')

OUTPUT_PATH = sys.argv[1]#'/eos/user/g/ggrosso/BINNED_TOYS/M/M_1/' 
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

NP=1
toy = sys.argv[2]

#NevtMC = 1000000
NevSig = 0
NevBkg =8000
NevMC = 80000
NevData = NevSig+NevBkg
N_bins=1000
N_D = NevData
N_R = NevMC
latentsize=3
weight_clipping=7
total_epochs=200000
patience=1000

seed=datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
np.random.seed(seed)
print('Random seed:'+str(seed))

# define model                                                                                                                                                                                       
def MyModel(nInput, latentsize):
    inputs = Input(shape=(nInput, ))
    dense  = Dense(latentsize, input_shape=(nInput,), activation='sigmoid', W_constraint = WeightClip(weight_clipping))(inputs)
    output = Dense(1, input_shape=(latentsize,), activation='linear', W_constraint = WeightClip(weight_clipping))(dense)
    model = Model(inputs=[inputs], outputs=[output])
    return model

#Loss function definition                                                                                                                                                                                
def binLoss(yTrue,yPred):
    Zeros=tf.zeros_like(yTrue)
    Ones=tf.ones_like(yTrue)
    D_label= tf.where(tf.equal(yTrue, Ones), Ones, Zeros)
    R_label= tf.where(tf.equal(yTrue, Ones), Zeros, yTrue)#<--- R_label modified 10-09-18
    return K.sum(-1.*D_label*(yPred) + R_label*N_D*(K.exp(yPred)-1)) #<---K.sum() modified 06-09-18

def Loss(yTrue,yPred):
    return K.sum(-1.*yTrue*(yPred) + (1-yTrue)*N_D/float(N_R)*(K.exp(yPred)-1))

for i in range(1):
    ID='/M_BKG_patience'+str(patience)+'_ref'+str(NevMC)+'_bkg'+str(NevBkg)+'_sig'+str(NevSig)+'_epochs'+str(total_epochs)+'_latent'+str(latentsize)+'_wclip'+str(weight_clipping)
    OUTPUT_PATH = OUTPUT_PATH+ID
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    TOY_ID = '/Toy2D_patience'+str(patience)+'_'+str(NevMC)+'ref_'+str(NevBkg)+'_'+str(NevSig)+'_'+toy
#    TOY_ID = 'Toy1D_binned_'+str(NevtMC)+'_'+str(NevBkg)+'_'+str(NevSig)+'_'+sys.argv[2]
    print('Start analyzing Toy '+toy)

    # reference                                                                                                                                                      
    featureMC1 = np.random.exponential(scale=0.125, size=NevMC)
    featureMC2 = np.random.rand(NevMC,1)
    #featureMC2 = np.random.normal(loc=0.5, scale=0.1, size=NevtMC)
    featureMC1 = featureMC1.reshape((NevMC, 1))
    featureMC = np.concatenate((featureMC1, featureMC2), axis=1)
    targetMC = np.zeros_like(featureMC1)
    featureMC = np.concatenate((featureMC, targetMC), axis=1)
    print 'featureMC.shape'
    print featureMC.shape
    
    # data sample                                                                                                                                                      
    featureData1 = np.random.exponential(scale=0.125, size=NevBkg)
    featureData2 = np.random.rand(NevBkg, 1)
    #featureData2 = np.random.normal(loc=0.5, scale=0.1, size=NevBkg)

    featureSig1 = np.random.normal(loc=0.8, scale=0.02, size=NevSig)
    featureSig2 = np.random.rand(NevSig, 1)
    #featureSig2 = np.random.normal(loc=0.7, scale=0.05, size=NevSig)

    # concatenate sig + bkg                                                                                                                                           
    featureData1 = np.concatenate((featureData1, featureSig1), axis =0)
    featureData2 = np.concatenate((featureData2, featureSig2), axis =0)
    featureData1 = featureData1.reshape((NevBkg+NevSig, 1))
    featureData2 = featureData2.reshape((NevBkg+NevSig, 1))
    targetData = np.ones_like(featureData1)

    #featureData1 = np.concatenate((featureData1, targetData), axis = 1)
    #featureData2 = np.concatenate((featureData2, targetData), axis = 1)
    featureData = np.concatenate((featureData1, featureData2), axis = 1)
    featureData= np.concatenate((featureData, targetData), axis = 1) 
    print 'featureData.shape'
    print featureData.shape
    
    feature = np.concatenate((featureData, featureMC))
    np.random.shuffle(feature)

    target=feature[:, -1]
    feature=feature[:, :-1]

    BSMfinder = MyModel(feature.shape[1], latentsize)                                                                    
    BSMfinder.compile(loss = Loss,  optimizer = 'adam')#, metrics=[binLossR, binLossD])
    #print(BSMfinder.summary())

#    total_epochs=150000
    batch_size=feature.shape[0]

    hist = BSMfinder.fit(feature, target, batch_size=batch_size, epochs=total_epochs, verbose=0)
    print('Finish training Toy '+toy)
    # split the training sample in reference and data                                                                                                

    Data = feature[target==1]
    Reference = feature[target!=1]
    #print(Data.shape, Reference.shape)

    # inference                                                                                                  
    f_Data = BSMfinder.predict(Data, batch_size=batch_size)                                                   
    f_Reference = BSMfinder.predict(Reference, batch_size=batch_size)                                             
#    f_All = BSMfinder.predict(feature, batch_size=batch_size)

    # metrics
    loss = np.array(hist.history['loss'])

    # test statistic
    final_loss=loss[-1]
    t_e_OBS = -2*final_loss#*feature.shape[0]

    #ideal test statistic
    Data = Data[:, 0]
    # NP1
    Ns=10
    loc=0.8
    scale=0.02
    ratio1 = (1 + Ns*(1./NevBkg)*np.divide(norm.pdf(Data, loc=loc, scale=scale),
                                        expon.pdf(Data, scale=0.125))
                              )
    t_ID_NP1 = 2*(np.sum(np.log(ratio1))-Ns)

    # NP2
    Ns=90
    ratio2 = (1 + Ns*(1./NevBkg)*np.divide(256.*np.multiply(np.multiply(Data, Data),np.exp(-8.*Data)),
                                        expon.pdf(Data, scale=0.125))
                              )
    t_ID_NP2 = 2*(np.sum(np.log(ratio2))-Ns)
    
    # NP3
    Ns=35
    loc=0.2
    scale=0.02
    ratio3 = (1 + Ns*(1./NevBkg)*np.divide(norm.pdf(Data, loc=loc, scale=scale),
                                        expon.pdf(Data, scale=0.125))
                              )
    t_ID_NP3 = 2*(np.sum(np.log(ratio3))-Ns)

    # save t
    log_t = OUTPUT_PATH+TOY_ID+'_t.txt'
    out = open(log_t,'w')
    out.write("%f\n" %(t_e_OBS))
    out.close()

    log_tid =OUTPUT_PATH+TOY_ID+'_tID_NP1.txt'
    out = open(log_tid,'w')
    out.write("%f\n" %(t_ID_NP1))
    out.close()

    log_tid =OUTPUT_PATH+TOY_ID+'_tID_NP2.txt'
    out = open(log_tid,'w')
    out.write("%f\n" %(t_ID_NP2))
    out.close()

    log_tid =OUTPUT_PATH+TOY_ID+'_tID_NP3.txt'
    out = open(log_tid,'w')
    out.write("%f\n" %(t_ID_NP3))
    out.close()

    # write the loss history                                                                                                                         
    log_history =OUTPUT_PATH+TOY_ID+'_history.h5'
    f = h5py.File(log_history,"w")
    keepEpoch = np.array(range(total_epochs))
    keepEpoch = keepEpoch % patience == 0
    f.create_dataset('loss', data=loss[keepEpoch], compression='gzip')
    f.close()

    # save the model       
    log_model =OUTPUT_PATH+TOY_ID+'_model.json'                                                                                                                          
    log_weights =OUTPUT_PATH+TOY_ID+'_weights.h5'
    model_json = BSMfinder.to_json()
    with open(log_model, "w") as json_file:
        json_file.write(model_json)
    
    BSMfinder.save_weights(log_weights)
    '''
    # save outputs                                                                                                                                   
    log_predictions =OUTPUT_PATH+TOY_ID+'_predictions.h5'
    f = h5py.File(log_predictions,"w")
    f.create_dataset('featureData', data=f_Data, compression='gzip')
    f.create_dataset('featureMC', data=f_Reference, compression='gzip')
    f.create_dataset('featureAll', data=f_All, compression='gzip')
    f.create_dataset('targetAll', data=target, compression='gzip')
    f.create_dataset('featureAll_initial', data=feature, compression='gzip')
    f.close()
    '''
    print('Output saved for Toy '+sys.argv[2])
    print('----------------------------\n')
