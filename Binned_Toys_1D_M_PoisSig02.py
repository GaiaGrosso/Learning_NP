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

NP=3
toy = sys.argv[2]

N_Exp_Sig = 90
# <--- Poisson N_Sig modified 06-09-18
N_Sig = np.random.poisson(lam=N_Exp_Sig, size=1)
NevSig = N_Sig[0]
NevBkg =2000
NevData = NevSig+NevBkg
N_D = NevData
N_bins=1000

latentsize=4
weight_clipping=35
total_epochs=500000
patience=1000

seed=datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
np.random.seed(seed)
print('Random seed:'+str(seed))


def NP2_gen(size, seed):
    if size>10000:
        raise Warning('Sample size is grater than 100: Generator will not approximate the tale well')
    sample = np.array([])
    #normalization factor                                                                                          
    np.random.seed(seed)
    Norm = 256.*0.25*0.25*np.exp(-2)
#    Norm = 0.5*2.*2.*np.exp(-2)
    while(len(sample)<size):
        x = np.random.uniform(0,15) #assuming not to generate more than 10 000 events                             
        p = np.random.uniform(0, Norm)
#        if p<= 0.5*x*x*np.exp(-x):
        if p<= 256.*x*x*np.exp(-8.*x):
            sample = np.append(sample, x)
    return sample

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


for i in range(1):
    ID='M_NP'+str(NP)+'_patience'+str(patience)+'_bins'+str(N_bins)+'_bkg'+str(NevBkg)+'_sig'+str(N_Exp_Sig)+'_epochs'+str(total_epochs)+'_latent'+str(latentsize)+'_wclip'+str(weight_clipping)
    OUTPUT_PATH = OUTPUT_PATH+ID
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)    TOY_ID = '/Toy1D_patience'+str(patience)+'_'+str(N_bins)+'bins_'+str(NevBkg)+'_'+str(NevSig)+'_'+toy
#    TOY_ID = 'Toy1D_binned_'+str(NevtMC)+'_'+str(NevBkg)+'_'+str(NevSig)+'_'+sys.argv[2]
    print('Start analyzing Toy '+toy)

    # create sample                                                                                                                                                     
    featureData = np.random.exponential(scale=0.125, size=(NevBkg, 1))
    #print featureData.shape                                                                                                                                            
    if NP==1:
        featureSig = np.random.normal(scale=0.02, loc=0.8, size=(NevSig, 1))
    elif NP==2:
        featureSig = NP2_gen(size=NevSig, seed=seed)
        featureSig = np.expand_dims(featureSig, axis=1)
    elif NP==3:
        featureSig = np.random.normal(scale=0.02, loc=0.2, size=(NevSig, 1))
    #print featureSig.shape                                                                                                                                             
    featureData = np.append(featureData, featureSig)
    featureData = np.expand_dims(featureData, axis = 1)
    #print featureData.shape                                                                                                                                            
    targetData=np.ones((featureData.shape[0], 1))
    featureData = np.concatenate((featureData, targetData), axis = 1)

    # binning reference generation
    ref_bins=N_bins
    ref_xmin=np.min(featureData[:, 0])
    ref_xmax=np.max(featureData[:, 0])
    ref_range=ref_xmax-ref_xmin
    ref_bin = ref_range*1./ref_bins
    ref_ns=np.array([])
    ref_xs=np.array([])
    for i in range(ref_bins):
        ref_x = ref_xmin+0.5*(ref_bin*(i)+ref_bin*(i+1))
        ref_xs = np.append(ref_xs, ref_x)
        ref_integral = expon.cdf(ref_xmin+ref_bin*(i+1), scale=0.125)-expon.cdf(ref_xmin+ref_bin*(i), scale=0.125)
        ref_ns = np.append(ref_ns, ref_integral)

    featureMC = ref_xs
    featureMC = np.expand_dims(featureMC, axis=1)
    #print featureMC.shape
    targetMC = ref_ns
    targetMC = np.expand_dims(targetMC, axis=1)
    featureMC = np.concatenate((featureMC, targetMC), axis = 1)

    feature = np.concatenate((featureData, featureMC))
    np.random.shuffle(feature)

    target=feature[:, -1]
    feature=feature[:, :-1]

    BSMfinder = MyModel(feature.shape[1], latentsize)                                                                    
    BSMfinder.compile(loss = binLoss,  optimizer = 'adam')#, metrics=[binLossR, binLossD])
    #print(BSMfinder.summary())

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


    # save seed
    log_seed = OUTPUT_PATH+TOY_ID+'_seed.txt'
    out = open(log_seed,'w')
    out.write("%f\n" %(seed))
    out.close()
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
