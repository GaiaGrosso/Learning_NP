import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import math, os, h5py
import pandas as pd

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

from utils import *
from NNutils import *


feature    = np.array([])
targets    = np.array([])
weights    = np.array([])
nuisance   = np.array([])
nu_list    = np.array([-3., -1., 1., 3.])
nu_std     = np.std(nu_list)
nu_list_std= nu_list*1./nu_std

lumi_scale = 1.
mass_cut = 120
folder = '/eos/user/g/ggrosso/PhD/DiMuon_Scouting/root_to_h5/'
np.random.seed(1234)
REF, W_REF = read_data_training_nu(folder, mass_cut, nu=0, muonpt_scale_str='0', trim_list=trim_list, columns_training=columns_training, columns_weight=co\
lumns_weight)
N_REF = REF.shape[0]
idx_REF = np.arange(N_REF)
for nu in nu_list:
    i = np.where(nu==nu_list)[0][0]
    if nu==0:
        muonpt_scale_str = '0'
    else:
        muonpt_scale_str=str(nu)
    nu_std = nu_list_std[i]
    DATA, W_DATA = read_data_training_nu(folder, mass_cut, nu=nu,
                                         muonpt_scale_str=muonpt_scale_str, trim_list=trim_list,
                                         columns_training=columns_training, columns_weight=columns_weight)
    N_DATA = DATA.shape[0]
    idx_DATA = np.arange(N_DATA)
    np.random.shuffle(idx_REF)
    np.random.shuffle(idx_DATA)
    mask_REF   = (idx_REF<int(N_REF*lumi_scale))
    mask_DATA  = (idx_DATA<int(N_DATA*lumi_scale))
    feature_nu = np.concatenate((REF[mask_REF], DATA[mask_DATA]), axis=0)
    targets_nu = np.append(np.zeros(REF[mask_REF].shape[0]), np.ones(DATA[mask_DATA].shape[0]))
    nuisanc_nu = np.ones(feature_nu.shape[0])*nu_std
    weights_nu = np.append(W_REF[mask_REF], W_DATA[mask_DATA])
    if feature.shape[0]==0:
        feature = feature_nu
        targets = targets_nu
        weights = weights_nu
        nuisance= nuisanc_nu
    else:
        feature = np.concatenate((feature, feature_nu), axis=0)
        targets = np.concatenate((targets, targets_nu), axis=0)
        weights = np.concatenate((weights, weights_nu), axis=0)
        nuisance= np.concatenate((nuisance,nuisanc_nu), axis=0)
print(feature.shape)

# check the binned distribution ratio
h_list = []
plt.figure(figsize=(10, 8))
for i in range(len(nu_list)):
    nu = nu_list[i]
    nu_std = nu_list_std[i]
    pt = feature[:, 0]
    pt_nu = pt[(targets==1)*(nuisance==nu_std)]
    w = weights[(targets==1)*(nuisance==nu_std)]
    bins = bins_dict['leadmupt']
    h = plt.hist(pt_nu, weights=w, bins=bins, label='nu: %s'%(str(nu)), histtype='step')[0]
    h_list.append(h)
    #print(h)                                                                                                                                              
plt.ylabel('Events')
plt.yscale('log')
plt.legend()
plt.close()

# data preprocessing                                                                                                                                       
mean_std = np.mean(REF, axis=0)
std_std  = np.std(REF, axis=0)
## standardize dataset
for j in range(feature.shape[1]):
    vec  = feature[:, j]
    mean = mean_std[j]
    std  = std_std[j]
    if np.min(vec) < 0:
        vec = vec- mean
        vec = vec *1./ std
    elif np.max(vec) > 1.0:# Assume data is exponential -- just set mean to 1.                                                                            \
                                                                                                                                                           
        vec = vec *1./ mean
    feature[:, j] = vec

feature = feature[:, :5] # remove mass from inputs
target  = np.stack([targets, weights, nuisance], axis=1)

# split training, validation and test                                                                                                                      
fraction_validation = 0.1
fraction_test       = 0.1
fraction_training   = 1 - fraction_validation - fraction_test

N   = feature.shape[0]
idx = np.arange(N)
np.random.seed(0)
np.random.shuffle(idx)

mask_val   = (idx<int(N*fraction_validation))
mask_test  = (idx>=int(N*fraction_validation))*(idx<int(N*(fraction_validation+fraction_test)))
mask_train = (idx>=int(N*(fraction_validation+fraction_test)))

feature_test,  target_test  = feature[mask_test],  target[mask_test]
feature_train, target_train = feature[mask_train], target[mask_train]
feature_val,   target_val   = feature[mask_val],   target[mask_val]

feature_test,  target_test  = tf.convert_to_tensor(feature_test, dtype=tf.float32),  tf.convert_to_tensor(target_test, dtype=tf.float32)
feature_train, target_train = tf.convert_to_tensor(feature_train, dtype=tf.float32), tf.convert_to_tensor(target_train, dtype=tf.float32)
feature_val,   target_val   = tf.convert_to_tensor(feature_val, dtype=tf.float32), tf.convert_to_tensor(target_val, dtype=tf.float32)
print(feature_test.shape, feature_train.shape, feature_val.shape)

# model                                                                                                                                                    
inputsize = 5 # [PT1, PT2, ETA1, ETA2, DELTA_PHI]                                                                                                          
layers    = [inputsize, 10, 10, 10, 1]
print('Layers:')
print(layers)
architecture  = ''
for l in layers:
    architecture += str(l)+'_'
nu_string = ''
for nu in nu_list:
    nu_string += str(nu)+'_'
total_epochs      = 10000
patience          = 100
wc                = 100
poly_degree       = 2
activation        = 'relu'
batch_fraction    = 0.05
gather_after = int(1./batch_fraction)

OUTPUT_PATH= '/eos/user/g/ggrosso/PhD/DiMuon_Scouting/muonpt_parametric_training_output_2018_Mcut120/'
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    print('mk %s'%(OUTPUT_PATH))
    
if inputsize != feature.shape[1]:
    logging.error('data.shape[1] must be %i'%(inputsize))
    exit()
    
input_shape  =  (None, inputsize)
model        = ParametricNet(input_shape, architecture=layers, activation=activation, 
                             poly_degree=poly_degree, name='ParNet')
print(model.summary())
optimizer = tf.keras.optimizers.Adam()

# training                                                                                                                                                 
pars_total = np.array([])
loss_total = np.array([])
loss_val_total = np.array([])
for i in range(int(total_epochs/patience)):
    clipping = WeightClip(wc)
    loss = np.array([])
    pars = np.array([])
    train_vars = model.trainable_variables
    # Create empty gradient list (not a tf.Variable list)                                                                                                  
    accum_gradient = [tf.zeros_like(this_var) for this_var in train_vars]
    losstot = 0
    for j in range(patience):
        feature_tmp, target_tmp = random_pick(feature_train, target_train, seed=j, fraction=batch_fraction)
        with tf.GradientTape() as tape:
            pred_tmp = model(feature_tmp)
            loss_value = ParametricLoss_poly(target_tmp, pred_tmp)#(target_tmp.astype(np.double), tf.cast(pred_tmp, dtype=tf.float64))                     

        losstot += loss_value
        grads = tape.gradient(loss_value, model.trainable_variables)
        # Accumulate the gradients                                                                                                                         
        accum_gradient = [(acum_grad+grad) for acum_grad, grad in zip(accum_gradient, grads)]
        # Now, after executing all the tapes you needed, we apply the optimization step                                                                    
        # (but first we take the average of the gradients)                                                                                                 
        if j%gather_after==0:
            accum_gradient_mean = [this_grad*1./gather_after for this_grad in accum_gradient]
            optimizer.apply_gradients(zip(accum_gradient_mean,train_vars))
            accum_gradient = [tf.zeros_like(this_var) for this_var in train_vars]
            losstot = 0
    for module in model.layers:
        for layer in module.layers:
            layer.set_weights([clipping(w) for w in layer.get_weights()])
    pred_val = model(feature_val)
    poly_val = Delta_poly(target_val, pred_val)
    loss_val = ParametricLoss_poly(target_val, pred_val)*1./gather_after
    for par in train_vars:
        pars = np.append(pars, par.numpy())
    pars = np.expand_dims(pars, axis=1).T
    loss = np.append(loss, losstot/patience)
    loss_val = np.array([loss_val])
    if pars_total.shape[0]==0:
        pars_total = pars
        loss_total = loss
        loss_val_total = loss_val
    else:
        pars_total = np.concatenate((pars_total, pars), axis=0)
        loss_total = np.concatenate((loss_total, loss), axis=0)
        loss_val_total = np.concatenate((loss_val_total, loss_val), axis=0)
    print('epoch: %i, loss: %f, val_loss: %f'%(int(i*patience), loss_total[-1], loss_val_total[-1]))
    # save                                                                                                                                                                                                                                                                                                   
    f = h5py.File(OUTPUT_PATH+'/%shistory.h5'%(architecture), 'w')
    f.create_dataset('pars', data=pars_total, compression='gzip')
    f.create_dataset('loss', data=loss_total, compression='gzip')
    f.create_dataset('loss_val', data=loss_val_total, compression='gzip')
    f.close()
    log_weights = '%s/polydeg%i_ARC%sNU%sweights_batchfrac%s_gradfreq%i_ep%i.h5'%(OUTPUT_PATH, poly_degree,
                                                                                  architecture, nu_string, str(batch_fraction), 
                                                                                  gather_after, i)
    model.save_weights(log_weights)
    
    # val plot                                                                                                                                             
    h_list_red = h_list
    fig      = plt.figure(figsize=(6, 6))
    fig.patch.set_facecolor('white')
    ax1    = fig.add_axes([0., 0., 1, 0.99])
    for k in range(len(nu_list)):
        nu = nu_list[k]
        nu_std = nu_list_std[k]
        x = 0.5*(bins[1:]+bins[:-1])
        y = [np.log(h_list_red[k][a]/(1e-10+h_0[a])) for a in range(len(h_0))]
        ax1.plot(x, y, label=r'muon pt scale %i $\sigma$'%(nu), color=colors[k])
        plt.text(x=15, y=0.04-0.01*k,
                 s=r'$\nu_{\rm{S}}=$ '+str(np.around(nu, 1)),
                 fontsize=21, color=colors[k], fontname="serif")
        maskR = (target_val[:, -1]==nu_std).numpy() * (target_val[:, 0]==0).numpy()
        maskD = (target_val[:, -1]==nu_std).numpy() * (target_val[:, 0]==1).numpy()
        featR = feature_val.numpy()[maskR]
        featD = feature_val.numpy()[maskD]
        deltR = np.exp(poly_val[maskR])
        weig = target_val[:, 1].numpy()
        weigR = weig[maskR]
        weigD = weig[maskD]
        hist_sumD = plt.hist(featD[:, 0]*mean_std[0], weights=weigD, bins=bins, alpha=0.)
        hist_sumW = plt.hist(featR[:, 0]*mean_std[0], weights=weigR*deltR, bins=bins, alpha=0.)
        hist_sum  = plt.hist(featR[:, 0]*mean_std[0], weights=weigR,      bins=bins, alpha=0.)
        plt.scatter(x[:-1], np.log(hist_sumW[0][:-1]/hist_sum[0][:-1]), color=colors[k], s=20, facecolors='none')
    plt.plot(x, np.zeros_like(x), color='black')
    plt.xlabel(r'$P_T$', fontsize=22, fontname="serif")
    plt.ylim(-0.05, 0.05)
    plt.xlim(10, 2500)
    plt.ylabel(r'$\log\,r(P_T,\nu_{\rm{S}})$',fontsize=26, rotation=90, labelpad=0, fontname="serif")
    plt.xticks(fontsize=16, fontname="serif")
    plt.yticks(fontsize=16, fontname="serif")
    plt.xscale('log')
    fig.savefig(log_weights.replace('.h5', '.png'))
    plt.show()                                                                                                                                            
    plt.close()
       
