import os
import sys
import datetime
import tensorflow as tf
import h5py
from tensorflow import keras
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import metrics, losses, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow import Variable
import numpy as np

from NNUtils import *
from SampleUtils import *
#############################################                                                                                                                            
seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
np.random.seed(seed)
print('Random seed:'+str(seed))

N_ref   = 1000000
N_R     = N_ref
N_Bkg   = 200000
N_D     = N_Bkg
N_Bkg_P = np.random.poisson(lam=N_Bkg, size=1)
N_Bkg_p = N_Bkg_P[0]
print('N_Bkg: '+str(N_Bkg))
print('N_Bkg_Pois: '+str(N_Bkg_p))

#### Architecture: ###########################                                                                                                                            
inputsize   = 5
latentsize  = 5
n_layers    = 3
layers      = [inputsize]

for _ in range(n_layers):
    layers.append(latentsize)
layers.append(1)
print(layers)
hidden_layers = layers[1:-1]
architecture  = ''
for l in hidden_layers:
    architecture += str(l)+'_'

###############################################                                                                                                                           
patience    = 10000
wc          = 2.15
total_epochs= 300000

########## Nuisance parameters ################                                                                                                                           
endcaps_barrel_scale_r = 3
endcaps_barrel_efficiency_r = 1

sigma_sb  = 0.005
sigma_se  = endcaps_barrel_scale_r * sigma_sb
sigma_eb  = 0.025
sigma_ee  = endcaps_barrel_efficiency_r * sigma_eb

scale_barrel       = 0+3.*sigma_sb
scale_endcaps      = 0+3.*sigma_se
efficiency_barrel  = 0+0 *sigma_eb
efficiency_endcaps = 0+0 *sigma_ee
###############################################                                                                                                                          \
                                                                                                                                                                          
nfile_REF=66
nfile_SIG=1

############ CUTS ##############################                                                                                                                         
M_cut  = 100.
PT_cut = 20.
ETA_cut= 2.4

################################################                                                                                                                          
####### define output path #####################                                                                                                                          

OUTPUT_PATH = sys.argv[1]
ID ='/Z_5D_Mcut'+str(M_cut)+'_PTcut'+str(PT_cut)+ '_ETAcut'+str(ETA_cut)
ID+='_sb'+str(scale_barrel)+'_se'+str(scale_endcaps)+'_eb'+str(efficiency_barrel)+'_ee'+str(efficiency_endcaps)
ID+='_patience'+str(patience)+'_ref'+str(N_ref)+'_bkg'+str(N_Bkg)
ID+='_epochs'+str(total_epochs)+'_latent'+str(latentsize)+'_layers'+str(n_layers)+'_wclip'+str(wc)
OUTPUT_PATH = OUTPUT_PATH+ID
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
OUTPUT_FILE_ID = '/Toy5D_seed'+str(seed)+'_patience'+str(patience)+'_'+str(N_ref)+'ref_'+str(N_Bkg)

# do not run the job if the toy label is already in the folder                                                                                                           \
                                                                                                                                                                          
if os.path.isfile("%s/%s_t.txt" %(OUTPUT_PATH, OUTPUT_FILE_ID)):
    exit()

# Read data ###################################                                                                                                                           
#reference+bkg                                                                                                                                                            
INPUT_PATH_REF = '/eos/project/d/dshep/BSM_Detection/DiLepton_SM/'
HLF_REF = BuildSample_DY(N_Events=N_ref+N_Bkg_p, INPUT_PATH=INPUT_PATH_REF, seed=seed, nfiles=nfile_REF)
print(HLF_REF.shape)

HLF_BKG = HLF_REF[N_ref:, :]
HLF_BKG = Apply_MuonMomentumScale_Correction_ETAregion(HLF_BKG, muon_scale=scale_barrel, eta_min=0, eta_max=1.2)
HLF_BKG = Apply_MuonMomentumScale_Correction_ETAregion(HLF_BKG, muon_scale=scale_endcaps, eta_min=1.2, eta_max=2.4)
HLF_REF[N_ref:, :] = HLF_BKG
print(HLF_REF.shape)

target_REF=np.zeros(N_ref)
print('target_REF shape ')
print(target_REF.shape)
target_DATA=np.ones(N_Bkg_p)
print('target_DATA shape ')
print(target_DATA.shape)
target = np.append(target_REF, target_DATA)
print('target shape ')
print(target.shape)
feature = HLF_REF
print('feature shape ')
print(feature.shape)
#### CUTS #####################################                                                                                                                           
mll = feature[:, -1]
pt1 = feature[:, 0]
pt2 = feature[:, 1]
eta1= feature[:, 2]
eta2= feature[:, 3]
weights = 1*(mll>=M_cut)*(np.abs(eta1)<ETA_cut)*(np.abs(eta2)<ETA_cut)*(pt1>=PT_cut)*(pt2>=PT_cut)
feature = feature[weights>0.01]
target  = target[weights>0.01]
weights = weights[weights>0.01]
print('weights shape')
print(weights.shape)

#Apply efficiency modifications #################                                                                                                                        \
                                                                                                                                                                          
weights = weights *(target+(N_D*1./N_R)*(1-target))
weights[target==1] = Apply_Efficiency_Correction_global(weights[target==1],  muon_efficiency=efficiency_barrel)
print(weights.shape)
#remove mass from the inputs ####################                                                                                                                         
target    = np.expand_dims(target, axis=1)
weights   = np.expand_dims(weights, axis=1)
feature   = feature[:, :-1]
target    = np.concatenate((target,weights), axis=1 )

#standardize dataset ############################
mean_list = []
std_list  = []
for j in range(feature.shape[1]):
    vec  = feature[:, j]
    mean = np.mean(vec)
    std  = np.std(vec)
    mean_list.append(mean)
    std_list.append(std)
    if np.min(vec) < 0:
        vec = vec- mean
        vec = vec *1./ std
    elif np.max(vec) > 1.0:# Assume data is exponential -- just set mean to 1.                                                                                            
        vec = vec *1./ mean
    feature[:, j] = vec

#### training ###################################                                                                                                                        
batch_size = feature.shape[0]
print(batch_size)
bins, sb, se, eb, ee, q_b, q_e, m_sb, m_se, m_eb, m_ee, c_sb, c_se, c_eb, c_ee = ReadFit_PTbins_from_h5('/eos/user/g/ggrosso/PhD/BSM/Sistematiche/MuMu/SM/PTbinsFits_phot\
onlike_v3.h5')
ref_idx      = int(np.around(sb.shape[0]/2))-1
sb0          = np.random.normal(loc=scale_barrel, scale=sigma_sb, size=1)[0]
eb0          = np.random.normal(loc=efficiency_barrel, scale=sigma_eb, size=1)[0]
Mmatrix      = np.stack([m_sb, m_eb, m_se, m_ee], axis=0)
Qmatrix      = np.stack([q_b, q_e], axis=0)
NUmatrix     = np.array([[sb[ref_idx+1]], [eb[ref_idx+1]] ])
NURmatrix    = np.array([[sb[ref_idx]],   [eb[ref_idx]] ])
NU0matrix    = np.array([[sb0, eb0 ]])
SIGMAmatrix  = np.array([[sigma_sb, sigma_eb ]])
EB_ratio     = np.array([[endcaps_barrel_scale_r, endcaps_barrel_efficiency_r]])
mean_pts     = np.array([ mean_list[0], mean_list[1] ])
batch_size   = feature.shape[0]
inputsize    = feature.shape[1]
input_shape  = (None, inputsize)
model        = BSMfinderUpgrade(input_shape=input_shape,
                                edgebinlist=bins, endcaps_barrel_r=EB_ratio, means=mean_pts,
                                A1matrix=Mmatrix, A0matrix=Qmatrix,
                                NUmatrix=NUmatrix, NURmatrix=NURmatrix, NU0matrix=NU0matrix, SIGMAmatrix=SIGMAmatrix,
                                architecture=layers, weight_clipping=wc)

model.compile(loss=NPLLoss_New,  optimizer='adam')
print(model.summary())
hist        = model.fit(feature, target, batch_size=batch_size, epochs=total_epochs, verbose=False)
print('End training ')

# metrics #######################################                                                                                                                         
loss  = np.array(hist.history['loss'])
scale = np.array(hist.history['scale_barrel'])
norm  = np.array(hist.history['efficiency_barrel'])
laux  = np.array(hist.history['Laux'])
# test statistic ################################                                                                                                                         
final_loss = loss[-1]
t_OBS      = -2*final_loss

# save t ########################################                                                                                                                         
log_t = OUTPUT_PATH+OUTPUT_FILE_ID+'_t.txt'
out   = open(log_t,'w')
out.write("%f\n" %(t_OBS))
out.close()

# save history ########################                                                                                                                                   
log_history = OUTPUT_PATH+OUTPUT_FILE_ID+'_history'+str(patience)+'.h5'
f           = h5py.File(log_history,"w")
epoch       = np.array(range(total_epochs))
keepEpoch   = epoch % patience == 0
f.create_dataset('loss',              data=loss[keepEpoch],    compression='gzip')
f.create_dataset('Laux',              data=laux[keepEpoch],    compression='gzip')
f.create_dataset('scale_barrel',      data=scale[keepEpoch],   compression='gzip')
f.create_dataset('efficiency_barrel', data=norm[keepEpoch],    compression='gzip')
f.create_dataset('epoch',             data=epoch[keepEpoch],   compression='gzip')
f.close()

# save the model ################################                                                                                                                         
log_weights = OUTPUT_PATH+OUTPUT_FILE_ID+'_weights.h5'
model.save_weights(log_weights)

print('----------------------------\n')


