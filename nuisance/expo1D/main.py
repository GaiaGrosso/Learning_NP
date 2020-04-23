import glob
import numpy as np
import os
import h5py
import time
import datetime
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, chi2, uniform, poisson, ncx2

DIR_INPUT = '/eos/user/g/ggrosso/BSM_outputs/ToyModel1Dfast/Pois/Toy1Dfast_scaleREF1.0_scaleDATA1.03_patience10000_ref50000_bkg5000_epochs1000000_model4_wclip0.08/'

scale_DATA = DIR_INPUT.split("scaleDATA",1)[1] 
scale_DATA = float(scale_DATA.split("_",1)[0])
scale_REF  = DIR_INPUT.split("scaleREF",1)[1] 
scale_REF  = float(scale_REF.split("_",1)[0])
print('scale DATA: %s'%(str(scale_DATA))
print('scale REF:  %s'%(scale_REF))

N_ref  = DIR_INPUT.split("ref",1)[1] 
N_ref  = int(N_ref.split("_",1)[0])
N_Data = DIR_INPUT.split("bkg",1)[1] 
N_Data = int(N_Data.split("_",1)[0])
print('N(REF): %i, N(DATA): %i'%(N_ref, N_Data))

sigma      = 0.1
scale_star = scale_DATA

# Compute Corrections
t_corrected = np.array([])
nubest      = np.array([])
nu0         = np.array([])
mean        = np.array([])
t           = np.array([])
Delta       = np.array([])
Delta1      = np.array([])
Delta2      = np.array([])
for fileIN in glob.glob("%s/*.txt" %DIR_INPUT):
    t_corrected, t, Delta1, Delta2, Delta, mean, nu0, nubest = Compute_correction_Expo1D_ExpoNu(fileIN, N_ref, N_Data,
                                                    scale_REF, scale_DATA, scale_star, sigma,
                                                    t_corrected, t, Delta1, Delta2, Delta, mean, nu0, nubest, 
                                                    Pois=True, verbose=True)
													
# Compute p-values and plotting
pv1, pv2 = Delta_hist(Delta1, Delta2, scale_REF, scale_star, sigma, 
                      df1=1, df2=1, bins1=10, bins2=10, 
                      plot=True, verbose=True)
pv3, pv4 = t_hist(t, t_corrected, Delta, Delta1, Delta2, nubest, 
                  scale_REF, scale_star, sigma, df=13, n=7,
                  bins1=10, bins2=7, bins3=7, bins4=7, 
                  xmin2=np.min(t_corrected), xmax2=np.max(t), 
                  xmin3=np.min(t-Delta1), xmax3=np.max(t), 
                  xmin4=np.min(t), xmax4=np.max(t-Delta2),
                  plot=True, verbose=True)
nu_hist(nubest, mean, scale_REF, scale_star, sigma)
correlation_plots(t, Delta, Delta1, Delta2, nubest, nu0, scale_REF, scale_star, sigma)
