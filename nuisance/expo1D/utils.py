import glob
import numpy as np
import os
import h5py
import time
import datetime
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, chi2, uniform, poisson, ncx2


def Build_Data1D_Expo(scale, N_ref, N_Bkg, seed):
  """
  Build_Data1D_Expo generates a reference sample and a data sample distributed according to an exponential distribution:
  x ~ 1/scale * exp(-x/scale)
  Inputs:
  - scale:        parameter of the exponential distribution (float)
  - N_ref, N_Bkg: size of the reference and the data samples (int)
  - seed:         numpy random seed for the initialization (int)
  """
    np.random.seed(seed)
    feature = np.random.exponential(scale=scale, size=(N_ref+N_Bkg, 1))
    feature_REF  = feature[:N_ref, :]
    feature_DATA = feature[N_ref:, :]
    target_REF   = np.zeros(N_ref)
    target_DATA  = np.ones(N_Bkg)
    
    return feature_REF, feature_DATA, target_REF, target_DATA


def Apply_nu(DATA, nu):
  """
  Apply_nu modifies the data sample applying the effects of a nuisance parameter on the scale of the exponential distribution:
  x --> x' = nu * x
  
  Inputs:
  - DATA: data sample (numpy array shape:(N, 1))
  - nu: nuisance effect
  """
    DATA_new = nu*np.copy(DATA)
    return DATA_new
