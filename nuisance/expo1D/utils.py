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

  
 def Compute_correction_Expo1D_ExpoNu(fileIN, N_ref, N_Data, 
                                      scale_REF, scale_DATA, scale_star, sigma,
                                      t_corrected_list, nubest_list, t_list, Delta1_list, 
                                      Delta2_list, Delta_list, mean_list, nu0_list,
                                      Pois=False, verbose=False):
  
  """
  Compute_correction_Expo1D_ExpoNu reads the NN output (t= -2*Loss_final) from fileIN; 
  computes the correction terms Delta1 and Delta2 and applies them to t (t_corrected = t - (Delta1 + Delta2) ).
  Parametrization chosen for the exponential distribution affected by a scale nuisance:
  x ~ 1/exp(nu) * exp(-x/exp(nu))
  
  Inputs:
  - fileIN: file storing the NN output (.txt)
  - N_ref: size of the reference sample (int)
  - N_Data: median size of the data sample, poissonian distributed (int)
  - scale_REF: exp(nu_REF)
  - scale_DATA: exp(nu_DATA)
  - scale_star: exp(nu_star)
  - sigma: nu_DATA ~ Norm(nu_star, sigma)
  - t_list: list to which t for fileIN is appended
  - t_corrected_list: list to which t_corrected for fileIN is appended
  - Delta_list: list to which Delta for fileIN output correction is appended
  - Delta1_list: list to which Delta1 for fileIN output correction is appended
  - Delta2_list: list to which Delta2 for fileIN output correction is appended
  - nubest_list: list to which nu_best for fileIN output correction is appended
  - nu0_list: list to which nu0 genrated according to Norm(nu_star, sigma) is appended
  - mean_list: list to which the sample mean is appended
  - Pois: True if the size of the data sample has to be generated according to Poissonian(N_Data)
  - verbose: True to print the outputs
  """
    
    # read t
    f = open(fileIN)
    lines = f.readlines()
    if len(lines)==0:
        print("No t collected")
        return t_corrected_list, nubest_list, t_list, Delta1_list, Delta2_list, Delta_list, mean_list, nu0_list
    t = float(lines[0])
    t_list = np.append(t_list, t)
    f.close()
    
    # seed
    seed = fileIN.split("seed",1)[1] 
    seed = int(seed.split("_",1)[0])
    np.random.seed(seed)
    
    # Pois(N_Data)
    if Pois:
        N_Data_P = np.random.poisson(lam=N_Data, size=1)
        N_Data_p = N_Data_P[0]
    else:
        N_Data_p = N_Data
    
    # DATA
    feature_REF, feature_DATA, target_REF, target_DATA = Build_Data1D_Expo(1., N_ref, N_Data_p, seed)
    feature_REF  = Apply_nu(feature_REF, p_scale_REF)
    feature_DATA = Apply_nu(feature_DATA, p_scale_DATA)

    # Maximum
    Deltas  = np.array([])
    Deltas1 = np.array([])
    Deltas2 = np.array([])
    # list to be scanned to find the nu best fit
    scale_list = np.linspace(scale_star-0.1, p_star+0.1, 1000)
    
    # reinitialize the random seed so that the correction is not determined by the seed which generates the samples (which is fixed)
    seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
    np.random.seed(seed)
    
    # generate nu0 ~ Norm(nu_star, sigma) (value of nu which is derived by auxiliary measurements)
    nu_star = np.log(scale_star)
    nu0 = np.random.normal(loc=nu_star, scale=sigma, size=1)
    nu0 = nu0[0]
    
    # search the maximum
    for sc in scale_list:
        nu = np.log(sc)
        nu_REF = np.log(scale_REF)
        L_aux0  = norm.pdf(nu0,    loc=nu0, scale=sigma)
        L_aux1  = norm.pdf(nu,     loc=nu0, scale=sigma)
        L_aux2  = norm.pdf(nu_REF, loc=nu0, scale=sigma)
        #a1 = -1./(2*sigma**2)*((ps-nu0)*(ps-nu0)-(p_scale_REF-nu0)*(p_scale_REF-nu0))
        #a2 = 1./(2*sigma**2)*((p_scale_REF-nu0)*(p_scale_REF-nu0))
        #a3 = -1./(2*sigma**2)*((ps-nu0)*(ps-nu0))
        
        sum_log = -feature_DATA.shape[0]*np.log(sc*1./scale_REF) + (1./scale_REF-1./sc)*np.sum(feature_DATA[:, 0])
        
        d1 =  2*(sum_log + np.log(L_aux1/L_aux2))
        d2 = -2*np.log(L_aux0/L_aux2)
        d  =  2*(sum_log + np.log(L_aux1/L_aux0))
        
        Deltas  = np.append(Deltas , d)
        Deltas1 = np.append(Deltas1, d1)
        Deltas2 = np.append(Deltas2, d2)
        
    #######################################################    
    nubest      = np.log(scale_list[np.argmax(Deltas)])
    R           = Deltas[np.argmax(Deltas)]  #np.max(Deltas[:, 1])
    R1          = Deltas1[np.argmax(Deltas)]
    R2          = Deltas2[np.argmax(Deltas)]
    t_corrected = t-R
    
    t_corrected_list = np.append(t_corrected_list, t_corrected)
    nubest_list      = np.append(nubest_list, nubest)
    mean_list        = np.append(mean_list, np.mean(feature_DATA[:, 0]))
    nu0_list         = np.append(nu0_list, nu0)
    Delta_list       = np.append(Delta_list,  R )
    Delta1_list      = np.append(Delta1_list, R1)
    Delta2_list      = np.append(Delta2_list, R2)
    
    # Print results:
    if verbose:
        print('N BKG:         %i'%(N_Data_p)) 
        print('REF  pscale:   %f'%(scale_REF)) 
        print('DATA pscale:   %f'%(scale_DATA)) 
        print('DATA mean:     %f'%(np.mean(feature_DATA[:, 0])))
        print('Generated nu0: %f'%(nu0)) 
        print('Best nu:       %f'%(nubest)) 

        print('Correction TOT: %s'%(R))
        print('Correction   1: %s'%(R1))
        print('Correction   2: %s'%(R2))
        print('t             : %s'%(t))
        print('Corrected t   : %s'%(t_corrected))
    return t_corrected_list, nubest_list, t_list, Delta1_list, Delta2_list, Delta_list, mean_list, nu0_list

