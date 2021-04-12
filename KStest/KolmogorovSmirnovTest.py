import glob
import numpy as np
import os
import h5py
import time
import datetime
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, chi2, uniform, poisson, ncx2

def KSDistanceToUniform(sample):
    Ntrials      = sample.shape[0]
    #ECDF         = np.array([(i)*1./Ntrials for i in np.arange(Ntrials-1)])
    ECDF         = np.array([i*1./Ntrials for i in np.arange(Ntrials+1)])
    sortedsample = np.sort(sample)
    sortedsample = np.append(0, np.sort(sample))
    KSdist       = 0
    if Ntrials==1:
        KSdist = np.maximum(1-sortedsample[1], sortedsample[1])
    else:
        KSdist = np.max([np.maximum(np.abs(sortedsample[i+1]-ECDF[i]), np.abs(sortedsample[i+1]-ECDF[i+1])) for i in np.arange(Ntrials)])
    return KSdist

def KSTestStat(data, ndof):
    sample = chi2.cdf(data, ndof)
    KSdist = KSDistanceToUniform(sample)
    return KSdist

def GenUniformToy(Ntrials):
    sample = np.random.uniform(size=(Ntrials,))
    KSdist = KSDistanceToUniform(sample)
    return KSdist

def GetTSDistribution(Ntrials, Ntoys=1000):
    KSdistDistribution = []
    for i in range(Ntoys):
        KSdist = GenUniformToy(Ntrials)
        KSdistDistribution.append(KSdist)
    return np.array(KSdistDistribution)

def pvalue(KSTestStat_Value, KSdistDistribution):
    #print(np.sum((1*KSdistDistribution>KSTestStat_Value)))
    #print(KSdistDistribution.shape[0])
    pval_right=np.sum(1*(KSdistDistribution>KSTestStat_Value))*1./KSdistDistribution.shape[0]
    return pval_right

def GenToyFromEmpiricalPDF(sample):
    Ntrials = sample.shape[0]
    indeces = np.random.randint(low=0, high=Ntrials, size=(Ntrials,))
    toy     = np.array([sample[indeces[i]] for i in range(Ntrials)])
    return toy

def KS_test(sample, dof, Ntoys=100000):
    Ntrials            = sample.shape[0]
    KSTestStat_Value   = KSTestStat(sample, dof)
    KSdistDistribution = GetTSDistribution(Ntrials=Ntrials, Ntoys=Ntoys)
    pval               = pvalue(KSTestStat_Value, KSdistDistribution)
    return pval
