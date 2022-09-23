import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import math, os, h5py
import pandas as pd
import tensorflow as tf

def DeltaPhi(phi1, phi2):
    result  = phi1 - phi2;
    result -= 2*math.pi*(result >  math.pi)
    result += 2*math.pi*(result <= -math.pi)
    return result
  
  def Apply_MuonMomentumScale_Correction(data_5D, scale_4D, muon_scale=0):
    muon_mass = 0.1#GeV/c2                                                                                                                        
    m1SF = scale_4D[:, 0]
    m2SF = scale_4D[:, 2]
    m1SF_err = scale_4D[:, 1]
    m2SF_err = scale_4D[:, 3]

    pt1  = data_5D[:, 0]/m1SF*(m1SF + muon_scale*m1SF_err)
    pt2  = data_5D[:, 1]/m2SF*(m2SF + muon_scale*m2SF_err)
    eta1 = data_5D[:, 2]
    eta2 = data_5D[:, 3]
    dphi = data_5D[:, 4]

    px1= pt1
    px2= pt2*np.cos(dphi)
    py1= np.zeros_like(pt1)
    py2= pt2*np.sin(dphi)
    pz1= pt1*np.sinh(eta1)
    pz2= pt2*np.sinh(eta2)
    E1 = np.sqrt(px1*px1+py1*py1+pz1*pz1+muon_mass*muon_mass)
    E2 = np.sqrt(px2*px2+py2*py2+pz2*pz2+muon_mass*muon_mass)
    px = px1+px2
    py = py1+py2
    pz = pz1+pz2
    E  = E1+E2
    mll= np.sqrt(E*E-px*px-py*py-pz*pz)

    data_5D_new      = np.copy(data_5D)
    data_5D_new[:, 0]= pt1
    data_5D_new[:, 1]= pt2
    data_5D_new[:, 5]= mll
    return data_5D_new
  
  labels_dict  = { 'trim_DYJetsToLL_M50'        : 5,
                 'trim_WJetsToLNu'            : 0,
                 'trim_WJetsToLNu_plus_ext1'  : 0,
                 'trim_WW'                    : 1,
                 'trim_WWTo2L2Nu_PSweights'   : 1,
                 'trim_WW_plus_ext1'          : 1,
                 'trim_WZTo2L2Q'              : 1,
                 'trim_WZTo3LNu'              : 1,
                 'trim_ZZTo2L2Q'              : 1,
                 'trim_ZZTo2L2Nu'             : 1,
                 'trim_ZZTo4L'                : 1,
                 'trim_ZZTo4L_1star'          : 1,
                 'trim_TTTo2L2Nu_1star'       : 2,
                 'trim_TTTo2L2Nu'             : 2,
                 'trim_TTTo2L2Nu_PSweights'   : 2,
                 'trim_TTToSemiLeptonic_1star': 2,
                 'trim_TTToSemiLeptonic_PSweights_1star':2,
                 'trim_TTToSemilepton'        : 2,
                 'trim_ST_tW_top_5f'          : 3,
                 'trim_ST_tW_top_5f_PSweights': 3,
                 'trim_ST_tW_antitop_5f'      : 3,
                 'trim_ST_tW_antitop_5f_PSweights': 3,
                 'trim_ST_tchannel_top_5f'    : 3,
                 'trim_ST_tchannel_top_4f'    : 3,
                 'trim_ST_tchannel_antitop_5f': 3,
                 'trim_ST_tchannel_antitop_4f': 3,
                 'trim_ST_tchannel_antitop_5f_PSweights': 3,
                 'trim_ST_schannel_4f'        : 3,
                 'trim_ST_schannel_4f_PSweights' :3,
                 'trim_ZToMuMu_M_50_120'      : 4,
                 'trim_ZToMuMu_M_50_120_52files': 4,
                 'trim_ZToMuMu_M_50_120_theoryUnc_57files': 4,
                 'trim_ZToMuMu_M_120_200'     : 4,
                 'trim_ZToMuMu_M_120_200_ext1'     : 4,
                 'trim_ZToMuMu_M_120_200_theoryUnc': 4,
                 'trim_ZToMuMu_M_200_400'     : 4,
                 'trim_ZToMuMu_M_200_400_ext1'     : 4,
                 'trim_ZToMuMu_M_200_400_theoryUnc': 4,
                 'trim_ZToMuMu_M_400_800'     : 4,
                 'trim_ZToMuMu_M_400_800_ext1'     : 4,
                 'trim_ZToMuMu_M_400_800_theoryUnc': 4,
                 'trim_ZToMuMu_M_800_1400'    : 4,
                 'trim_ZToMuMu_M_800_1400_theoryUnc': 4,
                 'trim_ZToMuMu_M_1400_2300'   : 4,
                 'trim_ZToMuMu_M_1400_2300_theoryUnc': 4,
                 'trim_ZToMuMu_M_2300_3500'   : 4,
                 'trim_ZToMuMu_M_2300_3500_theoryUnc_2files': 4,
                 'trim_ZToMuMu_M_3500_4500'   : 4,
                 'trim_ZToMuMu_M_3500_4500_theoryUnc': 4,
                 'trim_ZToMuMu_M_4500_6000'   : 4,
                 'trim_ZToMuMu_M_4500_6000_theoryUnc': 4,
                 'trim_ZToMuMu_M_6000_Inf'    : 4,
                 'trim_ZToMuMu_M_6000_Inf_theoryUnc': 4
               }
ref_labels = ['W+jets', 'WW+WZ+ZZ', r'$t\bar{t}$', r'$t/\bar{t}$', 'DY', 'DY']
DIR_INPUT = '/eos/cms/store/group/phys_exotica/darkPhoton/schhibra/MLTechnique/RecoNTuples2018/v3/'
trim_list = [
             'trim_WJetsToLNu'            , 'trim_WW'                    , 'trim_WZTo2L2Q'             , 'trim_WZTo3LNu'             ,
             'trim_ZZTo2L2Q'              , 'trim_ZZTo2L2Nu'             , 'trim_ZZTo4L'               ,
             'trim_ZToMuMu_M_50_120' ,
             #'trim_ZToMuMu_M_120_200', 'trim_ZToMuMu_M_200_400', 'trim_ZToMuMu_M_400_800',                                                                
             'trim_ZToMuMu_M_120_200_ext1', 'trim_ZToMuMu_M_200_400_ext1', 'trim_ZToMuMu_M_400_800_ext1'    ,
             'trim_ZToMuMu_M_800_1400'    , 'trim_ZToMuMu_M_1400_2300'   , 'trim_ZToMuMu_M_2300_3500'  , 'trim_ZToMuMu_M_3500_4500'   ,
             'trim_ZToMuMu_M_4500_6000'   , 'trim_ZToMuMu_M_6000_Inf'    ,
             'trim_TTTo2L2Nu_1star'       , 'trim_TTToSemiLeptonic_1star',
             'trim_ST_tW_top_5f'          , 'trim_ST_tW_antitop_5f'      ,
             'trim_ST_tchannel_top_5f'    , 'trim_ST_tchannel_antitop_5f', 'trim_ST_schannel_4f',
             #'trim_DYJetsToLL_M50',                                                                                                                       
            ]
trim_list_data = ['trim_Run2018A_SM_DM', 'trim_Run2018B_SM_DM', 'trim_Run2018C_SM_DM', 'trim_Run2018D_SM_DM']
columns_MC   = ['mcweight', 'puweight', 'exweight', 'trgweight',
               'm1dB', 'm1dz', 'm1iso', 'm1pt', 'm1eta', 'm1phi', 'm1SF', 'm1SFErr',
               'm2dB', 'm2dz', 'm2iso', 'm2pt', 'm2eta', 'm2phi', 'm2SF', 'm2SFErr',
               'mass', 'dimuonpt', 'nbjets', 'nmu', 'genleadmupt'
              ]
columns_scalecorr = ['m1SF', 'm1SFErr', 'm2SF', 'm2SFErr']
columns_training  = ['leadmupt', 'subleadmupt', 'leadmueta', 'subleadmueta', 'delta_phi', 'mass']
columns_weight    = ['weight']
LUMINOSITY   = 13977.334 + 7057.8 + 6894.8 + 31742.6 # pb-1 [RunA, RunB, RunC, RunD]                                                                                                        
xsec_dict    = { 'trim_DYJetsToLL_M50'        : 6225.,
                 'trim_WJetsToLNu'            : 61526.7,
                 'trim_WW'                    : 115.,
                 'trim_WZTo2L2Q'              : 6.331,
                 'trim_WZTo3LNu'              : 5.052,
                 'trim_ZZTo2L2Q'              : 3.688,
                 'trim_ZZTo2L2Nu'             : 0.5644,
                 'trim_ZZTo4L'                : 1.325,
                 'trim_TTTo2L2Nu_1star'       : 88.29,
                 'trim_TTToSemiLeptonic_1star': 365.34,
                 'trim_ST_tW_top_5f'          : 19.2,
                 'trim_ST_tW_antitop_5f'      : 19.23,
                 'trim_ST_tchannel_top_5f'    : 119.7,
                 'trim_ST_tchannel_antitop_5f': 71.74,
                 'trim_ST_schannel_4f'        : 3.74,
                 'trim_ZToMuMu_M_50_120'      : 2112.904,
                 'trim_ZToMuMu_M_120_200_ext1': 20.553,
                 'trim_ZToMuMu_M_200_400_ext1': 2.886,
                 'trim_ZToMuMu_M_400_800_ext1': 0.2517,
                 'trim_ZToMuMu_M_800_1400'    : 0.01707,
                 'trim_ZToMuMu_M_1400_2300'   : 0.001366,
                 'trim_ZToMuMu_M_2300_3500'   : 0.00008178,
                 'trim_ZToMuMu_M_3500_4500'   : 0.000003191,
                 'trim_ZToMuMu_M_4500_6000'   : 0.0000002787,
                 'trim_ZToMuMu_M_6000_Inf'    : 0.000000009569
               } # pb                                                                                                                                                                       


mass_max = 6000
mass_min = 120
bins_dict = {
    'leadmupt': np.append(np.append(np.arange(0, 400, 20), np.arange(460, 700, 80)), [800, 900, 1200, 1900]),
    'subleadmupt': np.append(np.arange(0, 450, 50), [ 480, 600, 700, 800, 900, 1000, 1200, 1900]),
    'leadmueta': np.arange(-2, 2.1, 0.1),
    'subleadmueta': np.arange(-2, 2.1, 0.1),
    'delta_phi': np.arange(-3.5, 3.7, 0.2),
    'mass': np.append(np.arange(mass_min, 1100, 50), [1120, 1200,1300, 1450,1600, 1800, 2100, 2400, 3000])
}
xlabel_dict = {
    'leadmupt': r'$p_{\rm{T},1}$ (GeV/$c$)',
    'subleadmupt': r'$p_{\rm{T},2}$ (GeV/$c$)',
    'leadmueta': r'$\eta_1$',
    'subleadmueta': r'$\eta_2$',
    'delta_phi': r'$\Delta\phi_{1,2}$',
    'mass': r'$m_{\mu^{+}\mu^{-}}$ (GeV/$c^2$)'
}
colors = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3']

def read_data_training_nu(folder, mass_cut, nu, muonpt_scale_str, trim_list, columns_training, columns_weight):
    #muonpt_scale_str = str(nu)                                                                                                                                                             
    print(muonpt_scale_str)
    mc_folder   = '%s/MC_2018_M%i_muptscale_%s/'%(folder, mass_cut, muonpt_scale_str)
    DATA   = np.array([])
    W_DATA = np.array([])
    Y_DATA = np.array([])
    REF    = np.array([])
    W_REF  = np.array([])
    i=0
    for process in trim_list:
        if 'DY' in process: continue
        f = h5py.File(mc_folder+process+'.h5', 'r')
        read_file = np.array([])
        for p in columns_training+columns_weight:
            col = np.array(f.get(p))
            col = np.expand_dims(col, axis=1)
            if read_file.shape[0]==0:
                read_file = col
            else:
                read_file = np.concatenate((read_file, col), axis=1)                                                                                                                                                                         
        if REF.shape[0]==0:
            REF    = read_file[:, :-1]
            W_REF  = read_file[:, -1:]
        else:
            REF    = np.concatenate((REF,  read_file[:, :-1]), axis=0)
            W_REF  = np.concatenate((W_REF,  read_file[:, -1:]), axis=0)
        i+=1
    mask = (W_REF[:, 0]>0)*(W_REF[:, 0]<0.5)*(REF[:, 0]>0)*(REF[:, 1]>0)
    return REF[mask], W_REF[mask]

def random_pick(feature, target, seed, fraction=0.1):
    np.random.seed(seed)
    N = feature.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    mask = (idx<int(N*fraction))
    return tf.convert_to_tensor(feature[mask], dtype=tf.float32), tf.convert_to_tensor(target[mask], dtype=tf.float32)

h_0 = [0.00000000e+00, 1.13671080e+04, 7.81098427e+04, 1.80987164e+05,
 9.29998114e+04, 4.71592306e+04, 2.59384642e+04, 1.50995220e+04,
 9.20488084e+03, 5.81664860e+03, 3.85126239e+03, 2.59251438e+03,
 1.76447304e+03, 1.24465368e+03, 9.06274715e+02, 6.50472014e+02,
 4.89538609e+02, 3.54766734e+02, 2.76671974e+02, 5.96471748e+02,
 2.49573210e+02, 1.08081488e+02, 8.77568749e+01, 1.60219015e+01,
 1.36326346e+01, 2.58589354e+00]
h2_0 = [0.00000000e+00, 3.43299503e+03, 2.45199380e+04, 5.92034853e+04,
 2.83628930e+04, 1.09641902e+04, 5.26920550e+03, 2.92999614e+03,
 1.70671368e+03, 1.00749080e+03, 6.01292013e+02, 3.76220608e+02,
 2.37526614e+02, 1.58406345e+02, 1.12354306e+02, 7.43393254e+01,
 5.36645137e+01, 3.47319699e+01, 2.78887045e+01, 5.00946643e+01,
 1.87302609e+01, 6.75895670e+00, 4.18439987e+00, 7.37624519e-01,
 6.73698008e-01, 1.15445356e-01]

