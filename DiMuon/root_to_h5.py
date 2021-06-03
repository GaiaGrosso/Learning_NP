import ROOT
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
import pandas as pd

def DeltaPhi(phi1, phi2):
    result  = phi1 - phi2;
    result -= 2*math.pi*(result >  2*math.pi)
    result += 2*math.pi*(result <= 0*math.pi)
    return result
	
##### PARAMETERS:  
DIR_INPUT = '/eos/cms/store/group/phys_exotica/darkPhoton/schhibra/MLTechnique/RecoNTuples2018/v1/'
trim_list = ['trim_DYJetsToLL_M50_1star', 'trim_WJetsToLNu'            , 'trim_WW'            , 'trim_WZTo2L2Q'        ,
             'trim_WZTo3LNu'            , 'trim_ZZTo2L2Q'              , 'trim_ZZTo2L2Nu'     , 'trim_ZZTo4L'          ,
             'trim_TTTo2L2Nu_1star'     , 'trim_TTToSemiLeptonic_1star', 'trim_ST_tW_top_5f'  , 'trim_ST_tW_antitop_5f',
             'trim_ST_tchannel_top_5f'  , 'trim_ST_tchannel_antitop_5f', 'trim_ST_schannel_4f', 'trim_Run2018A_SM_DM'  ,
            ]
columns_MC   = ['mcweight', 'puweight', 'puweight_unc', 'exweight', 'trgweight', 
               'm1dB', 'm1dz', 'm1iso', 'm1pt', 'm1eta', 'm1phi', 
               'm2dB', 'm2dz', 'm2iso', 'm2pt', 'm2eta', 'm2phi', 
               'mass', 'dimuonpt', 'nbjets'
              ]
columns_DATA = ['m1dB', 'm1dz', 'm1iso', 'm1pt', 'm1eta', 'm1phi', 
                'm2dB', 'm2dz', 'm2iso', 'm2pt', 'm2eta', 'm2phi',
                'mass', 'dimuonpt', 'nbjets'
               ]
columns_training = [
                    'm1pt', 'm2pt', 'm1eta', 'm2eta', 'delta_phi', 'mass', 'weight'
                    ]
LUMINOSITY   = 13977 # pb-1
xsec_dict    = { 'trim_DYJetsToLL_M50_1star'  : 6225., 
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
                 'trim_Run2018A_SM_DM'        : 0
               } # pb

###############
for process in trim_list:
    print('Loading '+process)
    df   = ROOT.RDataFrame('tree', DIR_INPUT+process+'.root')
    if process == trim_list[-1]:
        npdf = df.AsNumpy(columns_DATA)
    else:
        npdf = df.AsNumpy(columns_MC)
    pddf = pd.DataFrame(npdf)
    pddf_sel = pddf.loc[ (np.abs(pddf['m1eta'])<1.9) & (np.abs(pddf['m2eta'])<1.9) &
                         (pddf['m1dB'] <0.2)         & (pddf['m2dB'] <0.2)         & 
                         (pddf['m1dz'] <0.5)         & (pddf['m2dz'] <0.5)         &
                         (pddf['m1iso']<0.15)        & (pddf['m2iso']<0.15)        &
                         (pddf['mass'] <200)         & (pddf['mass'] >110)         &
                         (pddf['nbjets']==0)
                        ]
		# create delta_phi column
    pddf_sel['delta_phi'] = DeltaPhi(pddf_sel['m1phi'].to_numpy(), pddf_sel['m2phi'].to_numpy())
    # create weight column
    if process == trim_list[-1]:
        pddf_sel['weight'] = np.ones_like(pddf_sel['mass'].to_numpy())
    else:
        pddf_sel['weight'] = LUMINOSITY*xsec_dict[process]*pddf_sel['puweight'].to_numpy()*pddf_sel['exweight'].to_numpy()*pddf_sel['trgweight'].to_numpy()*pddf_sel['mcweight'].to_numpy()

    # plot mass
    histtype ='bar'
    stacked  =True
    if process == trim_list[-1]:
        histtype ='step'
        stacked  =False
    plt.hist(pddf_sel['mass'].to_numpy(), weights = pddf_sel['weight'].to_numpy(),
             range=(110, 200), bins=18, label=process, histtype=histtype, stacked=stacked)
    # save to h5 file
    f = h5py.File('./v1/'+process+'.h5', 'w')
    for p in columns_training:
        f.create_dataset(p, data=pddf_sel[p].to_numpy(), compression='gzip')
    f.close()
    
plt.xlabel('mass', fontsize=14)
plt.legend(fontsize=14, bbox_to_anchor=(1.05, 0.95))
plt.yscale('log')
plt.show()
