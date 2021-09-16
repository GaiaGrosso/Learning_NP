import numpy as np

trim_list = [#'trim_DYJetsToLL_M50'        ,                                                                                                                  
             'trim_WJetsToLNu'            ,
             'trim_WW'                    ,
             'trim_WZTo2L2Q'             , 'trim_WZTo3LNu'             ,
             'trim_ZZTo2L2Q'              , 'trim_ZZTo2L2Nu'             , 'trim_ZZTo4L'               ,
             'trim_ZToMuMu_M_50_120'      , 'trim_ZToMuMu_M_120_200'     , 'trim_ZToMuMu_M_200_400'    , 'trim_ZToMuMu_M_400_800'     ,
             'trim_ZToMuMu_M_800_1400'    , 'trim_ZToMuMu_M_1400_2300'   , 'trim_ZToMuMu_M_2300_3500'  , 'trim_ZToMuMu_M_3500_4500'   ,
             'trim_ZToMuMu_M_4500_6000'   , 'trim_ZToMuMu_M_6000_Inf'    ,
             'trim_TTTo2L2Nu_1star'       , 'trim_TTToSemiLeptonic_1star',
             'trim_ST_tW_top_5f'          , 'trim_ST_tW_antitop_5f'      , 'trim_ST_schannel_4f',
             'trim_ST_tchannel_top_5f'    , 'trim_ST_tchannel_antitop_5f',
            ]
trim_list_data = ['trim_Run2018A_SM_DM', 'trim_Run2018B_SM_DM', 'trim_Run2018C_SM_DM', 'trim_Run2018D_SM_DM']
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
                 'trim_ZToMuMu_M_120_200'     : 20.553,
                 'trim_ZToMuMu_M_200_400'     : 2.886,
                 'trim_ZToMuMu_M_400_800'     : 0.2517,
                 'trim_ZToMuMu_M_800_1400'    : 0.01707,
                 'trim_ZToMuMu_M_1400_2300'   : 0.001366,
                 'trim_ZToMuMu_M_2300_3500'   : 0.00008178,
                 'trim_ZToMuMu_M_3500_4500'   : 0.000003191,
                 'trim_ZToMuMu_M_4500_6000'   : 0.0000002787,
                 'trim_ZToMuMu_M_6000_Inf'    : 0.000000009569
               } # pb  
SM_expected = {'trim_DYJetsToLL_M50'        : 668421,
               'trim_WJetsToLNu'            : 975,
               'trim_WW'                    : 9810.683981,
               'trim_WZTo2L2Q'              : 822.942998,
               'trim_WZTo3LNu'              : 668.865502,
               'trim_ZZTo2L2Q'              : 483.348504,
               'trim_ZZTo2L2Nu'             : 141.393364,
               'trim_ZZTo4L'                : 133.077105,
               'trim_TTTo2L2Nu_1star'       : 52789.033768,
               'trim_TTToSemiLeptonic_1star': 290.564446,
               'trim_ST_tW_top_5f'          : 3995.111774,
               'trim_ST_tW_antitop_5f'      : 3988.280298,
               'trim_ST_tchannel_top_5f'    : 22.308081,
               'trim_ST_tchannel_antitop_5f': 12.041985,
               'trim_ST_schannel_4f'        : 1.733015,
                'trim_ZToMuMu_M_50_120'      : 240424.149297,
                 'trim_ZToMuMu_M_120_200'     : 373168.169394,
                 'trim_ZToMuMu_M_200_400'     : 63564.677511,
                 'trim_ZToMuMu_M_400_800'     : 6956.995938,
                 'trim_ZToMuMu_M_800_1400'    : 593.336840,
                 'trim_ZToMuMu_M_1400_2300'   : 54.676282,
                 'trim_ZToMuMu_M_2300_3500'   : 3.468106,
                 'trim_ZToMuMu_M_3500_4500'   : 0.138162,
                 'trim_ZToMuMu_M_4500_6000'   : 0.011960,
                 'trim_ZToMuMu_M_6000_Inf'    : 0.000408,
           }

SM_simulations = {'trim_DYJetsToLL_M50'     : 185953,
               'trim_WJetsToLNu'            : 17,
               'trim_WW'                    : 9277,
               'trim_WZTo2L2Q'              : 64773,
               'trim_WZTo3LNu'              : 29709,
               'trim_ZZTo2L2Q'              : 64508,
               'trim_ZZTo2L2Nu'             : 203328,
               'trim_ZZTo4L'                : 11017,
               'trim_TTTo2L2Nu_1star'       : 201752,
               'trim_TTToSemiLeptonic_1star': 805,
               'trim_ST_tW_top_5f'          : 27074,
               'trim_ST_tW_antitop_5f'      : 20478,
               'trim_ST_tchannel_top_5f'    : 18,
               'trim_ST_tchannel_antitop_5f': 10,
               'trim_ST_schannel_4f'        : 195,
                'trim_ZToMuMu_M_50_120'      : 5931,
                 'trim_ZToMuMu_M_120_200'     : 31266,
                 'trim_ZToMuMu_M_200_400'     : 37107,
                 'trim_ZToMuMu_M_400_800'     : 46690,
                 'trim_ZToMuMu_M_800_1400'    : 58625,
                 'trim_ZToMuMu_M_1400_2300'   : 67456,
                 'trim_ZToMuMu_M_2300_3500'   : 71588,
                 'trim_ZToMuMu_M_3500_4500'   : 73085,
                 'trim_ZToMuMu_M_4500_6000'   : 72338,
                 'trim_ZToMuMu_M_6000_Inf'    : 72063,
           }
simulations_frac = {}
SM_simulations_TOT = 0
for process in trim_list:
    SM_simulations_TOT += SM_simulations[process]

for process in trim_list:
    simulations_frac[process] = SM_simulations[process]*1./SM_simulations_TOT

expected_frac = {}
SM_expected_TOT = 0
for process in trim_list:
    SM_expected_TOT += SM_expected[process]

for process in trim_list:
    expected_frac[process] = SM_expected[process]*1./SM_expected_TOT

##################################################################################                                                                                                                            
# nuisance central values                                              
csec_nuisances_data = {'trim_DYJetsToLL_M50'        : 0,
                       'trim_WJetsToLNu'            : 0,
		       'trim_WW'                    : 0,
                       'trim_WZTo2L2Q'              : 0,
                       'trim_WZTo3LNu'              : 0,
                       'trim_ZZTo2L2Q'              : 0,
                       'trim_ZZTo2L2Nu'             : 0,
                       'trim_ZZTo4L'                : 0,
                       'trim_TTTo2L2Nu_1star'       : 0,
                       'trim_TTToSemiLeptonic_1star': 0,
                       'trim_ST_tW_top_5f'          : 0,
                       'trim_ST_tW_antitop_5f'      : 0,
                       'trim_ST_tchannel_top_5f'    : 0,
                       'trim_ST_tchannel_antitop_5f': 0,
                       'trim_ST_schannel_4f'        : 0,
                       'trim_ZToMuMu_M_50_120'      : 0,
                       'trim_ZToMuMu_M_120_200'     : 0,
                       'trim_ZToMuMu_M_200_400'     : 0,
                       'trim_ZToMuMu_M_400_800'     : 0,
                       'trim_ZToMuMu_M_800_1400'    : 0,
                       'trim_ZToMuMu_M_1400_2300'   : 0,
                       'trim_ZToMuMu_M_2300_3500'   : 0,
                       'trim_ZToMuMu_M_3500_4500'   : 0,
                       'trim_ZToMuMu_M_4500_6000'   : 0,
                       'trim_ZToMuMu_M_6000_Inf'    : 0,
                   }
csec_nuisances_reference = {'trim_DYJetsToLL_M50'        : 0,
                            'trim_WJetsToLNu'            : 0,
                            'trim_WW'                    : 0,
                            'trim_WZTo2L2Q'              : 0,
                            'trim_WZTo3LNu'              : 0,
                            'trim_ZZTo2L2Q'              : 0,
                            'trim_ZZTo2L2Nu'             : 0,
                            'trim_ZZTo4L'                : 0,
                            'trim_TTTo2L2Nu_1star'       : 0,
                            'trim_TTToSemiLeptonic_1star': 0,
                            'trim_ST_tW_top_5f'          : 0,
                            'trim_ST_tW_antitop_5f'      : 0,
                            'trim_ST_tchannel_top_5f'    : 0,
                            'trim_ST_tchannel_antitop_5f': 0,
                            'trim_ST_schannel_4f'        : 0,
                            'trim_ZToMuMu_M_50_120'      : 0,
                            'trim_ZToMuMu_M_120_200'     : 0,
                            'trim_ZToMuMu_M_200_400'     : 0,
                            'trim_ZToMuMu_M_400_800'     : 0,
                            'trim_ZToMuMu_M_800_1400'    : 0,
                            'trim_ZToMuMu_M_1400_2300'   : 0,
                            'trim_ZToMuMu_M_2300_3500'   : 0,
                            'trim_ZToMuMu_M_3500_4500'   : 0,
                            'trim_ZToMuMu_M_4500_6000'   : 0,
                            'trim_ZToMuMu_M_6000_Inf'    : 0,
                        }
csec_nuisances_sigma = {'trim_DYJetsToLL_M50'        : 0,
                        'trim_WJetsToLNu'            : 0,
                        'trim_WW'                    : 0,
                        'trim_WZTo2L2Q'              : 0,
                        'trim_WZTo3LNu'              : 0,
                        'trim_ZZTo2L2Q'              : 0,
                        'trim_ZZTo2L2Nu'             : 0,
                        'trim_ZZTo4L'                : 0,
                        'trim_TTTo2L2Nu_1star'       : 0,
                        'trim_TTToSemiLeptonic_1star': 0,
                        'trim_ST_tW_top_5f'          : 0,
                        'trim_ST_tW_antitop_5f'      : 0,
                        'trim_ST_tchannel_top_5f'    : 0,
                        'trim_ST_tchannel_antitop_5f': 0,
                        'trim_ST_schannel_4f'        : 0,
                        'trim_ZToMuMu_M_50_120'      : 0.05,
                        'trim_ZToMuMu_M_120_200'     : 0.05,
                        'trim_ZToMuMu_M_200_400'     : 0.05,
                        'trim_ZToMuMu_M_400_800'     : 0.05,
                        'trim_ZToMuMu_M_800_1400'    : 0.05,
                        'trim_ZToMuMu_M_1400_2300'   : 0.05,
                        'trim_ZToMuMu_M_2300_3500'   : 0.05,
                        'trim_ZToMuMu_M_3500_4500'   : 0.05,
                        'trim_ZToMuMu_M_4500_6000'   : 0.05,
                        'trim_ZToMuMu_M_6000_Inf'    : 0.05,
                    }

weights_file = './Parametric_5D_exp_Mcut500_sigma0.05_-3.0000000000000004_3.0000000000000004_patience20_epochs600_layers5_1_1_actrelu_wclip1.0/model_weights720_fullbatch.h5'
sigma = weights_file.split('sigma', 1)[1]
sigma = float(sigma.split('_', 1)[0])
scale_list=weights_file.split('sigma', 1)[1]
scale_list=scale_list.split('_patience', 1)[0]
scale_list=np.array([float(s) for s in scale_list.split('_')[1:]])*sigma
shape_std = np.std(scale_list)
activation= weights_file.split('act', 1)[1]
activation=activation.split('_', 1)[0]
wclip= weights_file.split('wclip', 1)[1]
wclip = float(wclip.split('/', 1)[0])
layers=weights_file.split('layers', 1)[1]
layers= layers.split('_act', 1)[0]
architecture = [int(l) for l in layers.split('_')]
csec_trim_ZToMuMu_parNN = { 'poly_degree'   : 1,
                            'architectures' : [architecture],
                            'wclips'       : [wclip],
                            'activation'    : activation,
                            'shape_std'     : shape_std,
                            'weights_file'  : weights_file
                        }
