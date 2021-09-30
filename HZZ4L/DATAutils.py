import numpy as np
bkg_list = ['AllData_ZX_redTree_2018',
            'ggTo2e2tau_Contin_MCFM701_redTree_2018',
            'ggTo4mu_Contin_MCFM701_redTree_2018',
            'ggTo2mu2tau_Contin_MCFM701_redTree_2018',
            'ggTo4tau_Contin_MCFM701_redTree_2018',
            'ggTo2e2mu_Contin_MCFM701_redTree_2018',
            'ggTo4e_Contin_MCFM701_redTree_2018',
            'ZZTo4lext_redTree_2018_0',
            'ZZTo4lext_redTree_2018_1',
            'ZZTo4lext_redTree_2018_2',
            'ZZTo4lext_redTree_2018_3',
        ]

csec_nuisances_sigma = {
    'AllData_ZX_redTree_2018': 0,
    'ggTo2e2tau_Contin_MCFM701_redTree_2018': 0,
    'ggTo4mu_Contin_MCFM701_redTree_2018': 0,
    'ggTo2mu2tau_Contin_MCFM701_redTree_2018': 0,
    'ggTo4tau_Contin_MCFM701_redTree_2018': 0,
    'ggTo2e2mu_Contin_MCFM701_redTree_2018': 0,
    'ggTo4e_Contin_MCFM701_redTree_2018': 0,
    'ZZTo4lext_redTree_2018_0': 0,
    'ZZTo4lext_redTree_2018_1': 0,
    'ZZTo4lext_redTree_2018_2': 0,
    'ZZTo4lext_redTree_2018_3': 0,
}

csec_nuisances_reference = {
    'AllData_ZX_redTree_2018': 0,
    'ggTo2e2tau_Contin_MCFM701_redTree_2018': 0,
    'ggTo4mu_Contin_MCFM701_redTree_2018': 0,
    'ggTo2mu2tau_Contin_MCFM701_redTree_2018': 0,
    'ggTo4tau_Contin_MCFM701_redTree_2018': 0,
    'ggTo2e2mu_Contin_MCFM701_redTree_2018': 0,
    'ggTo4e_Contin_MCFM701_redTree_2018': 0,
    'ZZTo4lext_redTree_2018_0': 0,
    'ZZTo4lext_redTree_2018_1': 0,
    'ZZTo4lext_redTree_2018_2': 0,
    'ZZTo4lext_redTree_2018_3': 0,
}

csec_nuisances_data = {
    'AllData_ZX_redTree_2018': 0,
    'ggTo2e2tau_Contin_MCFM701_redTree_2018': 0,
    'ggTo4mu_Contin_MCFM701_redTree_2018': 0,
    'ggTo2mu2tau_Contin_MCFM701_redTree_2018': 0,
    'ggTo4tau_Contin_MCFM701_redTree_2018': 0,
    'ggTo2e2mu_Contin_MCFM701_redTree_2018': 0,
    'ggTo4e_Contin_MCFM701_redTree_2018': 0,
    'ZZTo4lext_redTree_2018_0': 0,
    'ZZTo4lext_redTree_2018_1': 0,
    'ZZTo4lext_redTree_2018_2': 0,
    'ZZTo4lext_redTree_2018_3': 0,
}


#########  

weights_file = '..../Parametric_5D_exp_Mcut500_sigma0.05_-3.0000000000000004_3.0000000000000004_patience20_epochs600_layers5_1_1_actrelu_wclip1.0/model_weights720_fullbatch.h5'
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

# Plot tools: ###############                                                                                                                                                       
color_code = {
    'AllData_ZX_redTree_2018':                 '#ffffcc',
    'ggTo2e2tau_Contin_MCFM701_redTree_2018':  '#ffeda0',
    'ggTo4mu_Contin_MCFM701_redTree_2018':     '#fed976',
    'ggTo2mu2tau_Contin_MCFM701_redTree_2018': '#feb24c',
    'ggTo4tau_Contin_MCFM701_redTree_2018':    '#fd8d3c',
    'ggTo2e2mu_Contin_MCFM701_redTree_2018':   '#fc4e2a',
    'ggTo4e_Contin_MCFM701_redTree_2018':      '#e31a1c',
    'ZZTo4lext_redTree_2018':                  '#b10026',
}

label_code = {
    'AllData_ZX_redTree_2018':                 r'$Z+X$',
    'ggTo2e2tau_Contin_MCFM701_redTree_2018':  r'$gg \rightarrow 2e2\tau$',
    'ggTo4mu_Contin_MCFM701_redTree_2018':     r'$gg \rightarrow 4\mu$',
    'ggTo2mu2tau_Contin_MCFM701_redTree_2018': r'$gg \rightarrow 2\mu2\tau$',
    'ggTo4tau_Contin_MCFM701_redTree_2018':    r'$gg \rightarrow 4\tau$',
    'ggTo2e2mu_Contin_MCFM701_redTree_2018':   r'$gg \rightarrow 2e2\mu$',
    'ggTo4e_Contin_MCFM701_redTree_2018':      r'$gg \rightarrow 4e$',
    'ZZTo4lext_redTree_2018':                  r'$qq \rightarrow 4l$',
}

bins_code = {
        'ZZMass': np.arange(60, 500, 5), 
        'ZZPt': np.arange(0, 500, 5), 
        'ZZEta': np.arange(-2.5, 2.5, 0.1), 
        'ZZPhi': np.arange(-3.2, 3.2, 0.1),
    
        'Z1Mass': np.arange(0, 200, 2), 
        'Z1Pt': np.arange(0, 500, 5), 
        'Z1Eta': np.arange(-2.5, 2.5, 0.1), 
        'Z1Phi': np.arange(-3.2, 3.2, 0.1),
        
        'Z2Mass': np.arange(0, 200, 2), 
        'Z2Pt': np.arange(0, 500, 5), 
        'Z2Eta': np.arange(-2.5, 2.5, 0.1), 
        'Z2Phi': np.arange(-3.2, 3.2, 0.1),
    
        'Z1Z2DeltaPhi' : np.arange(-3.2, 3.2, 0.1),
    
        'l1Pt': np.arange(0, 500, 5),
        'l1Eta': np.arange(-2.5, 2.5, 0.1),
        'l1Phi': np.arange(-3.2, 3.2, 0.1),
        'l2Pt': np.arange(0, 500, 5),
        'l2Eta': np.arange(-2.5, 2.5, 0.1),
        'l2Phi': np.arange(-3.2, 3.2, 0.1),
        'l3Pt': np.arange(0, 500, 5),
        'l3Eta': np.arange(-2.5, 2.5, 0.1),
        'l3Phi': np.arange(-3.2, 3.2, 0.1),
        'l4Pt': np.arange(0, 500, 5),
        'l4Eta': np.arange(-2.5, 2.5, 0.1),
        'l4Phi': np.arange(-3.2, 3.2, 0.1),
}


xlabel_code = {
        'ZZMass': r'$m_{\rm{ZZ}}$', 
        'ZZPt': r'$PT_{\rm{ZZ}}$', 
        'ZZEta': r'$\eta_{\rm{ZZ}}$', 
        'ZZPhi': r'$\phi_{\rm{ZZ}}$',
    
        'Z1Mass': r'$m_{\rm{Z1}}$', 
        'Z1Pt': r'$PT_{\rm{Z1}}$', 
        'Z1Eta': r'$\eta_{\rm{Z1}}$', 
        'Z1Phi': r'$\phi_{\rm{Z1}}$',
    
        'Z2Mass': r'$m_{\rm{Z2}}$', 
        'Z2Pt': r'$PT_{\rm{Z2}}$', 
        'Z2Eta': r'$\eta_{\rm{Z2}}$', 
        'Z2Phi': r'$\phi_{\rm{Z2}}$',
    
        'Z1Z2DeltaPhi' : r'$\Delta\phi_{\rm{Z1,Z2}}$',
        
        'l1Pt': r'$PT_{l1}$',
        'l1Eta': r'$\eta_{l1}$',
        'l1Phi': r'$\phi_{l1}$',
        'l2Pt': r'$PT_{l2}$',
        'l2Eta': r'$\eta_{l2}$',
        'l2Phi': r'$\phi_{l2}$',
        'l3Pt': r'$PT_{l3}$',
        'l3Eta': r'$\eta_{l3}$',
        'l3Phi': r'$\phi_{l3}$',
        'l4Pt': r'$PT_{l4}$',
        'l4Eta': r'$\eta_{l4}$',
        'l4Phi': r'$\phi_{l4}$',
}
