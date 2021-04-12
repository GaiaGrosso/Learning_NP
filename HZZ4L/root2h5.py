import numpy as np
import h5py, glob
import math
import uproot
import os

directory = './Output_hzz4l' # specify your directory path
subdirs   = [x[0] for x in os.walk(directory)]

data_labels = [
                #'DoubleEG', 
                #'MuonEG',
                #'EGamma', 
                'DoubleMuon' #for 4mu final state
]
mcsig_labels = [
                'GluGluH',
                'VBF_HToZZTo4L'
]
mcbkg_labels = [
                'GluGluToContin', 
                #'VBFToContinToZZTo4l', we ingore it!
                'ZZTo4L_13TeV_powheg',
                'DYJets'
]

# to encode the name of the process
code_keys = {
    'DoubleMuon'             : 0,
    'DYJetsToLL'             : 1,
    'GluGluToContinToZZTo4L' : 2,
    'ZZTo4L_13TeV_powheg'    : 3,
    'GluGluHToZZTo4L_M125'   : 4,
    'VBF_HToZZTo4L_M125'     : 5 
}

def Collect_from_file(fileROOT, y_label, subdir,
                      pt1, pt2, pt3, pt4, eta1, eta2, eta3, eta4, phi1, phi2, phi3, phi4,
                      Z1_pt, Z2_pt, H_pt, Z1_eta, Z2_eta, H_eta, Z1_phi, Z2_phi, H_phi, Z1_mass, Z2_mass, H_mass,
                      cosThetaStar, cosTheta1, cosTheta2, PHI, PHI1, labels, weights, key_process):
    '''
    label: 0 if MC bkg
           1 if data
           2 if MC sig (not needed for searches, maybe used to set limits)
    key_process : string ID for the process (in subdir)
    pt1,2,3,4  : transverse momenta of leptons
    eta1,2,3,4 : pseudorapidities of leptons
    phi1,2,3,4 : azimuthal angles of leptons
    Z1_[pt, eta, phi, mass]: Z1 quadrimomentum
    Z2_[pt, eta, phi, mass]: Z2 quadrimomentum
    H_[pt, eta, phi, mass]:  H  quadrimomentum
    cosThetaStar, cosTheta1, cosTheta2, PHI, PHI1 : for theta and phi definition see AN
    '''
    fileup   = uproot.open(fileROOT)
    tree     = fileup['Events']
    branches = tree.arrays(namedecode='utf-8')
    if branches['iLepton1'].shape[0]==0:
        # empty file
        print('empty file')
        return
    
    for i in range(len(branches['iLepton1'])):
       
        # ZZ -> 4mu
        if branches['is4m'][i]:
            pt1.append(branches['Muon_pt'][i][branches['iLepton1'][i]])
            pt2.append(branches['Muon_pt'][i][branches['iLepton2'][i]])
            pt3.append(branches['Muon_pt'][i][branches['iLepton3'][i]])
            pt4.append(branches['Muon_pt'][i][branches['iLepton4'][i]])
            eta1.append(branches['Muon_eta'][i][branches['iLepton1'][i]])
            eta2.append(branches['Muon_eta'][i][branches['iLepton2'][i]])
            eta3.append(branches['Muon_eta'][i][branches['iLepton3'][i]])
            eta4.append(branches['Muon_eta'][i][branches['iLepton4'][i]])
            phi1.append(branches['Muon_phi'][i][branches['iLepton1'][i]])
            phi2.append(branches['Muon_phi'][i][branches['iLepton2'][i]])
            phi3.append(branches['Muon_phi'][i][branches['iLepton3'][i]])
            phi4.append(branches['Muon_phi'][i][branches['iLepton4'][i]])
        # ZZ -> 4e
        elif branches['is4e'][i]:
            continue
            '''
            pt1.append(branches['Electron_pt'][i][branches['iLepton1'][i]])
            pt2.append(branches['Electron_pt'][i][branches['iLepton2'][i]])
            pt3.append(branches['Electron_pt'][i][branches['iLepton3'][i]])
            pt4.append(branches['Electron_pt'][i][branches['iLepton4'][i]])
            eta1.append(branches['Electron_eta'][i][branches['iLepton1'][i]])
            eta2.append(branches['Electron_eta'][i][branches['iLepton2'][i]])
            eta3.append(branches['Electron_eta'][i][branches['iLepton3'][i]])
            eta4.append(branches['Electron_eta'][i][branches['iLepton4'][i]])
            phi1.append(branches['Electron_phi'][i][branches['iLepton1'][i]])
            phi2.append(branches['Electron_phi'][i][branches['iLepton2'][i]])
            phi3.append(branches['Electron_phi'][i][branches['iLepton3'][i]])
            phi4.append(branches['Electron_phi'][i][branches['iLepton4'][i]])
            '''
        # ZZ -> 2e2m
        elif branches['is2e2m'][i]:
            continue
            '''
            if branches['Muon_eta'][i][branches['iLepton1'][i]]-branches['Muon_eta'][i][branches['iLepton2'][i]] == branches['Z1_dEta'][i]:
                # Z1 from muons
                pt1.append(branches['Muon_pt'][i][branches['iLepton1'][i]])
                pt2.append(branches['Muon_pt'][i][branches['iLepton2'][i]])
                pt3.append(branches['Electron_pt'][i][branches['iLepton3'][i]])
                pt4.append(branches['Electron_pt'][i][branches['iLepton4'][i]])
                eta1.append(branches['Muon_eta'][i][branches['iLepton1'][i]])
                eta2.append(branches['Muon_eta'][i][branches['iLepton2'][i]])
                eta3.append(branches['Electron_eta'][i][branches['iLepton3'][i]])
                eta4.append(branches['Electron_eta'][i][branches['iLepton4'][i]])
                phi1.append(branches['Muon_phi'][i][branches['iLepton1'][i]])
                phi2.append(branches['Muon_phi'][i][branches['iLepton2'][i]])
                phi3.append(branches['Electron_phi'][i][branches['iLepton3'][i]])
                phi4.append(branches['Electron_phi'][i][branches['iLepton4'][i]])
            else:
                # Z1 from electrons
                pt1.append(branches['Electron_pt'][i][branches['iLepton1'][i]])
                pt2.append(branches['Electron_pt'][i][branches['iLepton2'][i]])
                pt3.append(branches['Muon_pt'][i][branches['iLepton3'][i]])
                pt4.append(branches['Muon_pt'][i][branches['iLepton4'][i]])
                eta1.append(branches['Electron_eta'][i][branches['iLepton1'][i]])
                eta2.append(branches['Electron_eta'][i][branches['iLepton2'][i]])
                eta3.append(branches['Muon_eta'][i][branches['iLepton3'][i]])
                eta4.append(branches['Muon_eta'][i][branches['iLepton4'][i]])
                phi1.append(branches['Electron_phi'][i][branches['iLepton1'][i]])
                phi2.append(branches['Electron_phi'][i][branches['iLepton2'][i]])
                phi3.append(branches['Muon_phi'][i][branches['iLepton3'][i]])
                phi4.append(branches['Muon_phi'][i][branches['iLepton4'][i]])
            '''
        key_process.append(subdir.split('/')[-1])
        labels.append(label)
        weights.append(branches['eventWeightLumi'][i])
        Z1_pt.append(branches['Z1_pt'][i])
        Z2_pt.append(branches['Z2_pt'][i])
        H_pt.append(branches['H_pt'][i])
        Z1_eta.append(branches['Z1_eta'][i])
        Z2_eta.append(branches['Z2_eta'][i])
        H_eta.append(branches['H_eta'][i])
        Z1_phi.append(branches['Z1_phi'][i])
        Z2_phi.append(branches['Z2_phi'][i])
        H_phi.append(branches['H_phi'][i])
        Z1_mass.append(branches['Z1_mass'][i])
        Z2_mass.append(branches['Z2_mass'][i])
        H_mass.append(branches['H_mass'][i])
        cosThetaStar.append(branches['cosThetaStar'][i])
        cosTheta1.append(branches['cosTheta1'][i])
        cosTheta2.append(branches['cosTheta2'][i])
        PHI.append(branches['phi'][i])
        PHI1.append(branches['phi1'][i])
    print('%s loaded. Length: %i'%(fileROOT.split('/')[-1], len(pt1)))
    return

if __name__ == '__main__':
    pt1  = []
    pt2  = []
    pt3  = []
    pt4  = []

    eta1 = []
    eta2 = []
    eta3 = []
    eta4 = []

    phi1 = []
    phi2 = []
    phi3 = []
    phi4 = []

    Z1_mass = []
    Z2_mass = []
    H_mass  = []

    Z1_pt = []
    Z2_pt = []
    H_pt  = []

    Z1_eta = []
    Z2_eta = []
    H_eta  = []

    Z1_phi = []
    Z2_phi = []
    H_phi  = []

    cosThetaStar = []
    cosTheta1    = []
    cosTheta2    = []
    PHI          = []
    PHI1         = []

    labels       = []
    weights      = []
    keys         = []
    for subdir in subdirs:
        label = 0
        if any(ext in subdir for ext in mcbkg_labels):
            label = 0
        elif any(ext in subdir for ext in data_labels):
            label = 1    
        elif any(ext in subdir for ext in mcsig_labels):
            label = 2
        else: continue
        print(subdir)
        for fileROOT in glob.glob('%s/*.root'%(subdir)):
            # skip bad files
            if 'DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8_RunIIFall17/5C8B9676-1B82-E911-9C8A-1418774121A1_Skim.root' in fileROOT: continue
            Collect_from_file(fileROOT, label, subdir,
                          pt1, pt2, pt3, pt4, eta1, eta2, eta3, eta4, phi1, phi2, phi3, phi4,
                          Z1_pt, Z2_pt, H_pt, Z1_eta, Z2_eta, H_eta, Z1_phi, Z2_phi, H_phi, Z1_mass, Z2_mass, H_mass,
                          cosThetaStar, cosTheta1, cosTheta2, PHI, PHI1, labels, weights, keys)
    keys_int = []
    for k in keys:
        for c in list(code_keys.keys()):
            if c in k: 
                keys_int.append(code_keys[c])
                continue
                
    pt1  = np.array(pt1)
    pt2  = np.array(pt2)
    pt3  = np.array(pt3)
    pt4  = np.array(pt4)

    eta1 = np.array(eta1)
    eta2 = np.array(eta2)
    eta3 = np.array(eta3)
    eta4 = np.array(eta4)

    phi1 = np.array(phi1)
    phi2 = np.array(phi2)
    phi3 = np.array(phi3)
    phi4 = np.array(phi4)

    Z1_mass = np.array(Z1_mass)
    Z2_mass = np.array(Z2_mass)
    H_mass  = np.array(H_mass)

    Z1_pt = np.array(Z1_pt)
    Z2_pt = np.array(Z2_pt)
    H_pt  = np.array(H_pt)

    Z1_eta = np.array(Z1_eta)
    Z2_eta = np.array(Z2_eta)
    H_eta  = np.array(H_eta)

    Z1_phi = np.array(Z1_phi)
    Z2_phi = np.array(Z2_phi)
    H_phi  = np.array(H_phi)

    cosThetaStar = np.array(cosThetaStar)
    cosTheta1    = np.array(cosTheta1)
    cosTheta2    = np.array(cosTheta2)
    PHI          = np.array(PHI)
    PHI1         = np.array(PHI1)

    labels   = np.array(labels)
    weights  = np.array(weights)
    keys     = np.array(keys)
    keys_int = np.array(keys_int) 
    
    # save on h5 files
    f  = h5py.File('./data.h5', 'w')
    f.create_dataset('pt1', data=pt1, compression='gzip')
    f.create_dataset('pt2', data=pt2, compression='gzip')
    f.create_dataset('pt3', data=pt3, compression='gzip')
    f.create_dataset('pt4', data=pt4, compression='gzip')

    f.create_dataset('eta1', data=eta1, compression='gzip')
    f.create_dataset('eta2', data=eta2, compression='gzip')
    f.create_dataset('eta3', data=eta3, compression='gzip')
    f.create_dataset('eta4', data=eta4, compression='gzip')

    f.create_dataset('phi1', data=phi1, compression='gzip')
    f.create_dataset('phi2', data=phi2, compression='gzip')
    f.create_dataset('phi3', data=phi3, compression='gzip')
    f.create_dataset('phi4', data=phi4, compression='gzip')

    f.create_dataset('Z1_pt', data=Z1_pt, compression='gzip')
    f.create_dataset('Z2_pt', data=Z2_pt, compression='gzip')
    f.create_dataset('Z1_eta', data=Z1_eta, compression='gzip')
    f.create_dataset('Z2_eta', data=Z2_eta, compression='gzip')
    f.create_dataset('Z1_phi', data=Z1_phi, compression='gzip')
    f.create_dataset('Z2_phi', data=Z2_phi, compression='gzip')
    f.create_dataset('Z1_mass', data=Z1_mass, compression='gzip')
    f.create_dataset('Z2_mass', data=Z2_mass, compression='gzip')

    f.create_dataset('H_pt', data=H_pt, compression='gzip')
    f.create_dataset('H_eta', data=H_eta, compression='gzip')
    f.create_dataset('H_phi', data=H_phi, compression='gzip')

    f.create_dataset('cosThetaStar', data=cosThetaStar, compression='gzip')
    f.create_dataset('cosTheta1', data=cosTheta1, compression='gzip')
    f.create_dataset('cosTheta2', data=cosTheta2, compression='gzip')
    f.create_dataset('PHI', data=PHI, compression='gzip')
    f.create_dataset('PHI1', data=PHI1, compression='gzip')

    f.create_dataset('labels', data=labels, compression='gzip')
    f.create_dataset('weights', data=weights, compression='gzip')
    f.create_dataset('process', data=keys_int, compression='gzip')

    f.close()
