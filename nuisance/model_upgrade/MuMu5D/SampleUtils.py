import os
import h5py
import numpy as np


def BuildSample_DY(N_Events, INPUT_PATH, seed, nfiles=20):
    np.random.seed(seed)
    #random integer to select Zprime file between n files                                                                                                                
    u = np.arange(nfiles)#np.random.randint(100, size=100)                                                                                                               
    np.random.shuffle(u)
    #BACKGROUND                                                                                                                                                          
    #extract N_Events from files                                                                                                                                         
    toy_label = INPUT_PATH.split("/")[-2]
    print(toy_label)

    HLF = np.array([])

    for u_i in u:
        f = h5py.File(INPUT_PATH+toy_label+str(u_i+1)+".h5", 'r')
        keys=list(f.keys())
        #check whether the file is empty                                                                                                                                 \
                                                                                                                                                                          
        if len(keys)==0:
            continue
        cols=np.array([])
        for i in range(len(keys)):
            feature = np.array(f.get(keys[i]))
            feature = np.expand_dims(feature, axis=1)
            if i==0:
                cols = feature
            else:
                cols = np.concatenate((cols, feature), axis=1)

        np.random.shuffle(cols) #don't want to select always the same event first                                                                                        

        if HLF.shape[0]==0:
            HLF=cols
            i=i+1
        else:
            HLF=np.concatenate((HLF, cols), axis=0)
        f.close()
        #print(HLF_REF.shape)                                                                                                                                            \
                                                                                                                                                                          
        if HLF.shape[0]>=N_Events:
            HLF=HLF[:N_Events, :]
            break
    #print('HLF shape')                                                                                                                                                  \
                                                                                                                                                                          
    print(HLF.shape)
    return HLF[:, [4, 5, 1, 2, 0, 3]]

def Apply_MuonMomentumScale_Correction_ETAregion(HLF, muon_scale=0, eta_min=0, eta_max=2.4):
    muon_mass=0.1 #GeV/c2                                                                                                                                                \
                                                                                                                                                                          
    pt1 =HLF[:, 0]+ HLF[:, 0]*muon_scale*(np.abs(HLF[:, 2])<eta_max)*(np.abs(HLF[:, 2])>=eta_min)
    pt2 =HLF[:, 1]+ HLF[:, 1]*muon_scale*(np.abs(HLF[:, 3])<eta_max)*(np.abs(HLF[:, 3])>=eta_min)
    eta1=HLF[:, 2]
    eta2=HLF[:, 3]
    dphi=HLF[:, 4]

    px1=pt1
    px2=pt2*np.cos(dphi)
    py1=np.zeros_like(pt1)
    py2=pt2*np.sin(dphi)
    pz1=pt1*np.sinh(eta1)
    pz2=pt2*np.sinh(eta2)
    E1 =np.sqrt(px1*px1+py1*py1+pz1*pz1+muon_mass*muon_mass)
    E2 =np.sqrt(px2*px2+py2*py2+pz2*pz2+muon_mass*muon_mass)

    px=px1+px2
    py=py1+py2
    pz=pz1+pz2
    E =E1+E2
    mll=np.sqrt(E*E-px*px-py*py-pz*pz)

    HLF_new=np.copy(HLF)
    HLF_new[:, 0]=pt1
    HLF_new[:, 1]=pt2
    HLF_new[:, 5]=mll
    return HLF_new

def Apply_MuonEfficiency_Correction_ETAregion(weights, feature, muon_efficiency=0, eta_min=0, eta_max=2.4):
    eta1=feature[:, 2]
    eta2=feature[:, 3]
    weights_new  = weights*(np.ones_like(weights) + muon_efficiency*(np.abs(eta1)>=eta_min)*(np.abs(eta1)<eta_max))
    weights_new *= (np.ones_like(weights) + muon_efficiency*(np.abs(eta2)>=eta_min)*(np.abs(eta2)<eta_max))
    return weights_new
