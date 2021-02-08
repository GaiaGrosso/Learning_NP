import glob, os, h5py, time, datetime, sys, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class WeightClipping(object):
    def __init__(self, wc=1):
        self.wc = wc
    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight
            w = torch.le(w, 0)*torch.max(w, -1*torch.ones_like(w)*self.wc)+ torch.ge(w, 0)*torch.min(w, torch.ones_like(w)*self.wc)

class BinStepLayer(nn.Module):
    def __init__(self, edgebinlist):
        super(BinStepLayer, self).__init__()
        self.edgebinlist = edgebinlist #torch.arange(0, 2.5, 0.05)
        self.nbins       = self.edgebinlist.shape[0]-1
        self.layer1      = torch.nn.Linear(1, self.nbins*2)
        self.layer2      = torch.nn.Linear(self.nbins*2, self.nbins, bias=False)
        self.weight      = 100.
        
        # fix the weights and biases
        with torch.no_grad():
            for i in range(self.nbins+1):
                if i < self.nbins:
                    for j in range(self.nbins*2):
                            self.layer2.weight[i, j]   =  0.
                if i==0:
                    self.layer1.weight[2*i, 0]   = self.weight
                    self.layer1.bias[2*i]        = -1.*self.weight*self.edgebinlist[i]
                    self.layer2.weight[i, 2*i]   =  1.
                elif i==self.nbins:
                    self.layer1.weight[2*i-1, 0] = self.weight
                    self.layer1.bias[2*i-1]      = -1.*self.weight*self.edgebinlist[i]
                    self.layer2.weight[i-1, 2*i-1] = -1.
                else:
                    self.layer1.weight[2*i-1, 0] = self.weight
                    self.layer1.bias[2*i-1]      = -1.*self.weight*self.edgebinlist[i]
                    self.layer1.weight[2*i, 0]   = self.weight
                    self.layer1.bias[2*i]        = -1.*self.weight*self.edgebinlist[i]
                    self.layer2.weight[i, 2*i-1] =  1.
                    self.layer2.weight[i-1, 2*i] = -1.
                    
        # freeze layers 1,2 parameters (not trainable!)
        for param in self.parameters():
            param.requires_grad = False
                    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(torch.sign(x))
        x = self.layer2(x)
        return x

class BSM(nn.Module):
    def __init__(self, architecture=[1, 4, 1], weight_clipping=1.):
        super(BSM, self).__init__()
        self.wclip      = weight_clipping
        self.layers     = nn.ModuleList([nn.Linear(architecture[i], architecture[i+1]) for i in range(len(architecture)-2)])
        self.layer_out  = nn.Linear(architecture[-2], architecture[-1])
        self.activation = nn.Sigmoid()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x))
        x = self.layer_out(x)
        x = torch.squeeze(x)
        return x

    def WeightClipping(self):
        with torch.no_grad():
            for i, m in enumerate(self.layers):
                m.weight.masked_scatter_(m.weight>self.wclip, nn.Parameter(torch.ones_like(m.weight)*self.wclip))
                m.weight.masked_scatter_(m.weight<-self.wclip, nn.Parameter(torch.ones_like(m.weight)*-1*self.wclip))

            self.layer_out.weight.masked_scatter_(self.layer_out.weight>self.wclip, nn.Parameter(torch.ones_like(self.layer_out.weight)*self.wclip))
            self.layer_out.weight.masked_scatter_(self.layer_out.weight<-self.wclip, nn.Parameter(torch.ones_like(self.layer_out.weight)*-1*self.wclip))
        return 

class ExpLayer(nn.Module):
    def __init__(self, n_nuisance, edgebinlist, Mmatrix, Qmatrix):
        super(ExpLayer, self).__init__()
        self.edgebinlist = edgebinlist #torch.arange(0, 1.5, 0.05)
        self.nbins       = self.edgebinlist.shape[0]-1
        self.n_nuisance  = n_nuisance
        self.q           = Qmatrix #torch.rand(self.n_nuisance, self.nbins)
        self.m           = Mmatrix #torch.rand(self.n_nuisance, self.nbins)
        self.layer2      = torch.nn.Linear(self.n_nuisance, self.nbins)    # linear fit binwise: b_i=q_ij, w_i=m_ij
    
        # fix the weights and biases
        with torch.no_grad():
            for i in range(self.n_nuisance):
                for j in range(self.nbins):
                    self.layer2.bias[j]      = self.q[i,j]
                    self.layer2.weight[j, i] = self.m[i,j]
        
        # freeze layers 2 parameters (not trainable!)
        for param in self.parameters():
            param.requires_grad = False
                
    def forward(self, x):
        x = self.layer2(x)
        return x # e_j(nu_1,..., nu_i) # [B x Nbins]
    
class NewModel(nn.Module):
    def __init__(self, n_nuisance, edgebinlist, Mmatrix, Qmatrix, NUmatrix, NURmatrix, NU0matrix, SIGMAmatrix, architecture):
        super(NewModel, self).__init__()
        self.oi  = BinStepLayer(edgebinlist)
        self.ei  = ExpLayer(n_nuisance, edgebinlist, Mmatrix, Qmatrix)
        self.eiR = ExpLayer(n_nuisance, edgebinlist, Mmatrix, Qmatrix)
        self.nu  = Variable(NUmatrix,  requires_grad=True)
        self.nuR = Variable(NURmatrix, requires_grad=False)
        self.nu0 = Variable(NU0matrix, requires_grad=False)
        self.sig = Variable(SIGMAmatrix, requires_grad=False)
        self.f   = BSM(architecture)
        
    def forward(self, x):
        out_f   = self.f(x)
        out_oi  = self.oi(x)
        nu      = torch.squeeze(self.nu)
        nuR     = torch.squeeze(self.nuR)
        nu0     = torch.squeeze(self.nu0)
        sigma   = torch.squeeze(self.sig)
        out_ei  = self.ei(self.nu.t())
        out_eiR = self.eiR(self.nuR.t())
        return [out_oi, out_ei, out_eiR, out_f, nu, nuR, nu0, sigma]
    
def NPLLoss_New(true, pred):
    oi  = pred[0] # shape (bathcsize, n_bins)
    ei  = pred[1] # shape (bathcsize, n_bins)
    eiR = pred[2] # shape (bathcsize, n_bins)
    f   = pred[3] # shape (batchsize,       )
    nu  = pred[4] # shape (n_nuisance,      )
    nuR = pred[5] # shape (n_nuisance,      )
    nu0 = pred[6] # shape (n_nuisance,      )
    sig = pred[7] # shape (n_nuisance,      )
    y   = true[0] # shape (batchsize,       )
    w   = true[1] # shape (batchsize,       )
    Lbinned = torch.sum(oi * torch.log(ei/eiR), 1)
    Laux    = -0.5*((nu-nu0)**2 - (nuR-nu0)**2)/sig**2 # Gaussian
    return torch.sum((1-y)*w*(torch.exp(f+Lbinned)-1) - y*w*(f+Lbinned)) - torch.sum(Laux)

def Read_FitBins(filename):
    '''
    Load the results of the fit of each bin dependence from the nuisance
    bins = list of bins edges (length = number of bins + 1)
    q    = list of intercepts 
    m    = list of linear coefficients
    c    = list of quadratic coefficients
    '''
    f = h5py.File(filename, "r")
    q = np.array(f.get("q"))
    m = np.array(f.get("m"))
    c = np.array(f.get("c"))
    b = np.array(f.get("bins"))
    n = np.array(f.get("nuisance"))
    f.close()
    return q, m, c, b, n
    
    
## 1D TEST #################################################
# DATA #
data = np.random.exponential(scale=1.05, size=(2000,1))
ref  = np.random.exponential(scale=1,    size=(200000,1))
data = np.concatenate((data, ref), axis=0)
tgt  = np.append(np.ones(2000),      np.zeros(200000))
w    = np.append(np.ones(2000), 0.01*np.ones(200000))
data = torch.from_numpy(data).double()
tgt  = torch.from_numpy(tgt).double()
w    = torch.from_numpy(w).double()

# PARAMETERS ##############
q_SCALE, m_SCALE, c_SCALE, bins, nu_fitSCALE = Read_FitBins('/eos/user/g/ggrosso/PhD/BSM/Sistematiche/Expo1D/Expo1D_BinFitSCALE.h5')
q_NORM,  m_NORM,  c_NORM,  bins, nu_fitNORM  = Read_FitBins('/eos/user/g/ggrosso/PhD/BSM/Sistematiche/Expo1D/Expo1D_BinFitNORM.h5')
Mmatrix     = torch.from_numpy(np.concatenate((m_SCALE.reshape(1,-1), m_NORM.reshape(1,-1)), axis=0)).double()
Qmatrix     = torch.from_numpy(np.concatenate((q_SCALE.reshape(1,-1), q_NORM.reshape(1,-1)), axis=0)).double()
NUmatrix    = torch.from_numpy(np.concatenate((nu_fitSCALE[5:6].reshape(1,-1), nu_fitNORM[5:6].reshape(1,-1)), axis=0)).double()
NURmatrix   = torch.from_numpy(np.concatenate((nu_fitSCALE[5:6].reshape(1,-1), nu_fitNORM[5:6].reshape(1,-1)), axis=0)).double()
NU0matrix   = torch.from_numpy(np.array([[np.random.normal(loc=1.05, scale=0.05,size=1)[0], np.random.normal(loc=1., scale=0.05,size=1)[0]]])).double()
SIGMAmatrix = torch.from_numpy(np.array([[0.05, 0.05]])).double()

# MODEL ###################
BSMarchitecture = [1, 4, 1]
weight_clipping = 8
model     = NewModel(n_nuisance=2, edgebinlist=bins, 
                     Mmatrix=Mmatrix, Qmatrix=Qmatrix, 
                     NUmatrix=NUmatrix, NURmatrix=NURmatrix,
                     NU0matrix=NU0matrix, SIGMAmatrix=SIGMAmatrix,
                     architecture=architecture, weight_clipping=weight_clipping).double()
loss      = NPLLoss_New
trainpars = [{'params': model._modules['f'].parameters()}, {'params': model.nu}]# {'params': model._modules['nu'].parameters()}]
optimizer = torch.optim.Adam(trainpars)

# TRAINING ###############
epochs   = 100000
patience = 1000
for epoch in range(epochs):
    model.zero_grad()
    output_i = model(data)
    loss_i   = loss([tgt, w], output_i)
    loss_i.backward()
    optimizer.step()
    model._modules['f'].apply(clipper)
    if not epoch%patience:
        print('Epoch: %i, Loss: %f'%(epoch, loss_i.item()))
