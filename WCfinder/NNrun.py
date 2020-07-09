import numpy as np
import datetime
import copy
from scipy.stats import chi2

#import tensorflow as tf
#import keras.backend as K
#from keras.constraints import Constraint

import torch
import torch.nn as nn
import torch.nn.functional as F

# PYTORCH #################################################################
class Net(nn.Module):

    def __init__(self, architecture=[1, 4, 1]):
        super(Net, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(architecture[i], architecture[i+1]) for i in range(len(architecture)-2)])
        self.output_layer  = nn.Linear(architecture[-2], architecture[-1])
        self.activation    = nn.Sigmoid()
        

    def forward(self, x):
        #x = self.input_layer(x)
        #x = self.sig_act(x)
        for i, l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        out = self.output_layer(x)
        out = torch.squeeze(out)
        return out

class WeightClipping(object):

    def __init__(self, wc=1):
        self.wc = wc

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight#.numpy()
            
            w = torch.le(w, 0)*torch.max(w, -1*torch.ones_like(w)*self.wc)+ torch.ge(w, 0)*torch.min(w, torch.ones_like(w)*self.wc)
            #print(w)
            #w.apply_(torch.le torch.max(w, 2, 1).expand_as(w))
            #w = torch.from_numpy(w)

############# Loss function definition ####

class _Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class BSM_Loss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(BSM_Loss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target, N_D, N_R):
        return torch.sum(-1.*torch.mul(target, input) + N_D/float(N_R)*torch.mul((1-target),(torch.exp(input)-1)))

def DeepCopy(input):
    output = copy.deepcopy(input)
    #output = output.cuda()
    return output
"""
# KERAS ########################################################################################

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range                                     \
                                                                                                               \
                                                                                                                
    '''
    def __init__(self, c=2):
        self.c = c
    def __call__(self, p):
        return K.clip(p, -self.c, self.c)
    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}
    
class ModelSigmoid(tf.keras.Model):

    def __init__(self, hidden_layers = [4], weight_clipping = 1.):
        super(ModelSigmoid, self).__init__()
        
        self.dense_input  =  tf.keras.layers.Dense(hidden_layers[0],   
                                                   activation='sigmoid'
                                                  )
        self.dense_layers = [tf.keras.layers.Dense(hidden_layers[i+1], 
                                                   activation='sigmoid', 
                                                   kernel_constraint = WeightClip(weight_clipping)
                                                  ) 
                             for i in range(len(hidden_layers)-1)
                            ]
        self.dense_output =  tf.keras.layers.Dense(1, 
                                                   activation='linear', 
                                                   kernel_constraint = WeightClip(weight_clipping)
                                                  )
        return

    def call(self, inputs):
        x     = self.dense_input(inputs)
        for l in self.dense_layers:
            x = l(x)
        out   = self.dense_output(x)
        return out[:, 0]

def Loss(yTrue, yPred, N_D, N_R):
    return K.sum(-1.*yTrue*(yPred) + (1-yTrue)*N_D/float(N_R)*(K.exp(yPred)-1))

def KerasLoss(N_D, N_R):
    def loss(yTrue, yPred):
        return K.sum(-1.*yTrue*(yPred) + (1-yTrue)*N_D/float(N_R)*(K.exp(yPred)-1))
    return loss

#####################################################################################################
"""

class NNrun:
    def __init__(self, wc, architecture, Nref, Ndata, seed, epochs, patience, data_loader, GPU=0, data_preprocess=True,
                 verbose=False):
        if np.mod(int(epochs),int(patience)):
            raise Exception('patience must be a divisor of epochs')
        if (GPU> len(tf.config.experimental.list_physical_devices('GPU'))-1) and (len(tf.config.experimental.list_physical_devices('GPU'))>0):
            raise Exception('the selected GPU is out of range. Choose a GPU between 0 and %i'%(len(tf.config.experimental.list_physical_devices('GPU'))-1))
        self.wc               = wc
        self.architecture     = architecture
        self.hidden_layers    = architecture[1:-1]
        self.seed             = seed
        self.N_R              = Nref
        self.N_D              = Ndata
        self.N_D_pois         = np.random.poisson(lam=Ndata, size=1)[0]
        self.feature          = 0
        self.target           = 0
        self.weights          = 0
        self.t                = 0
        self.data_preprocess  = data_preprocess
        self.epochs           = int(epochs)
        self.patience         = int(patience)
        self.data_loader      = data_loader
        #self.Model            = ModelSigmoid(hidden_layers = self.hidden_layers, weight_clipping = self.wc)
        self.Model_TORCH      = Net(architecture=architecture)
        #self.TFLoss           = Loss
        #self.KERASLoss        = KerasLoss
        self.TORCHLoss        = BSM_Loss()
        #self.optimizer        = tf.keras.optimizers.Adam() # fixed
        self.optimizer_TORCH  = torch.optim.Adam(self.Model_TORCH.parameters())
        self.run_time         = 0
        self.verbose          = verbose
        self.CPUdevice        = '/device:CPU:0'
        self.GPUdevice        = '/device:GPU:'+str(GPU)
        if torch.cuda.is_available():
        #if len(tf.config.experimental.list_physical_devices('GPU')):
            print('Training on GPU')
            self.GPU    = GPU#self.GPUdevice
            self.device = torch.device('cuda')
        else:
            print('Training on CPU')
            self.device = self.CPUdevice
            
        return
    
    def load_data(self):
        if torch.cuda.is_available():
            with torch.cuda.device(self.GPU):
        #with tf.device(self.device):
                feature_REF, feature_DATA, target_REF, target_DATA = self.data_loader(self.seed, self.N_R, self.N_D_pois)
                # standardization
                if self.data_preprocess:
                    for j in range(feature_REF.shape[1]):
                        vec  = feature_REF[:, j]
                        mean = np.mean(vec)
                        std  = np.std(vec)
                        if np.min(vec)<0:
                            feature_REF[:, j]  = (feature_REF[:, j]-mean)*1./std
                            feature_DATA[:, j] = (feature_DATA[:, j]-mean)*1./std
                        else:
                            feature_REF[:, j]  = feature_REF[:, j]/mean
                            feature_DATA[:, j] = feature_DATA[:, j]/mean
                self.feature = np.concatenate((feature_REF, feature_DATA), axis=0)
                self.target  = np.concatenate((target_REF, target_DATA), axis=0)
        return
    """    
    def run_model_TF2(self):
        with tf.device(self.device):
            if self.verbose:
                print('start running at: '+str(datetime.datetime.now()))
            t0 = datetime.datetime.now()
            for e in range(self.epochs):
                with tf.GradientTape() as tape:
                    prediction = self.Model(self.feature, training=True)
                    loss       = self.TFLoss(self.target, prediction, self.N_D, self.N_R)
                    gradients  = tape.gradient(loss, self.Model.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.Model.trainable_variables))
            t1 = datetime.datetime.now()
            if self.verbose:
                print('end running at: '+str(datetime.datetime.now()))
            # run time
            run_time = t1 - t0
            self.run_time = run_time.seconds+run_time.microseconds*1.e-6
            # compute t
            self.t = -2*loss.numpy()
        return
    
    def run_model_KERAS(self):
        with tf.device(self.device):
            self.Model.compile(loss = self.KERASLoss(self.N_D, self.N_R),  optimizer = self.optimizer)
            if self.verbose:
                print('start running at: '+str(datetime.datetime.now()))
            t0 = datetime.datetime.now()
    
            batch_size = self.feature.shape[0]
            hist       = self.Model.fit(self.feature, self.target, 
                                        batch_size=batch_size, epochs=self.epochs, 
                                        shuffle=False, verbose=0)
            t1 = datetime.datetime.now()
            if self.verbose:
                print('end running at: '+str(datetime.datetime.now()))
            # run time
            run_time = t1 - t0
            self.run_time = run_time.seconds+run_time.microseconds*1.e-6
            # compute t
            pred = self.Model.predict(self.feature)
            loss = self.TFLoss(self.target, pred, self.N_D, self.N_R)
            with tf.Session() as sess:  self.t =-2*loss.eval()
        return
    """
    def run_model_TORCH(self):
        clipper = WeightClipping(wc = self.wc)
        if self.verbose:
            print('start running at: '+str(datetime.datetime.now()))
        t0 = datetime.datetime.now()
        if torch.cuda.is_available():
            with torch.cuda.device(self.GPU):
                with torch.no_grad():                
                    featureIN      = torch.from_numpy(self.feature)
                    targetIN       = torch.from_numpy(self.target)
                    featureIN      = featureIN.double()
                    targetIN       = targetIN.double()
                    feature        = DeepCopy(featureIN)
                    target         = DeepCopy(targetIN)
                    feature        = feature.to(self.device)
                    target         = target.to(self.device)
                    # model compile
                    modelIN   = self.Model_TORCH
                    modelIN   = modelIN.double()
                    model     = DeepCopy(modelIN)
                    model     = model.to(self.device)
                    optimizer = self.optimizer_TORCH
                    loss      = self.TORCHLoss 
                
                # model training
                for i in range(self.epochs):
                    output_i = model(feature)
                    loss_i   = loss(output_i, target, self.N_D, self.N_R)
                    optimizer.zero_grad()
                    loss_i.backward()
                    optimizer.step()
                    model.apply(clipper)
                t1 = datetime.datetime.now()
                if self.verbose:
                    print('end running at: '+str(datetime.datetime.now()))
                # run time
                run_time = t1 - t0
                self.run_time = run_time.seconds+run_time.microseconds*1.e-6
                
                self.t = -2*loss_i.item()
                return