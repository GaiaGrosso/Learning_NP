import tensorflow as tf
import h5py
from tensorflow import keras
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import metrics, losses, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow import Variable
import numpy as np

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range                                                                                               \
                                                                                                                                                                          
    '''
    def __init__(self, c=2):
        self.c = c
    def __call__(self, p):
        return tf.clip_by_value(p, clip_value_min=-self.c, clip_value_max=self.c)
    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}

class BSMfinder(Layer):
    def __init__(self,input_shape, architecture=[1, 4, 1], weight_clipping=1.0):
        super(BSMfinder, self).__init__()
        self.hidden_layers = [Dense(architecture[i+1], input_shape=(architecture[i],), activation='sigmoid',
                                    kernel_constraint = WeightClip(weight_clipping)) for i in range(len(architecture)-2)]
        self.output_layer  = Dense(architecture[-1], input_shape=(architecture[-2],), activation='linear',
                                     kernel_constraint = WeightClip(weight_clipping))
        self.build(input_shape)

    def call(self, x):
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x)
        x = self.output_layer(x)
        return x



class BinStepLayer(Layer):
    def __init__(self, input_shape, edgebinlist):
        super(BinStepLayer, self).__init__()
        self.edgebinlist = edgebinlist
        self.nbins       = self.edgebinlist.shape[0]-1
        self.w1          = np.zeros((2*self.nbins, 1))
        self.w2          = np.zeros((self.nbins, 2*self.nbins))
        self.b1          = np.zeros((2*self.nbins, 1))
        self.weight      = 100.

        # fix the weights and biases                                                                                                                                     
        for i in range(self.nbins+1):
            if i < self.nbins:
                for j in range(self.nbins*2):
                        self.w2[i, j]   =  0.
            if i==0:
                self.w1[2*i, 0] = self.weight
                self.b1[2*i]    = -1.*self.weight*self.edgebinlist[i]
                self.w2[i, 2*i] =  1.
            elif i==self.nbins:
                self.w1[2*i-1, 0] = self.weight
                self.b1[2*i-1]      = -1.*self.weight*self.edgebinlist[i]
                self.w2[i-1, 2*i-1] = -1.
            else:
                self.w1[2*i-1, 0] = self.weight
                self.b1[2*i-1]    = -1.*self.weight*self.edgebinlist[i]
                self.w1[2*i, 0]   = self.weight
                self.b1[2*i]      = -1.*self.weight*self.edgebinlist[i]
                self.w2[i, 2*i-1] =  1.
                self.w2[i-1, 2*i] = -1.

        self.w1 = Variable(initial_value=self.w1.transpose(), dtype="float32", trainable=False, name='w1' )
        self.w2 = Variable(initial_value=self.w2.transpose(), dtype="float32", trainable=False, name='w2' )
        self.b1 = Variable(initial_value=self.b1.transpose(), dtype="float32", trainable=False, name='b1' )
        self.build(input_shape)

    def call(self, x):
        x1 = tf.matmul(x[:,0:1], self.w1) + self.b1
        x2 = tf.matmul(x[:,1:2], self.w1) + self.b1
        x = keras.activations.relu(keras.backend.sign(x1+x2))
        x = tf.matmul(x, self.w2)
        return x

class LinearExpLayer(Layer):
    def __init__(self, input_shape, A0matrix, A1matrix, endcaps_barrel_r):
        super(LinearExpLayer, self).__init__()
        self.a0         = Variable(initial_value=A0matrix[0, :],   dtype="float32", trainable=False, name='a0' )
        self.a1_barrel  = Variable(initial_value=A1matrix[:2, :],  dtype="float32", trainable=False, name='a1b' )
        self.a1_endcaps = Variable(initial_value=A1matrix[2:, :],  dtype="float32", trainable=False, name='a1e' )
        self.ebr        = Variable(initial_value=endcaps_barrel_r, dtype="float32", trainable=False, name='eb_ratio' )
        self.build(input_shape)

    def call(self, x):
        x = tf.matmul(x, self.a1_barrel) +  tf.matmul(tf.math.multiply(self.ebr, x), self.a1_endcaps) + self.a0
        return x # e_j(nu_1,..., nu_i) # [B x Nbins]

class BSMfinderUpgrade(Model):
    def __init__(self, input_shape, edgebinlist, A1matrix, A0matrix, endcaps_barrel_r, NUmatrix, NURmatrix, NU0matrix, SIGMAmatrix, architecture, weight_clipping, name=N\
one, **kwargs):
        super().__init__(name=name, **kwargs)
        self.oi  = BinStepLayer(input_shape, edgebinlist)
        self.ei  = LinearExpLayer(input_shape, A0matrix, A1matrix, endcaps_barrel_r)
        self.eiR = LinearExpLayer(input_shape, A0matrix, A1matrix, endcaps_barrel_r)
        self.nu  = Variable(initial_value=NUmatrix,         dtype="float32", trainable=True,  name='nu')
        self.nuR = Variable(initial_value=NURmatrix,        dtype="float32", trainable=False, name='nuR')
        self.nu0 = Variable(initial_value=NU0matrix,        dtype="float32", trainable=False, name='nu0')
        self.sig = Variable(initial_value=SIGMAmatrix,      dtype="float32", trainable=False, name='sigma')
        self.f   = BSMfinder(input_shape, architecture, weight_clipping)
        self.build(input_shape)

    def call(self, x):
        f       = self.f(x)
        oi      = self.oi.call(x)
        nu      = tf.squeeze(self.nu)
        nuR     = tf.squeeze(self.nuR)
        nu0     = tf.squeeze(self.nu0)
        sigma   = tf.squeeze(self.sig)
        ei      = tf.transpose(self.ei.call(tf.transpose(self.nu)))
        eiR     = tf.transpose(self.eiR.call(tf.transpose(self.nuR)))
        Lbinned = tf.matmul(oi, tf.math.log(ei/eiR))# (batchsize, 1)                                                                                                     
        Laux    = tf.reduce_sum(-0.5*((nu-nu0)**2 - (nuR-nu0)**2)/sigma**2 )# Gaussian (scalar value)                                                                    
        Laux    = Laux*tf.ones_like(f)
        output  = tf.keras.layers.Concatenate(axis=1)([f+Lbinned, Laux])
        self.add_metric(tf.reduce_mean(Laux), aggregation='mean', name='Laux')
        self.add_metric(nu[0], aggregation='mean', name='scale_barrel')
        self.add_metric(nu[1], aggregation='mean', name='efficiency_barrel')
        return output#[out_oi, out_ei, out_eiR, out_f, nu, nuR, nu0, sigma]                                                                                               


def NPLLoss_New(true, pred):
    f   = pred[:, 0]
    Laux= pred[:, 1]
    y   = true[:, 0] # shape (batchsize,       )                                                                                                                          
    w   = true[:, 1] # shape (batchsize,       )                                                                                                                         
    return tf.reduce_sum((1-y)*w*(tf.exp(f)-1) - y*w*(f)) - tf.reduce_mean(Laux)

def ReadFit_PTbins_from_h5(file_name='/eos/user/g/ggrosso/PhD/BSM/Sistematiche/MuMu/SM/PTbinsFits.h5'):
    f = h5py.File(file_name, 'r')
    q_b  = np.array(f.get('q_b'))
    q_e  = np.array(f.get('q_e'))
    m_sb = np.array(f.get('m_sb'))
    m_se = np.array(f.get('m_se'))
    m_eb = np.array(f.get('m_eb'))
    m_ee = np.array(f.get('m_ee'))
    c_sb = np.array(f.get('c_sb'))
    c_se = np.array(f.get('c_se'))
    c_eb = np.array(f.get('c_eb'))
    c_ee = np.array(f.get('c_ee'))
    bins = np.array(f.get('bins'))
    ee   = np.array(f.get('efficiency_endcaps'))
    eb   = np.array(f.get('efficiency_barrel'))
    se   = np.array(f.get('pt_scale_endcaps'))
    sb   = np.array(f.get('pt_scale_barrel'))
    f.close()
    return bins, sb, se, eb, ee, q_b, q_e, m_sb, m_se, m_eb, m_ee, c_sb, c_se, c_eb, c_ee
