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

class DNN(Model):
    def __init__(self, input_shape, architecture=[1, 4, 1], weight_clipping=1.0, activation='sigmoid', name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.hidden_layers = [Dense(architecture[i+1], input_shape=(architecture[i],), activation=activation,
                                    kernel_constraint = WeightClip(weight_clipping)) for i in range(len(architecture)-2)]
        self.output_layer  = Dense(architecture[-1], input_shape=(architecture[-2],), activation='linear',
                                     kernel_constraint = WeightClip(weight_clipping))
        self.build(input_shape)

    def call(self, x):
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x)
        x = self.output_layer(x)
        return x

class ParametricNet(Model):
    def __init__(self, input_shape, architecture=[[1, 10, 1]], weight_clipping=[1.], activation='sigmoid', name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.architecture = architecture
        self.wclip = weight_clipping
        if len(architecture)==1:
            self.architecture.append(architecture[0])
        if len(self.wclip)==1:
			self.wclip.append(weightclipping[0])
        self.layers = [DNN(input_shape=input_shape, architecture=self.architecture[i], weight_clipping=self.wclip[i], activation=activation) for i in range(2)]
        self.build(input_shape)

    def call(self, x):
		output = [self.layers[i](x) for i in range(2)]
        output  = tf.keras.layers.Concatenate(axis=1)(output)
        return output

class NPLM(Model):
    def __init__(self, input_shape, NU, NUR, NU0, SIGMA, architecture=[1, 10, 1], weight_clipping=1., ParNet_weights=None, train_nu=True, train_BSM=True, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        
		architecturePar = ParNet_weights.split('layers', 1)[1]
		architecturePar = architecturePar.split('_act', 1)[0]
		architecturePar = architecturePar.split('_')
		layersPar       = []
		for layer in architecturePar:
			layersPar.append(int(layer))
		inputsizePar    = layersPar[0]
		input_shapePar  = (None, inputsizePar)
		activationPar   = ParNet_weights.split('act', 1)[1]
		activationPar   = activationPar.split('_', 1)[0]
		wcPar           = ParNet_weights.split('wclip', 1)[1]
		wcPar           = float(wcPar.split('/', 1)[0])
		self.Delta = ParametricNet(input_shapePar, architecturePar, wcPar, activationPar)
		self.Delta.load_weights(ParNet_weights)
		#don't want to train Delta                                                                                                                                               
		for module in self.Delta.layers:
			for layer in module.layers:
				layer.trainable = False

        self.nu   = Variable(initial_value=NU,         dtype="float32", trainable=train_nu,  name='nu')
        self.nuR  = Variable(initial_value=NUR,        dtype="float32", trainable=False,     name='nuR')
        self.nu0  = Variable(initial_value=NU0,        dtype="float32", trainable=False,     name='nu0')
        self.sig  = Variable(initial_value=SIGMA,      dtype="float32", trainable=False,     name='sigma')
        if train_BSM:
            self.BSMfinder    = DNN(input_shape, architecture, weight_clipping)
        self.train_BSM = train_BSM
        self.build(input_shape)
		
	def call(self, x):
        nu      = tf.squeeze(self.nu)
        nuR     = tf.squeeze(self.nuR)
        nu0     = tf.squeeze(self.nu0)
        sigma   = tf.squeeze(self.sig)
		
        Laux    = tf.reduce_sum(-0.5*((nu-nu0)**2 - (nuR-nu0)**2)/sigma**2 )
        Laux    = Laux*tf.ones_like(x[:, 0:1])

        Lratio  = 0
		delta   = self.Delta.call(x)
		Lratio  = tf.math.log((1+delta[:, 0:1]*nu[0]/sigma[0])**2  + (delta[:, 1:2]*nu[0]/sigma[0])**2) # scale
		Lratio += tf.math.log((tf.ones_like(delta[:, 1:2])+nu[1])/(tf.ones_like(delta[:, 1:2])+nuR[1])) # norm
        BSM     = tf.zeros_like(Laux)
        if self.train_BSM:
            BSM = self.BSMfinder(x)
        output  = tf.keras.layers.Concatenate(axis=1)([BSM+Lratio, Laux])
        self.add_metric(tf.reduce_mean(Laux), aggregation='mean', name='Laux')
        self.add_metric(nu[0], aggregation='mean', name='scale_barrel')
        self.add_metric(nu[1], aggregation='mean', name='efficiency_barrel')
        return output
