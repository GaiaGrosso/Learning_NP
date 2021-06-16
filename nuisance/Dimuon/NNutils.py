import tensorflow as tf
import h5py
import os
from tensorflow import keras
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import metrics, losses, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow import Variable
from tensorflow import linalg as la
from tensorflow.keras import initializers
import numpy as np
import logging

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range                                                                                                       
    '''
    def __init__(self, c=2):
        self.c = c
    def __call__(self, p):
        return tf.clip_by_value(p, clip_value_min=-self.c, clip_value_max=self.c)
    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}

class BSMfinder(Model):
    def __init__(self,input_shape, architecture=[1, 4, 1], weight_clipping=1.0, activation='sigmoid', trainable=True, initializer=None, name=None, **kwargs):
        kernel_initializer="glorot_uniform"
        bias_initializer="zeros"
        if not initializer==None:
            kernel_initializer = initializer
            bias_initializer = initializer
        super().__init__(name=name, **kwargs)
        self.hidden_layers = [Dense(architecture[i+1], input_shape=(architecture[i],), activation=activation, trainable=trainable,
                                    kernel_constraint = WeightClip(weight_clipping), kernel_initializer=initializer, bias_initializer=initializer) for i in range(len(architectu\
re)-2)]
        self.output_layer  = Dense(architecture[-1], input_shape=(architecture[-2],), activation='linear', trainable=trainable,
                                     kernel_constraint = WeightClip(weight_clipping), kernel_initializer=initializer, bias_initializer=initializer)
        self.build(input_shape)

    def call(self, x):
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x)
        x = self.output_layer(x)
        return x
			
class NPLM_imperfect(Model):
	def __init__(self, input_shape, NU_S, NU_SR, NU_S0, SIGMA_S, NU_N, NU_NR, NU_N0, SIGMA_N, N_Bkg, 
							 BSMarchitecture=[1, 10, 1], BSMweight_clipping=1., correction='1D', 
							 delta_weights=None,  edgebinlist=[],means=[], points=[], poly_coeff=[], binned_var_col=[],
							 train_nu=True, train_f=True, name='NPLM', **kwargs):
		'''
		
		'''
		super().__init__(name=name, **kwargs)
		if correction not in ['NN','BIN_POLY','BIN_POINTS']:
			logging.error("value %s for binning is not valid. Choose between 'NN', 'BIN_POLY', 'BIN_POINTS'. ")
			exit()
			if correction=='NN' and delta_weights==None:
				logging.error("Missing argument 'ParNet_weights', required in the 'NN' correction mode")
				exit()
			if correction=='BIN_POLY' and len(poly_coeff)==0:
				logging.error("Missing content in argument 'poly_coeff', required in the 'BIN_POLY' connection mode")
				exit()
			if correction=='BIN_POINTS' and len(points)==0:
				logging.error("Missing content in argument 'points', required in the 'BIN_POINTS' connection mode")
				exit()
			if 'BIN' in correction:
				self.oi   = BinStepLayer(input_shape, edgebinlist, mean=means) for i in range(len(edgebinlist))]
				self.binned_var_col = binned_var_col
				
			if correction=='BIN_POLY':
				if len(NU_S) != len(poly_coeff):
					logging.error("Number of scale nuisance and length of 'poly_coeff' do not match.")
					exit()
				if len(poly_coeff[0])!=len(edgebinlist)-1:
					logging.error("Mismatch between lengths: must be len('edgebinlist')=len('poly_coeff')+1.")
					exit()
				self.expectation_layers = [PolyInterpolationExpLayer(input_shape, poly_coeff[i]) for i in range(len(poly_coeff))]
			if correction=='BIN_POINTS':
				if len(NU_S) != len(points):
					logging.error("Number of scale nuisance and length of 'points' do not match.")
					exit()
				if len(points[0])!=len(edgebinlist)-1:
					logging.error("Mismatch between lengths: must be len('edgebinlist')=len('points')+1.")
					exit()
				self.expectation_layers = [LinearInterpolationExpLayer(input_shape, points) for i in range(len(points))]
			if correction=='NN':
				delf.deltas = []
				for j in range(len(delta_weights)):
					delta_poly_degree = delta_weights[j].split('/')[-3]
					delta_poly_degree = int(delta_poly_degree.split('_')[0])
					self.delta_poly_degree = delta_poly_degree
					delta_architecture = delta_weights[j].split('layers', 1)[1]
					delta_architecture = delta_architecture.split('_act', 1)[0]
					delta_architecture = delta_architecture.split('_')
					delta_layers       = []
					for layer in delta_architecture:
						layersPar.append(int(layer))
					delta_inputsize = delta_layers[0]
					delta_input     = (None, inputsdelta_inputsizeizePar)
					delta_activation= delta_weights[j].split('act', 1)[1]
					delta_activation= delta_activation.split('_', 1)[0]
					delta_wc        = delta_weights[j].split('wclip', 1)[1]
					delta_wc        = float(delta_wc.split('/', 1)[0])
					delta_sb        = delta_weights[j].split('sb', 1)[1]
					delta_sb        = float(delta_sb.split('_', 1)[0])
					self.delta_sb   = delta_sb
					self.deltas.append(ParametricNet(delta_input, delta_layers, delta_wc, delta_activation, poly_degree=delta_poly_degree) )
					self.deltas[-1].load_weights(delta_weights[j])
					#don't want to train Delta                                                                  
					for module in self.deltas[-1].layers:
						for layer in module.layers:
							layer.trainable = False
				self.nu_s   = Variable(initial_value=NU_S,         dtype="float32", trainable=train_nu,  name='nu_s')
				self.nuR_s  = Variable(initial_value=NUR_S,        dtype="float32", trainable=False,     name='nuR_s')
				self.nu0_s  = Variable(initial_value=NU0_S,        dtype="float32", trainable=False,     name='nu0_s')
				self.sig_s  = Variable(initial_value=SIGMA_S,      dtype="float32", trainable=False,     name='sigma_s')
				self.nu_n   = Variable(initial_value=NU_N,         dtype="float32", trainable=train_nu,  name='nu_n')
				self.nuR_n  = Variable(initial_value=NUR_N,        dtype="float32", trainable=False,     name='nuR_n')
				self.nu0_n  = Variable(initial_value=NU0_N,        dtype="float32", trainable=False,     name='nu0_n')
				self.sig_n  = Variable(initial_value=SIGMA_N,      dtype="float32", trainable=False,     name='sigma_n')
				self.N_Bkg  = Variable(initial_value=N_Bkg,        dtype="float32", trainable=False,     name='N_Bkg')
				if train_f: self.BSMfinder = BSMfinder(input_shape, BSMarchitecture, BSMweight_clipping)
				if len(N_Bkg) != len(NU_N):
					logging.error("'N_Bkg' must have the same length as 'NU_N'.")
					exit()
				self.train_f = train_f
				self.correction = correction
				self.build(input_shape)
			
		def call(self, x):
			nu_s     = tf.squeeze(self.nu_s)
     	nuR_s    = tf.squeeze(self.nuR_s)
     	nu0_s    = tf.squeeze(self.nu0_s)
      sigma_s  = tf.squeeze(self.sig_s)
			nu_n     = tf.squeeze(self.nu_n)
     	nuR_n    = tf.squeeze(self.nuR_n)
     	nu0_n    = tf.squeeze(self.nu0_n)
      sigma_n  = tf.squeeze(self.sig_n)
			
			# Auxiliary likelihood (gaussian prior)
      Laux     = tf.reduce_sum(-0.5*((nu_s-nu0_s)**2 - (nuR_s-nu0_s)**2)/sigma_s**2 ) + tf.reduce_sum(-0.5*((nu_n-nu0_n)**2 - (nuR_n-nu0_n)**2)/sigma_n**2 )
			Laux     = Laux*tf.ones_like(x[:, 0:1])
			
			# scale effects
      Lratio  = 0
			if 'BIN' in self.correction:
				oi  = self.oi.call(x[:, self.binned_var_col:self.binned_var_col+1])
				ei  = 0
				eiR = 0
				for j in range(len(self.expectation_layers)):
					ei  += self.expectation_layers[j].call(self.nu_s[j])          
					eiR += self.expectation_layers[j].call(self.nuR_s[j])
				Lratio = tf.matmul(oi, tf.math.log(ei/eiR))
					
			if 'PAR' in self.correction :
				for j in range(len(self.deltas)):
					delta = self.delta[j].call(x)
					for i in range(self.poly_degree):
						Lratio += delta[:, i:i+1]*(nu_s[j]/self.delta_sb[j])**(i+1)
			
			# normalization effects
			Lratio += tf.math.log( tf.reduce_sum(tf.multiply(self.N_Bkg, tf.exp(nu_n)))/tf.reduce_sum(tf.multiply(self.N_Bkg, tf.exp(nuR_n))) )
			BSM     = tf.zeros_like(Laux)
														
      if self.train_f:		
      	BSM = self.f(x)
     	output  = tf.keras.layers.Concatenate(axis=1)([BSM+Lratio, Laux])
      self.add_metric(tf.reduce_mean(Laux), aggregation='mean', name='Laux')
			for i in range(len(nu_s)):
      	self.add_metric(nu_s[i], aggregation='mean', name='scale_%i'%(i))
			for i in range(len(nu_n)):
      	self.add_metric(nu_n[i], aggregation='mean', name='norm_%i'%(i))
      return output
		
class BinStepLayer(Layer):
    def __init__(self, input_shape, edgebinlist, mean):
        super(BinStepLayer, self).__init__()
        self.edgebinlist = edgebinlist
        self.nbins       = edgebinlist.shape[0]-1
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
        self.mean = Variable(initial_value=mean, dtype="float32", trainable=False, name='mean' )
        self.build(input_shape)

    def call(self, x):
        x = tf.matmul(x*self.mean, self.w1) + self.b1
        x = keras.activations.relu(keras.backend.sign(x))
        x = tf.matmul(x, self.w2)
        return x

class PolyExpLayer(Layer):
    def __init__(self, input_shape, coeff):
        super(PolyExpLayer, self).__init__()
				self.a = []
				for i in range(len(coeff)):
					self.a.append(Variable(initial_value=coeff[i, :],   dtype="float32", trainable=False, name='a%i'%(i) ))
        self.build(input_shape)

    def call(self, x):
      y   = 0
			for i in range(len(self.a)):
				y += tf.math.multiply(self.a[i], x**i)   
			return y

class LinearInterpolationExpLayer(Layer):
    def __init__(self, input_shape, points):
        super(LinearInterpolationExpLayer, self).__init__()
        self.nbins       = points.shape[0]
        self.npoints     = points.shape[1]
        self.x           = points[0, :, 0]
        self.a1          = (points[:,1:, 1]-points[:,:-1, 1])/(points[:,1:, 0]-points[:,:-1, 0])
        self.a0          = (points[:,:-1, 1] - self.a1*points[:,:-1, 0]) +1e-10

        self.b1          = np.zeros((self.npoints, 1))
        self.w1          = np.zeros((self.npoints, 1))
        self.w2          = np.zeros((self.npoints-1, self.npoints))
        self.b2          = np.zeros((self.npoints-1, 1))
        self.weight      = 100.

        for i in range(self.npoints):
            if i==0 or i==(self.npoints-1):
                continue
            self.w1[i, 0] = self.weight
            self.b1[i, 0] = -1*self.weight*self.x[i]
        for i in range(self.npoints-1):
            self.w2[i, i]   = 1.
            self.w2[i, i+1] = -1.
            if i==0:
                self.b2[i, 0] = 1

        self.w1 = Variable(initial_value=self.w1, dtype="float32", trainable=False, name='w1' )
        self.w2 = Variable(initial_value=self.w2, dtype="float32", trainable=False, name='w2' )
        self.b1 = Variable(initial_value=self.b1, dtype="float32", trainable=False, name='b1' )
        self.b2 = Variable(initial_value=self.b2, dtype="float32", trainable=False, name='b2' )
        self.a0 = Variable(initial_value=self.a0, dtype="float32", trainable=False, name='a0' )
        self.a1 = Variable(initial_value=self.a1, dtype="float32", trainable=False, name='a1' )
        self.build(input_shape)
				
		def call(self, nu):
        scale = nu[:, 0:1] # [1, 1]                                                                                              
        norm  = nu[:, 1:2] # [1, 1]                                                                                              
        y = tf.matmul(self.w1, scale)+self.b1
        y = keras.activations.relu(keras.backend.sign(y)) # [npoints, 1]                                                         
        proj       = tf.matmul(self.w2, y) + self.b2  # [npoints-1, 1]                                                           
        proj_scale = tf.matmul(proj, scale) # [npoints-1, 1]                                                                     
        return tf.multiply(tf.matmul(self.a1, proj_scale)+tf.matmul(self.a0, proj), tf.ones((self.nbins, 1))+tf.matmul(tf.ones((self.nbins, 1)), norm))

class ParametricNet(Model):
    def __init__(self, input_shape, architecture=[1, 10, 1], weight_clipping=1., activation='sigmoid', degree=1,
                 initial_model=None, init_null=[False], train=[True] name="ParNet", **kwargs):
        super().__init__(name=name, **kwargs)
        self.a = []
				for i in range(degree):
					initializer = None
        	if null[i]:
            initializer = initializers.Zeros()
       		self.a.append(BSMfinder(input_shape, architecture, weight_clipping, activation=activation, trainable=train[i], initializer=initializer))
				self.degree = degree
				self.build(input_shape)
        if not initial_model == None:
        	self.load_weights(initial_model, by_name=True)
			
		def call(self, x):
			output = []
			for i in range(self.degree):
				output.append(self.a[i](x))
      output  = tf.keras.layers.Concatenate(axis=1)(output)
      return output

			
def NPLM_Imperfect_Loss(true, pred):
	f   = pred[:, 0]
	Laux= pred[:, 1]
	y   = true[:, 0]
	w   = true[:, 1]
	return tf.reduce_sum((1-y)*w*(tf.exp(f)-1) - y*w*(f)) - tf.reduce_mean(Laux)
														
			
