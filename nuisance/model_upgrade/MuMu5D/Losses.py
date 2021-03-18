import tensorflow as tf


def NPLLoss(true, pred):
    f   = pred[:, 0]
    y   = true[:, 0]
    w   = true[:, 1]
    return tf.reduce_sum((1-y)*w*(tf.exp(f)-1) - y*w*(f))

def NPLLoss_New(true, pred):
    f   = pred[:, 0]
    Laux= pred[:, 1]
    y   = true[:, 0]
    w   = true[:, 1]
    return tf.reduce_sum((1-y)*w*(tf.exp(f)-1) - y*w*(f)) - tf.reduce_mean(Laux)

def CorrectionParLoss(true, pred):
    Lratio  = pred[:, 0]
    Laux    = pred[:, 1]
    y       = true[:, 0]
    w       = true[:, 1]
    return tf.reduce_sum((1-y)*w*(tf.exp(Lratio)-1) - y*w*(Lratio)) - tf.reduce_mean(Laux)

def CorrectionBinLoss(true, pred):
    Lbinned = pred[:, 0]
    Laux    = pred[:, 1]
    N_R     = pred[:, 2]
    N       = pred[:, 3]
    y       = true[:, 0] # shape (batchsize,       )                                                                                                                                 
    w       = true[:, 1] # shape (batchsize,       )                                                                                                                                 
    return tf.reduce_sum(- y*w*(Lbinned)) - tf.reduce_mean(Laux) + tf.reduce_mean(N-N_R)

def ParametricQuadraticLoss(true, pred):
    a1 = pred[:, 0]
    a2 = pred[:, 1]
    y  = true[:, 0]
    w  = true[:, 1]
    nu = true[:, 2]
    f  = tf.multiply(a1, nu) + tf.multiply(a2, nu**2)
    return tf.reduce_sum((1-y)*w*(tf.exp(f)-1) - y*w*(f))

def ParametricQuadraticPositiveDefiniteLoss(true, pred):
    a1 = pred[:, 0]
    a2 = pred[:, 1]
    y  = true[:, 0]
    w  = true[:, 1]
    nu = true[:, 2]
    f  = tf.math.log((tf.ones_like(a1)+tf.multiply(a1, nu))**2 + tf.multiply(a2**2, nu**2))                                                                                                                         
    return tf.reduce_sum((1-y)*w*(tf.exp(f)-1) - y*w*(f))
