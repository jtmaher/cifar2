from keras.layers import Dense, Activation, Flatten, BatchNormalization, Lambda
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, Conv2DTranspose
from keras.regularizers import l2
from keras.layers.merge import add
from keras.backend import clip, tanh

reg=1e-4
init='glorot_normal'

def BN(x):
    return BatchNormalization(axis=1)(x)

def R(x):
    return Activation('relu')(x)

def C11(x, n, bias=False):
    return Conv2D(
        n, (1, 1),
        kernel_initializer=init,
        kernel_regularizer=l2(reg),
        use_bias=bias)(x)

def C(x, n, sz=3, st=1):
    return Conv2D(
        n, (sz, sz),
        strides=(st,st),
        padding='same',
        kernel_initializer=init,
        kernel_regularizer=l2(reg),
        use_bias=False,
    )(x)


def D(x, n, sz=3, st=1):
    return Conv2DTranspose(
        n, (sz, sz),
        strides=(st,st),
        padding='same',
        kernel_initializer=init,
        kernel_regularizer=l2(reg),
        use_bias = False,
    )(x)


def CL(x, thresh=2.5):
    return Lambda(lambda x: clip(x, -thresh, thresh))(x)

def SCL(x, thresh=2.5):
    return Lambda(lambda x: thresh * tanh(x / thresh))(x)

def RES(xin, in_filt, hidden_filt):
    x_1 = x = R(BN(xin))
    x = C11(x, hidden_filt)
    x = R(BN(x))
    x = C(x, hidden_filt)
    x = R(BN(x))
    x = C11(x, in_filt, bias=True)
    x = add([x, x_1])
    return x

def CRES(xin, in_filt, hidden_filt):
    x_1 = x = CL(BN(xin))
    x = C11(x, hidden_filt)
    x = CL(BN(x))
    x = C(x, hidden_filt)
    x = CL(BN(x))
    x = C11(x, in_filt, bias=True)
    x = add([x, x_1])
    return x

def SCRES(xin, in_filt, hidden_filt):
    x_1 = x = SCL(BN(xin))
    x = C11(x, hidden_filt)
    x = SCL(BN(x))
    x = C(x, hidden_filt)
    x = SCL(BN(x))
    x = C11(x, in_filt, bias=True)
    x = add([x, x_1])
    return x

def parameters(model):
    import numpy as np
    return np.sum([np.prod([int(t) for t in s.shape]) for s in model.weights])
