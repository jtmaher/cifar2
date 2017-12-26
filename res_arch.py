from blocks import C, D, R, BN, C11, RES, parameters, reg
from keras.layers import Input, Lambda
from keras.models import Model
from keras.backend import clip

def get_model(args):
    global reg
    reg = args.reg
    
    input_img = Input((3,32,32))
    x = input_img

    x = R(BN(x))
    x = C(x, 128, 7, 2)

    x = R(BN(x))
    x = C(x, 128, 3, 2)

    x = R(BN(x))
    x = C(x, 128, 3, 2)

    x = R(BN(x))
    x = C(x, 128, 3, 2)

    x = R(BN(x))
    x = C(x, 128, 3, 2)

    x = R(BN(x))
    x = D(x, 128, 3, 2)

    x = R(BN(x))
    x = D(x, 128, 3, 2)

    x = R(BN(x))
    x = D(x, 128, 3, 2)

    x = R(BN(x))
    x = D(x, 128, 3, 2)

    x = R(BN(x))
    x = D(x, 3, 7, 2)

    x = Lambda(lambda x: clip(x, -2.5, 2.5) + 0.001 * x)(x)

    return Model(input_img, x)
