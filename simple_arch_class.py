from blocks import C, D, R, BN, C11, RES, parameters, reg
from keras.layers import Input, Lambda, Dense, Activation, Flatten
from keras.models import Model
from keras.backend import clip

def get_model(args):
    global reg
    reg = args.reg
    
    input_img = Input((3,32,32))
    x = input_img

    x = R(BN(x))
    x = C(x, 128, 7, 2)
    x = RES(x,128,16)
    x = RES(x,128,16)
    x = RES(x,128,16)
    x = RES(x,128,16)
    x = RES(x,128,16)
    x = RES(x,128,16)
    x = RES(x,128,16)
    x = RES(x,128,16)
    
    x = R(BN(x))
    x = C(x, 256, 3, 2)
    x = RES(x,256,16)
    x = RES(x,256,16)
    x = RES(x,256,16)
    x = RES(x,256,16)
    x = RES(x,256,16)
    x = RES(x,256,16)
    x = RES(x,256,16)
    x = RES(x,256,16)

    x = R(BN(x))
    x = C(x, 512, 3, 2)
    x = RES(x, 512,16)
    x = RES(x, 512,16)
    x = RES(x, 512,16)
    x = RES(x, 512,16)
    x = RES(x, 512,16)
    x = RES(x, 512,16)
    x = RES(x, 512,16)
    x = RES(x, 512,16)

    x = R(BN(x))
    x = C(x, 1028, 3, 2)

    x = R(BN(x))
    x = C(x, 2048, 3, 2)

    x= R(BN(x))
    x = Flatten()(x)
    x = Dense(10)(x)
    x = Activation('softmax')(x)
    return Model(input_img, x)
