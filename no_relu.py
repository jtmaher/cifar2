from blocks import C, D, R, BN, C11, SCL, SCRES, parameters, reg
from keras.layers import Input, Lambda
from keras.models import Model


def get_model(args):
    global reg
    reg = args.reg

    input_img = Input((3,32,32))
    x = input_img

    x = BN(x)
    
    x = C(x, 128, 7, 2)
    x = BN(x)

    x = C(x, 128, 3, 2)
    x = BN(x)

    x = C(x, 128, 3, 2)
    x = BN(x)

    x = C(x, 128, 3, 2)
    x = BN(x)

    x = C(x, 128, 3, 2)
    x = BN(x)

    x = D(x, 128, 3, 2)
    x = BN(x)
    
    x = D(x, 128, 3, 2)
    x = BN(x)
        
    x = D(x, 128, 3, 2)
    x = BN(x)

    x = D(x, 128, 3, 2)
    x = BN(x)

    x = D(x, 3, 7, 2)
    x = BN(x)

    return Model(input_img, x)
