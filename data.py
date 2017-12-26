import numpy as np

from keras import backend as K
if K.backend()=='tensorflow':
        K.set_image_dim_ordering("th")

class AutoFlow(object):
    def __init__(self, flow):
        self.flow = flow

    def __next__(self):
        nx = self.flow.next()
        return (nx, nx)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.flow)


        
class CIFAR10(object):
    def __init__(self):
        from keras.datasets import cifar10
        from keras.utils.np_utils import to_categorical
        
        (train_x, train_y), (test_x, test_y) = cifar10.load_data()
        self.train_x =  train_x.astype('float32')/255
        self.train_y = to_categorical(train_y, 10)
        self.test_x = test_x.astype('float32')/255
        self.test_y = to_categorical(test_y, 10)

        def N(x):
                return (x - np.mean(x,axis=0)) /np.std(x,axis=0)

        self.train_mean = np.mean(train_x,axis=0)
        self.train_sd = np.std(train_x,axis=0)
        
        self.train_n_x = N(self.train_x)
        self.test_n_x = (self.test_x - np.mean(self.train_x, axis=0) ) / np.std(self.train_x,axis=0)
        
        
    def denorm(self, x):
        return x * np.std(self.train_x,axis=0) + np.mean(self.train_x, axis=0)


    def datagen_flow(self, batch_size, use_class=True):
        from keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(self.train_x)

        if use_class:
                return datagen.flow(self.train_n_x, self.train_y, batch_size=batch_size)
        else:
                return datagen.flow(self.train_n_x, batch_size=batch_size)
