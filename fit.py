import argparse
from data import CIFAR10
from os import mkdir
from shutil import copy
from keras.optimizers import Adam
import numpy as np
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, LambdaCallback
from keras.utils import plot_model

from visualizer import Visualizer
from data import AutoFlow


parser = argparse.ArgumentParser(description='Fit a model')

parser.add_argument('--output-dir', dest='output_dir')
parser.add_argument('--arch-file', dest='arch_file')
parser.add_argument('--reg', dest='reg', default=1e-4, type=float)
parser.add_argument('--lr', dest='lr', default=1e-4, type=float)
parser.add_argument('--batch-size', dest='batch_size', default=128, type=int)
parser.add_argument('--num_epochs', dest='num_epochs', default=250, type=int)
parser.add_argument('--verbose', dest='verbose', default=1, type=int)

args = parser.parse_args()

arch = __import__(args.arch_file)

mkdir(args.output_dir)

with open('%s/args' % args.output_dir, 'w') as f:
    f.write(str(vars(args)) + '\n')


copy(args.arch_file + '.py', args.output_dir)

model = arch.get_model(args)

model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=args.lr))

c10 = CIFAR10()

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=1e-7)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger(args.output_dir + '/train.csv', append=False)
vis = Visualizer(args.output_dir, model, c10)
visuals = LambdaCallback(on_epoch_end=lambda epoch, logs: vis.visualize(epoch,logs))


plot_model(model, to_file='%s/model.png' % args.output_dir, show_shapes=True)


model.fit_generator(
    AutoFlow(c10.datagen_flow(args.batch_size, use_class=False)),
    steps_per_epoch=c10.train_x.shape[0] // args.batch_size,
    validation_data=(c10.test_n_x, c10.test_n_x),
    epochs=args.num_epochs,
    verbose=args.verbose,
    max_queue_size=100,
    callbacks=[visuals, lr_reducer, early_stopper, csv_logger])

model.save( '%s/model.hdf5' % args.output_dir)
