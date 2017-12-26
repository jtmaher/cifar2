import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl


class Visualizer(object):
    def __init__(self, path, model, data, nr=20, nc=10):
        self.path = path
        self.model = model
        self.data = data
        self.nr = nr
        self.nc = nc
        self.ii = np.random.randint(data.test_n_x.shape[0], size=(nr,nc))

    def visualize(self, epoch=0, logs=None):
        nr = self.nr
        nc = self.nc
        ii = self.ii
        data = self.data
        
        fig = pl.figure(figsize=(5,5), dpi=200)
        pl.axis('off')
        preds=np.reshape(
            self.model.predict(data.test_n_x[np.reshape(ii, nr*nc),:,:,:]), (nr, nc, 3, 32, 32))
        preds=np.clip(data.denorm(preds),0,1)

        cols = []
        for i in range(nc):
            cols.append(np.concatenate(data.test_x[ii[:,i]], axis=1))
            cols.append(np.concatenate(preds[:,i], axis=1))
            
        pl.imshow(np.transpose(np.concatenate(cols,axis=2), (1,2,0)))
        pl.savefig('%s/%s_%d.png' % (self.path, self.model.name, epoch))
        pl.close(fig)

