from msmbuilder.metrics import Vectorized
import numpy as np

class CartesianMetric(Vectorized):
    def __init__(self, dimension=None):
        self.metric = 'euclidean'
        self.p = None
        self._dim = dimension
    
    def prepare_trajectory(self, trajectory):
        '''Prepare trajectory for distance computations on the *dimension*th
        diension'''
        if self._dim is None:
            out = np.double(trajectory['XYZList'])
        else:
            if not self._dim in range(0, trajectory['XYZList'].shape[1]):
                raise ValueError('Dimension mismatch')
            out = np.double(trajectory['XYZList'][:, [self._dim]])
        
        if not out.flags.contiguous:
            out = out.copy()
        return out