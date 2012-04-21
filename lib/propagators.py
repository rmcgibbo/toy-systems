import sys, os
from msmbuilder import Trajectory
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np


class Propagator(object):
    """Interface for Propagators"""
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def __init__(self):
        '''Initiaize the propagator'''
        pass
    
    @abstractmethod
    def run(self, num_timesteps):
        '''Run the propagator for the specified number of timesteps'''
        pass
    
    @abstractproperty
    def trajectory(self):
        '''Return the trajectory. This is the only method on this class that will
        actually get used'''
        try:
            return self._trajectory
        except:
            raise


class RandomWalker(Propagator):
    "Takes random walk with normally distributed step lengths"
    def __init__(self, num_dims, diffusion_const, xRange, dt=1):
        self.num_dims = num_dims
        if isinstance(diffusion_const, list):
            diffusion_const = np.array(diffusion_const)
        
        if callable(diffusion_const):
            self.diffusion_const = diffusion_const
        elif (np.isscalar(diffusion_const) and num_dims == 1) or (len(diffusion_const) == num_dims):
            self.diffusion_const = lambda x: diffusion_const
        else:
            raise ValueError("Diffusion const should be vector of length *num_dims* or callable")
        
        self.xRange = np.array(xRange)
        if not self.xRange.shape == (num_dims, 2):
            raise ValueError("shape should be %s, found %s" % (str((num_dims,2)), xRange.shape))
        self.dt = dt
    
    def run(self, num_timesteps):
        r = np.random.randn(self.num_dims, num_timesteps)
        x = np.zeros((self.num_dims, num_timesteps))
        for i in xrange(1,num_timesteps):
            x[:,i] = x[:,i-1] + np.sqrt(self.diffusion_const(x[:,i-1]) / self.dt) * r[:,i-1] * self.dt
            if (np.any(x[:,i] > self.xRange[:,1]) or (np.any(x[:,i] < self.xRange[:,0]))):
                print >> sys.stderr, 'Broke early at %d of %d' % (i, num_timesteps)
                break
        x = x[:, 0:i]
        self.positions = x.transpose()
        return self
    
    @property
    def trajectory(self):
        empty = np.array([0,0])
        trajectory = {'XYZList': self.positions, 'ChainID': empty, 'AtomNames': empty,
                      'ResidueNames':empty, 'AtomID': empty, 'ResidueID': empty}
        trajectory = Trajectory(trajectory)
        return trajectory


class LDPropagator(Propagator):
    "Langevin Dynamics"
    def __init__(self, n_dims, dV, xRange, kT, dt=0.001, gamma=1):
        self.ldi = LangevinDynamicsIntegrator(dV=dV, dt=dt, gamma=gamma, kT=kT, xRange=xRange)
        self.ldi.start([0.0] * n_dims)
    def run(self, num_timesteps):
        self.ldi.run(num_timesteps)
        return self
    @property
    def trajectory(self):
        empty = np.array([0,0])
        trajectory = {'XYZList': self.ldi.positions, 'ChainID': empty, 'AtomNames': empty,
                      'ResidueNames':empty, 'AtomID': empty, 'ResidueID': empty}
        trajectory = Trajectory(trajectory)
        return trajectory
    

class EDWProp(LDPropagator):
    '''Propagator on a PES that's a double well potential in 1 dimension, but
    just a harmonic in the other degrees'''
    def __init__(self, n_dims, k=1):
        def dV(x):
            if not len(x) == n_dims: raise ValueError("Wrong dims")
            result = k * x
            result[0] = (x[0] - 1.0) * x[0] * (x[0] + 1.0)
            return result
        xRange = np.array([[-5,5]] * n_dims)
        kT = 0.25
        super(EDWProp, self).__init__(n_dims, dV, xRange, kT)
    

class MHProp(LDPropagator):
    """2D propagator thats a double well in each dimension but with different heights"""
    def __init__(self, barrier1, barrier2):
        def dV(x):
            if not len(x) == 2: raise ValueError('Wrong dims')
            return np.array([self.barrier1 * (4*x[0]**3 - 4*x[0]),
                    self.barrier2 * (4*x[1]**3 - 4*x[1])])
        range_const = np.sqrt(1 + np.sqrt(11))
        xRange = np.array([[-self.barrier1 * range_const, self.barrier1 * range_const],
                           [-self.barrier2 * range_const, self.barrier2 * range_const]])
        kT = 1.0
        super(MHProp, self).__init__(2, dV, xRange, kT)
    

class MWProp(LDPropagator):
    """2D propagator that has different widths between the minima in different directions
    but the same height (1.0 i think)"""
    def __init__(self, width1, width2):
        def dV(x):
            if not len(x) == 2: raise ValueError('Wrong dims')
            return np.array([-(4*x[0]*(self.width1**2 - x[0]**2))/self.width1**4,
                             -(4*x[0]*(self.width2**2 - x[1]**2))/self.width2**4])
        range_const = np.sqrt(1 + np.sqrt(11))
        xRange = np.array([[-self.width1 * range_const, self.width1 * range_const],
                            [-self.width2 * range_const, self.width2 * range_const]])
        kT = 1.0
        super(MWProp, self).__init__(2, dV, xRange, kT)

class FlatProp(LDPropagator):
    """2D propagator that is flat in both directions"""
    def __init__(self):
        def dV(x):
            return np.zeros_like(x)
        xRange = np.array([[-5,5], [-5,5]])
        kT = 1.0
        super(FlatProp, self).__init__(2, dV, xRange, kT)

class Rough1DProp(LDPropagator):
    def __init__(self, freq=100, amp=1):
        def dV(x):
            return (x - 1.0) * x * (x + 1.0) + amp * np.sin(freq * x)
        xRange = np.array([-5, 5])
        kT = 0.25
        super(Rough1D, self).__init__(1, dV, kT, xRange)
    

class SinFlatProp(LDPropagator):
    def __init__(self, freq):
        def dV(x):
            return np.array([freq * np.sin(freq * x[0]), 0])
        xRange = np.array([[-5,5], [-5,5]])
        kT = 1.0
        super(SinFlatProp, self).__init__(2, dV, xRange, kT)

class LangevinDynamicsIntegrator(object):
    
    def __init__(self, dV, dt=0.001, gamma=1, kT=0.25, xRange=np.array([0, 1])):
        '''Initialize the integrator.
        
        dV is the potential function. It should be a callable that takes
        a single argument and returns a single argument for a 1D problem
        or an array (the gradient) for a problem in higher dimensions
        
        xRange should be a 2xN array for an N dimensional problem.
        '''
        
        self.dV = dV
        self.gamma = gamma
        self.kT = kT
        self.positions = np.array([])
        self.randCnst = np.sqrt(2 * kT * gamma)
        self.dt = dt
        self.xRange = xRange #np.array(xRange)
        
        if len(self.xRange.shape) == 1:
            self._dimension = 1
        elif len(self.xRange.shape) == 2:
            self._dimension = self.xRange.shape[0]
        else:
            raise ValueError("I'm confused")

        self.stdev = np.sqrt(self.dt)
    
    def start(self, x0):
        '''Start the integrator at position x0'''
        self.positions = np.array([x0])
        if self._dimension == 1:
            # this is a 1D potential
            if len(self.positions.shape) != 1:
                raise AttributeError('Dimension mismatch. xRange indicates a 1D problem, but you supplied a x0 that is %dD' \
                                     % len(self.positions.shape))
        else:
            if not len(self.xRange) == len(self.positions[0]):
                raise AttributeError('Dimension mismatch. xRange indicates you have a %dD problem, but you supplied an x0 that is %dD' \
                                        % (self.xRange.shape[1], self.positions.shape[1]))
    
    def run(self, n_steps):
        '''Integrate for n steps'''
        if self._dimension == 1:
            new_positions = np.zeros(n_steps)
        else:
            new_positions = np.zeros((n_steps, self._dimension))
        
        try:
            xi = self.positions[-1]
        except IndexError:
            raise RuntimeError('Did you forget to seed the integrator with start()?')
        
        for i in range(n_steps):
            if self._dimension == 1:
                randG = np.random.normal(0, self.stdev)
            else:
                randG = np.random.normal(0, self.stdev, (self._dimension,))
            xf = xi + (-self.dV(xi) * self.dt + self.randCnst * randG) / self.gamma
            
            # elastic collision
            if self._dimension == 1:
                if xf < self.xRange[0]:
                    xf = self.xRange[0] + (self.xRange[0] - xf)
                elif xf > self.xRange[1]:
                    xf = self.xRange[1] + (self.xRange[1] - xf)
            else:
                #print self.xRange
                for d in range(self._dimension):
                    if xf[d] < self.xRange[d, 0]:
                        xf[d] = self.xRange[d, 0] + (self.xRange[d, 0] - xf[d])
                    if xf[d] > self.xRange[d, 1]:
                        xf[d] = self.xRange[d, 1] + (self.xRange[d, 1] - xf[d])            
            new_positions[i] = xf
            xi = xf
        
        self.positions = np.concatenate((self.positions, new_positions))
