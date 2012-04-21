#!/usr/bin/env python
import os, sys
from toy_systems.propagators import EDWProp
from msmbuilder import arglib, Project

parser = arglib.ArgumentParser(description='Create toy data: EDWProp')
parser.add_argument('k', description='Steepness f=0.5*k*x^2 in the harmonic directions',
    default=1, type=float)
parser.add_argument('dims', description='number of dimensions', default=2, type=int)
parser.add_argument('timesteps', description='number of timesteps',
    default=10000, type=int)
parser.add_argument('num_trajectories', description='number of trajectories',
    default=1, type=int)
parser.add_argument('outdir')
args = parser.parse_args()
print args
arglib.die_if_path_exists(args.outdir)
os.mkdir(args.outdir)

trj_dir = os.path.abspath(os.path.join(args.outdir, 'Trajectories'))
os.mkdir(trj_dir)

for i in range(args.num_trajectories):
    prop = EDWProp(args.dims, args.k)
    prop.run(args.timesteps)
    traj = prop.trajectory
    traj.SaveToHDF(os.path.join(trj_dir, 'trj{0}.h5'.format(i)))

pdbfn = os.path.abspath(os.path.join(args.outdir, 'conf.pdb'))
with open(pdbfn, 'w') as f:
    print >> f, "ATOM      1 1HH3 ACE     1       0.000  0.0   0.0\n"

project = Project.CreateProjectFromDir(Filename=None, TrajFilePath=trj_dir,
                ConfFilename=pdbfn)
os.rmdir('./Data')
project.SaveToHDF(os.path.join(args.outdir, 'ProjectInfo.h5'))
print 'Done'