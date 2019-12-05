import os
import tensorflow as tf
import numpy as np
import sys
from os import listdir
from os.path import isfile, join

class dataloader:
    
    def __init__(self,
                 proteins, # list of proteins to load
                 datadir="/net/scratch/hiranumn/data_ver4/", # Base directory for all protein data
                 lengthmax=500, # Limit to the length of proteins, if bigger we ignore.
                 load_dtype=np.float32, # Data type to load with
                 digitization1 = [-20.0, -15.0, -10.0, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, 10.0, 15.0, 20.0],
                 digitization2 = np.arange(-20.5,20.6,1.0),
                 features=["obt", "prop", "dist"], # Kinds of features to incooperate.
                 verbose=False,
                 distribution=False,
                 include_native=True,
                 include_native_dist=True,
                ):
        
        self.n = {}
        self.samples_dict = {}
        self.sizes = {}
        
        self.load_dtype = load_dtype
        self.digitization1 = digitization1
        self.digitization2 = digitization2
        self.datadir = datadir
        self.features = features
        self.verbose = verbose
        self.distribution = distribution
        self.include_native = include_native
        self.include_native_dist = include_native_dist
        if self.verbose: print("features:", self.features)
            
        # Loading file availability
        temp = []
        for p in proteins:
            path = datadir+p+"/"
            samples_files = [f[:-4] for f in listdir(path) if isfile(join(path, f)) and "npz" in f]
            # Removing native from distribution if you are using distribution option
            if not self.include_native:
                samples_files = [s for s in samples_files if s != "native"]
            np.random.shuffle(samples_files)
            samples = []
            for s in samples_files:
                conds = []
                conds.append(isfile(path+s+".npz"))
                conds.append(isfile(path+s+".lddt.csv"))
                if np.all(conds): samples.append(s)
                    
            if len(samples) > 0:
                length = np.load(path+samples[0]+".npz")["tbt"].shape[-1]
                if length < lengthmax:
                    temp.append(p)
                    self.samples_dict[p] = samples
                    self.n[p] = len(samples)
                    self.sizes[p] = length
                
        # Make a list of proteins
        self.proteins = temp
        self.index = np.arange(len(self.proteins))
        np.random.shuffle(self.index)
        self.cur_index = 0

    def next(self, transform=True, pindex=-1):
        pname = self.proteins[self.index[self.cur_index]]
        if pindex == -1:
            pindex = np.random.choice(np.arange(self.n[pname]))
        sample = self.samples_dict[pname][pindex]
        psize = self.sizes[pname]
        data = np.load(self.datadir+pname+"/"+sample+".npz")
        
        # 3D information
        idx = data["idx"]
        val = data["val"]
        
        # 1D information
        angles = np.stack([np.sin(data["phi"]), np.cos(data["phi"]), np.sin(data["psi"]), np.cos(data["psi"])], axis=-1)
        obt = data["obt"].T
        prop = data["prop"].T
        
        # 2D information
        orientations = np.stack([data["omega6d"], data["theta6d"], data["phi6d"]], axis=-1)
        orientations = np.concatenate([np.sin(orientations), np.cos(orientations)], axis=-1)
        euler = np.concatenate([np.sin(data["euler"]), np.cos(data["euler"])], axis=-1)
        maps = data["maps"]
        tbt = data["tbt"].T
        sep = seqsep(psize)
        
        if self.include_native_dist:
            dist = np.load(self.datadir+pname+"/dist.npy")
        else:
            dist = np.load(self.datadir+pname+"/dist2.npy")
        
        # Get target
        lddt = np.genfromtxt(self.datadir+pname+"/"+sample+".lddt.csv", skip_header=11)[:,4]
        native = np.load(self.datadir+pname+"/native.npz")["tbt"][0]
        estogram = get_estogram((tbt[:,:,0], native), self.digitization1)
        
        # Transform input distance
        if transform:
            tbt[:,:,0] = f(tbt[:,:,0])
            maps = f(maps)
        
        self.cur_index += 1
        if self.cur_index == len(self.proteins):        
            self.cur_index = 0 
            np.random.shuffle(self.index)
            
        if self.verbose:
            print(angles.shape, obt.shape, prop.shape)
            print(tbt.shape, maps.shape, euler.shape, orientations.shape, sep.shape)
            
        if self.distribution:
            return (idx, val),\
                    np.concatenate([angles, obt, prop], axis=-1),\
                    np.concatenate([tbt, maps, euler, orientations, sep, dist], axis=-1),\
                    (lddt, estogram, native)
        else:
            return (idx, val),\
                    np.concatenate([angles, obt, prop], axis=-1),\
                    np.concatenate([tbt, maps, euler, orientations, sep], axis=-1),\
                    (lddt, estogram, native)
        
def f(X, cutoff=6, scaling=3.0):
    X_prime = np.maximum(X, np.zeros_like(X) + cutoff) - cutoff
    return np.arcsinh(X_prime)/scaling

def get_estogram(XY, digitization):
    (X,Y) = XY
    residual = X-Y
    estogram = np.eye(len(digitization)+1)[np.digitize(residual, digitization)]
    return estogram

# Sequence separtion features
def seqsep(psize, normalizer=100, axis=-1):
    ret = np.ones((psize, psize))
    for i in range(psize):
        for j in range(psize):
            ret[i,j] = abs(i-j)*1.0/100-1.0
    return np.expand_dims(ret, axis)

# Getting masks
def getMask(exclude):
    feature2D = [("distance",1), ("rosetta",9), ("distance",4), ("orientation",18), ("seqsep",1)]
    feature1D = [("angles",10), ("rosetta",4), ("ss",4), ("aa", 52)]
    for e in exclude:
        if e not in [i[0] for i in feature2D] and e not in [i[0] for i in feature1D]:
            print("Feature names do not exist.")
            print([i[0] for i in feature1D])
            print([i[0] for i in feature2D])
            return -1
    mask = []
    temp = []
    index = 0
    for f in feature1D:
        for i in range(f[1]):
            if f[0] in exclude: temp.append(index)
            index+=1
    mask.append(temp)
    temp = []
    index = 0
    for f in feature2D:
        for i in range(f[1]):
            if f[0] in exclude: temp.append(index)
            index+=1
    mask.append(temp)
    return mask