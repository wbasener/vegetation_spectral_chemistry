#import spcdal
import os
import spectral
from spectral import envi
from spectral import resampling
import numpy as np
import matplotlib.pyplot as plt
import lazypredict
import numpy as np
import pandas as pd
import math
import scipy.stats as stats
import itertools
import matplotlib.pyplot as plt

def getXY(lib, chem):
    # gets the X-values and y-values from a spectral library for the given chemical
    X1 = []
    y1 = []
    name1 = []

    for i,name in enumerate(lib.names):
        if lib.metadata['chemistry_'+chem][i] != '-1':
            X1.append(lib.spectra[i])
            y1.append(float(lib.metadata['chemistry_water'][i]))
            name1.append(name)
    X1 = np.array(X1)
    y1 = np.array(y1)   
    name1 = np.array(name1)    
    
    X1_n = np.zeros(X1.shape)
    # Normalize by subtracting the min of each spectrum and dividing by its L2 norm
    for i in range(X1_n.shape[0]):
        X1_n[i,:] = (X1[i,:] - np.min(X1[i,:]))/np.linalg.norm((X1[i,:] - np.min(X1[i,:])))
    
    m = np.mean(X1_n, axis=0)
    sd = np.std(X1_n, axis=0)  
    for i in range(X1_n.shape[0]):
        X1_n[i,:] = (X1_n[i,:] - m)/sd
    
    return [X1, X1_n, y1, name1]