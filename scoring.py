#!/home/arnaud/anaconda3/bin/python3.6


import sys, os
import pandas as pd
from matplotlib import pyplot as plt
import math
from scipy.stats import norm
from scipy.stats import gaussian_kde
import properscoring as ps


import numpy as np


def nrd(x):
    r = np.percentile(x, [0,25, 0.75])
    h = (r[1] - r[0])/1.34
    return 4 * 1.06 * min(math.sqrt(np.var(x)), h) * (x.shape[0])**(-1/5)

def crps_edf_samples(y, data):
    fun = lambda s: ps.crps_ensemble(s, data)
    vfun = np.vectorize(fun)
    return vfun(y)

        
def dss_edf_samples(y, data):
    m = data.mean()
    v = (data**2).mean() - m**2
    fun = lambda s: (((s - m)**2) / v) + 2*math.log(v)
    vfun = np.vectorize(fun)
    return vfun(y) 

def dss_norm(y, loc=0, scale=1):
    yy = y
    if loc != 1:
        yy = yy - loc
    if scale == 1:
        return yy**2
    else:
       return (y/scale) + 2*np.log(scale)
