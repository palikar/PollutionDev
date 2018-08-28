#!/home/arnaud/anaconda3/bin/python3.6
import sys, os
import pandas as pd
from matplotlib import pyplot as plt
import math
from scipy.stats import norm
from scipy.stats import gaussian_kde
import properscoring as ps

import numpy as np

        
def dss_edf_samples(y, data):
    """A functions calculating the Dawid Sebastiani score over some
observations given samples drawn from arbitrary distribution.
    y: the vector of realised obseravtions on which the scores are calculated
    data: samples drawn from some distribution that models the observations
    """
    m = data.mean()
    v = (data**2).mean() - m**2
    fun = lambda s: (((s - m)**2) / v) + 2*math.log(v)
    vfun = np.vectorize(fun)
    return vfun(y) 

def dss_norm(y, loc=0, scale=1):
    """A functions calculating the Dawid Sebastiani score over some
observations given some normal distribution.
    y: the vector of realised obseravtions on which the scores are calculated
    loc: the mean of the normal distribution modeling the data
    scale: the variance of the normal distribution modeling the data
    """
    yy = y
    if loc != 1:
        yy = yy - loc
    if scale == 1:
        return yy**2
    else:
       return (y/scale) + 2*np.log(scale)
