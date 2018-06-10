#!/home/arnaud/anaconda3/bin/python3.6



import sys, os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.stats import norm
import properscoring as ps


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

    
    
def log_edf_samples(y, m):
    bw = nrd(m)
    n = m.shape[0]
    w = np.repeat(1.0/n, n)
    s = np.repeat(bw, n)

    #lsmixnC
    nrow = y.shape[0]
    ls = np.zeros(shape=nrow)
    W = 0.0
    for i in range(0, n):
        W = W + w[i]
        for j in range(0, nrow):
            ls[j] = ls[j] + w[i] * norm.pdf(y[j], m[i], s[i])

    return np.log(W) - np.log(ls) 








def main():
    print("Starting")

    fig, ax = plt.subplots(1, 1)
    
    r = norm.rvs(size=int(sys.argv[1]), loc=0, scale=5)

    print("Mean: 0 , STD: 5")
    y = np.array([0.1, 0.2, -0.1, 0, 10,100,50,25])
    
    print("Data" + str(y))
    scores = dss_edf_samples(y, r)
    print("DSS: " + str(scores))
    scores = log_edf_samples(y, r)
    print("LOG: " + str(scores))
    score = crps_edf_samples(y)
    print("CRPS: " + str(score))
    

    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2, bins=100)
    ax.legend(loc='best', frameon=False)
    # plt.show()


if __name__ == '__main__':
    main()
