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

    
    nrow = y.shape[0]
    ls = np.zeros(shape=nrow)
    W = 0.0
    for i in range(0, n):
        W = W + w[i]
        for j in range(0, nrow):
            ls[j] = ls[j] + w[i] * norm.pdf(y[j], m[i], s[i])

    if W == 0:
        W = 0.00000000001
    if 0 in ls:
        ls =  ls + 0.0000000001
    
    return np.log(W) - np.log(ls) 




def main():
    print("Starting")

    fig, ax = plt.subplots(1, 1)
    
    r = norm.rvs(size=int(sys.argv[1]), loc=0, scale=5)

    print("Mean: 0 , STD: 5")
    y = np.array([0.1, 20])
    
    # print("Data" + str(y))
    # scores = dss_edf_samples(y, r)
    # print("DSS: " + str(scores))
    # scores = log_edf_samples(y, r)
    # print("LOG: " + str(scores))
    # score = crps_edf_samples(y, r)
    # print("CRPS: " + str(score))



    scores = np.empty(3, dtype=object)
    scores[0] = np.array([])
    scores[1] = np.array([])
    scores[2] = np.array([])
    for i in range(2, 1002, 20):
        r = norm.rvs(size=i, loc=0, scale=5)
        scores[0] = np.append(scores[0], dss_edf_samples(y, r)[0])
        scores[1] = np.append(scores[1], log_edf_samples(y, r)[0])
        scores[2] = np.append(scores[2], crps_edf_samples(y, r)[0])
    

    # plt.figure(figsize=(15,13), dpi=100)
    
    plt.subplot(2,1,1)
    plt.plot(range(1,1000,20), scores[0], "g", linewidth = 1.3, label="DSS")
    plt.plot(range(1,1000,20), scores[1], "r-", linewidth = 1.3, label="LOG")
    plt.plot(range(1,1000,20), scores[2], "b--", linewidth = 1, label="CRPS")
    plt.legend()
    plt.title('Scoring rules for observation ' + str(y[0]) + " and Normal distribution with Mean: 0 , STD: 5")
    plt.xlabel("Number of samples")
    plt.ylabel("Rule score")

    scores = np.empty(3, dtype=object)
    scores[0] = np.array([])
    scores[1] = np.array([])
    scores[2] = np.array([])
    for i in range(2, 1002, 20):
        r = norm.rvs(size=i, loc=0, scale=5)
        scores[0] = np.append(scores[0], dss_edf_samples(y, r)[1])
        scores[1] = np.append(scores[1], log_edf_samples(y, r)[1])
        scores[2] = np.append(scores[2], crps_edf_samples(y, r)[1])
        


    plt.subplot(2,1,2)
    plt.plot(range(1,1000,20), scores[0], "g", linewidth = 1.3, label="DSS")
    plt.plot(range(1,1000,20), scores[1], "r-", linewidth = 1.3, label="LOG")
    plt.plot(range(1,1000,20), scores[2], "b--", linewidth = 1, label="CRPS")
    plt.legend()
    plt.title('Scoring rules for observation ' + str(y[1]) + " and Normal distribution with Mean: 0 , STD: 5")
    plt.xlabel("Number of samples")
    plt.ylabel("Rule score")

    plt.show()


if __name__ == '__main__':
    main()
