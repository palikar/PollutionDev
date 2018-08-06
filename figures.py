#!/home/arnaud/anaconda3/bin/python3.6


import sys, os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.stats import norm
import properscoring as ps
import sklearn

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C





def draw_crps_intuition():
    plt.figure(figsize=(14,5), dpi=100)

        

    plt.subplot(1,2,1)
    plt.grid()

    obs=2.7
    mean=1
    std=2.5

    x_vals = np.linspace(-20, 20, 100)
    y_vals = norm.pdf(x_vals, mean, std)

    plt.plot(x_vals, y_vals,'r-', lw=5, alpha=0.6, label='Density of forecast')
    plt.plot([obs, obs], [0, 0.17], "k-", linewidth=3.5)
    plt.text(obs + 0.02, 0.18, "Obseravtion", fontsize=14)
    
    plt.xlim(-20,20)
    plt.ylim(0, 0.35)
    
    plt.legend()
    plt.title("Observation and forecast density")
    plt.xlabel("Density")
    plt.ylabel("Value")


    plt.subplot(1,2,2)
    plt.grid()
    

    x_vals = np.linspace(-20,obs,100)
    plt.fill_between(x_vals, np.repeat(0, 100), norm.cdf(x_vals, mean, std), color="blue", alpha=0.7, label="CRPS")
    x_vals = np.linspace(obs,20,100)
    plt.fill_between(x_vals, np.repeat(1, 100), norm.cdf(x_vals, mean, std), color="blue", alpha=0.7)
    x_vals = np.linspace(-20,20,100)
    y_vals = norm.cdf(x_vals, mean, std)

    plt.plot(x_vals, y_vals,'r-', lw=5, alpha=0.6, label='CDF of the forecast')
    plt.plot([obs, obs], [0, 1], "k-", linewidth=3.5, label='Observation')

    
    plt.legend()

    plt.xlim(-10,10)
    plt.ylim(0, 1.)
    plt.title("Observation and forecast CDF")
    plt.xlabel("CDF")
    plt.ylabel("Value")



    plt.show()





def draw_stochastic_regression():

    plt.figure(figsize=(14,5), dpi=100)

    x_vals = np.linspace(-3, 3, 5)
    y_vals = np.sin(x_vals)


    plt.plot(np.linspace(-10, 10, 200), np.sin(np.linspace(-10, 10, 200)) ,'b-', linewidth=1.5, label='True sin')
    plt.plot(x_vals, y_vals,'bo', markersize=15, label='Observations of sin')

    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)

    gp.fit(x_vals.reshape(-1, 1), y_vals)


    x_vals=np.linspace(-10, 10, 200)
    y_pred, sigma = gp.predict(x_vals.reshape(-1, 1), return_std=True)


    plt.plot(x_vals, y_pred, 'r-', lw=3, label="Forecast mean")


    plt.fill_between(x_vals,
                     y_pred + 0 * sigma,
                     y_pred - 0 * sigma,
                         color="red", alpha=0.5, label="Density")
    for i in np.linspace(0.0, 2.0, 20):
        plt.fill_between(x_vals,
                     y_pred + i * sigma,
                     y_pred - i * sigma,
                         color="red", alpha=0.05)


    plt.xlim(-7,7)
    plt.ylim(-1.5, 1.5)

    plt.grid()
    plt.title("Observation and forecast CDF")
    plt.legend()
    plt.xlabel("Input data")
    plt.ylabel("Output data")



    plt.show()

    
    

def draw_distribution_for_observation():

    def f(t):
        return np.exp(-t) * np.cos(2*np.pi*t)

    plt.figure(figsize=(14,5), dpi=150)

    samples = 100
    x_vals = np.linspace(-1, 4.7, samples)
    y_vals = f(x_vals) + np.random.normal(0, 0.2, samples)
    plt.plot(x_vals, y_vals,'b.', markersize=7, label='Data samples')


    plt.text(3.4, -1.5, "A prediction is made here", fontsize=12)
    plt.plot([4.8, 4.8], [-2,10],'k-', lw=0.7)


    draws_cnt=800
    draws = np.random.normal(0, 0.55, draws_cnt)
    plt.plot(np.repeat(4.8, draws_cnt), draws,'ro', markersize=5, alpha=0.01)
    
    plt.text(1.4, 2.5, "Forecast in form of samples from some distributions", fontsize=11, color="red")
    plt.arrow(2.5, 2.3, 4.8-2.65, 0-2.0, color="red",  head_width=0.1)    

    plt.ylim(-2, 5)
    # plt.grid()
    plt.title("Probabilistic regression")
    plt.legend()
    plt.grid()
    plt.xlabel("Input data")
    plt.ylabel("Output data")



    plt.show()



def main():
    draw_crps_intuition()
    draw_stochastic_regression()
    draw_distribution_for_observation()







if __name__ == '__main__':
    main()
