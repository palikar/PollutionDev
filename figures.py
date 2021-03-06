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
    plt.ylabel("Density")
    plt.xlabel("Value")


    plt.subplot(1,2,2)
    plt.grid()
    

    x_vals = np.linspace(-20,obs,100)
    plt.fill_between(x_vals, np.repeat(0, 100), norm.cdf(x_vals, mean, std), color="blue", alpha=0.7)
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
    plt.ylabel("CDF")
    plt.xlabel("Value")



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
                         color="red", alpha=0.5, label="Confidence region")
    for i in np.linspace(0.0, 2.0, 20):
        plt.fill_between(x_vals,
                     y_pred + i * sigma,
                     y_pred - i * sigma,
                         color="red", alpha=0.05)


    plt.xlim(-7,7)
    plt.ylim(-1.5, 1.5)

    plt.grid()
    plt.title("Probabilistic regression")
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
    
    plt.text(1.4, 2.5, "Forecast in form of samples from some distribution", fontsize=11, color="red")
    plt.arrow(2.5, 2.3, 4.8-2.65, 0-2.0, color="red",  head_width=0.1)    

    plt.ylim(-2, 5)
    # plt.grid()
    plt.title("Forecast distribution")
    plt.legend()
    plt.grid()
    plt.xlabel("Input data")
    plt.ylabel("Output data")



    plt.show()




def draw_rank_hist():
    plt.figure(figsize=(10,9), dpi=100)

     
    plt.subplot(3,2,1)
    plt.tight_layout()
    plt.grid()

    x = np.random.uniform(low=-1, high=5, size=15)
    plt.plot([x[0], x[0]], [0, 3], "b-", linewidth=1.3, label="Forecast Samples")
    for obs in x[1:]:
        plt.plot([x, x], [0, 3], "b-", linewidth=1.5)
        
    plt.plot([2, 2], [0, 4], "k-", linewidth=3.5, label="Observation")


    plt.xlim(-10,10)
    plt.ylim(0, 7)
    
    plt.legend()
    plt.title("Forecast samples and actual observation")
    plt.xlabel("Value")
    plt.ylabel("")
    
    plt.subplot(3,2,2)
    plt.tight_layout()

    x = np.random.uniform(low=0, high=6, size=1200)
    # x = np.random.normal(loc=2, scale=1, size=1000)
    bins = np.arange(7) - 0.5
    plt.hist(x, bins=bins, density=True, facecolor='lightblue', alpha=0.75, histtype='bar', edgecolor='black', linewidth=1.0, rwidth=0.9)

    # plt.xlim(0,2)
    plt.ylim(0, 0.6)
    plt.xticks(range(6))
    # plt.xlim([-1, 10])

    plt.grid(True)
    plt.title("Verification Rank Histogram")
    plt.xlabel("Rank")
    plt.ylabel("Fraction")

    plt.subplot(3,2,3)
    plt.tight_layout()
    plt.grid()

    x = np.random.uniform(low=1.5, high=2.5, size=15)
    plt.plot([x[0], x[0]], [0, 3], "b-", linewidth=1.3, label="Forecast Samples")
    for obs in x[1:]:
        plt.plot([x, x], [0, 3], "b-", linewidth=1.5)
        
    plt.plot([4, 4], [0, 3], "b-", linewidth=1.5)
    plt.plot([3.4, 3.4], [0, 3], "b-", linewidth=1.5)
    plt.plot([0, 0], [0, 3], "b-", linewidth=1.5)
    plt.plot([0.5, 0.5], [0, 3], "b-", linewidth=1.5)
    plt.plot([1, 1], [0, 3], "b-", linewidth=1.5)
    plt.plot([2, 2], [0, 4], "k-", linewidth=3.5, label="Observation")


    plt.xlim(-10,10)
    plt.ylim(0, 7)
    
    plt.legend()
    plt.title("Forecast samples and actual observation")
    plt.xlabel("Value")
    plt.ylabel("")
    
    plt.subplot(3,2,4)
    plt.tight_layout()

    x = np.random.normal(loc=2, scale=1, size=1000)
    bins = np.arange(7) - 0.5
    plt.hist(x, bins=bins, density=True, facecolor='lightblue', alpha=0.75, histtype='bar', edgecolor='black', linewidth=1.0, rwidth=0.9)

    # plt.xlim(0,2)
    plt.ylim(0, 0.6)
    plt.xticks(range(6))
    # plt.xlim([-1, 10])

    plt.grid(True)
    plt.title("Verification Rank Histogram")
    plt.xlabel("Rank")
    plt.ylabel("Fraction")

    

    plt.subplot(3,2,5)
    plt.tight_layout()
    plt.grid()

    x = np.random.uniform(low=2.5, high=4.5, size=15)
    plt.plot([x[0], x[0]], [0, 3], "b-", linewidth=1.3, label="Forecast Samples")
    for obs in x[1:]:
        plt.plot([x, x], [0, 3], "b-", linewidth=1.5)

    plt.plot([1.5, 1.5], [0, 3], "b-", linewidth=1.5)
    plt.plot([1.7, 1.7], [0, 3], "b-", linewidth=1.5)
    plt.plot([2.1, 2.1], [0, 3], "b-", linewidth=1.5)
    plt.plot([2, 2], [0, 4], "k-", linewidth=3.5, label="Observation")


    plt.xlim(-10,10)
    plt.ylim(0, 7)
    
    plt.legend()
    plt.title("Forecast samples and actual observation")
    plt.xlabel("Value")
    plt.ylabel("")
    
    plt.subplot(3,2,6)
    plt.tight_layout()


    x = np.hstack([
                   np.random.normal(loc=0.2, scale=0.9, size=2000)])
    bins = np.arange(7) - 0.5
    plt.hist(x, bins=bins, density=True, facecolor='lightblue', alpha=0.75, histtype='bar', edgecolor='black', linewidth=1.0, rwidth=0.9)

    # plt.xlim(0,2)
    plt.ylim(0, 0.9)
    plt.xticks(range(6))
    # plt.xlim([-1, 10])

    plt.grid(True)
    plt.title("Verification Rank Histogram")
    plt.xlabel("Rank")
    plt.ylabel("Fraction")


    plt.show()


def draw_point_vs_dens():
    plt.figure(figsize=(14,5), dpi=100)     
    plt.tight_layout()

    obs_x = 2.9
    mean = 2
    std = 1.3
    ylim = 0.4

    plt.subplot2grid((2, 3), (0, 0), rowspan=2)
    # plt.subplot(2, 3, 1, rowspan=2)
    plt.tight_layout()
    plt.plot([obs_x, obs_x],[0, 0.3],'b-', lw=2, alpha=1, label='Obseravtion')
    plt.xlim(-10,10)
    plt.ylim(0, ylim)
    plt.yticks([0.0])
    plt.grid()
    plt.legend()
    plt.title("Realised observation")
    plt.xlabel("Value")
    plt.ylabel("")

    plt.subplot(2,3,2)
    plt.tight_layout()
    y_vals = x_vals = np.linspace(-20, 20, 100)
    y_vals = norm.pdf(x_vals, mean, std)
    plt.plot(x_vals, y_vals,'r-', lw=2, alpha=1, label='Density of forecast')
    plt.fill_between(x_vals, np.repeat(0, y_vals.shape[0] ), y_vals, color='red', alpha=0.2)
    plt.xlim(-10,10)
    plt.ylim(0, ylim)
    plt.grid()
    plt.legend()
    plt.title("Predicted forecast")
    plt.xlabel("Value")
    plt.ylabel("PDF")

    plt.subplot(2,3,3)
    plt.tight_layout()
    y_vals = x_vals = np.linspace(-20, 20, 100)
    y_vals = norm.pdf(x_vals, mean, std)
    plt.plot(x_vals, y_vals,'r-', lw=2, alpha=1, label='Density of forecast')
    plt.plot([obs_x, obs_x],[0, 0.3],'b-', lw=2, alpha=1, label='Obseravtion')
    plt.fill_between(x_vals, np.repeat(0, y_vals.shape[0] ), y_vals, color='red', alpha=0.2)
    plt.xlim(-10,10)
    plt.ylim(0, ylim)
    plt.grid()
    plt.legend()
    plt.title("Comparasion between obsrvation and forecast")
    plt.xlabel("Value")
    plt.ylabel("PDF")
    

    plt.subplot(2, 3, 5)
    plt.tight_layout()
    plt.plot([1.9, 1.9],[0, 0.3],'r-', lw=2, alpha=1, label='Point estimate')
    plt.xlim(-10,10)
    plt.ylim(0, ylim)
    plt.yticks([0.0])
    plt.grid()
    plt.legend()
    plt.title("Predicted point estimate")
    plt.xlabel("Value")
    plt.ylabel("")

    
    plt.subplot(2,3, 6)
    plt.tight_layout()
    plt.plot([1.9, 1.9],[0, 0.3],'r-', lw=2, alpha=1, label='Point estimate')
    plt.plot([obs_x, obs_x],[0, 0.3],'b-', lw=2, alpha=1, label='Obseravtion')
    plt.xlim(-10,10)
    plt.ylim(0, ylim)
    plt.yticks([0.0])
    plt.grid()
    plt.legend()
    plt.title("Comparasion between obsrvation and the point esitmate")
    plt.xlabel("Value")
    plt.ylabel("")

    plt.show()

    

def main():
    """The script generates some of the graphics used in the thesis.
    """
    draw_crps_intuition()
    draw_stochastic_regression()
    draw_distribution_for_observation()
    draw_rank_hist()
    draw_point_vs_dens()






if __name__ == '__main__':
    main()
