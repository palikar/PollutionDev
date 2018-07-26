#!/usr/bin/python

import os, sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from bnn_model import *
from gm_model import *
from empirica_model import *
import utils as ut
from utils import test_train_split
from utils import select_data


import edward as ed
from scipy.stats import norm
import tensorflow as tf

from sklearn import model_selection







def train_bnn(X_train, X_test, y_train, y_test):

    layers = [10,10,10]
    batch_size_bnn = y_train.shape[0]
    epochs_bnn = 10000
    updates_per_batch_bnn = 1
    
    model_id = "bnn_l"+str(layers);
    print("Model id:" + model_id)
    print("Layers:" + str(layers))
    model = Bnn(model_id)
    model.build(X_train.shape[1], 1, layers)
    model.fit(X_train, y_train, M=batch_size_bnn, updates_per_batch=updates_per_batch_bnn, epochs=epochs_bnn)


    return model



def train_mdn(X_train, X_test, y_train, y_test):
    mixtures_cnt = 2
    layers = [20,50,20]
    iterations = 150000

    model_id = "mdn_l"+str(layers)
    print("Model id:" + model_id)
    mdn_model = Mdn(model_id, X_train, y_train, inner_dims=layers, num_mixtures=mixtures_cnt)
    mdn_model.fit(num_iter=iterations)
    return mdn_model





def visualize_bnn():
    pass



def visualize_mdn():
    pass
    






    

def main():    
    station = "SBC"
    value="P1"
    period="1D"
    train_per = 0.75

    
    X, y = select_data(station, value, period)
    X_train, X_test, y_train, y_test = test_train_split(X, y,train_size=train_per, random=False)
    y_train = y_train.reshape(y_train.shape[0],1)
    y_test = y_test.reshape(y_test.shape[0],1)    

    print("Period: " + period)
    print("Training samples: ", X_train.shape[0])
    print("Test samples: ", X_train.shape[0])
    print("Number of features: ", X_train.shape[1])
    print("Target value: " + value)
    print("Target station", station)
    print("-------------")
    

    
    model = train_mdn(X_train, X_test, y_train, y_test)
    #model.save("./env/models/" + model_id +"/", model_id)
    


    print("Fitting is now complete")
    
    

    plt.figure(figsize=(15,13), dpi=100)


    pis, mus, sigmas = model.eval_network(X_train)
    res = np.sum(pis.T*mus.T, axis=0)
    res_1 = np.sum(pis.T*sigmas.T, axis=0)
    
    plt.subplot(2,1,1)


    plt.plot(np.linspace(0, len(y_train), num=len(y_train)), res, '-g', color="green", linewidth=2.4,label='mean')
    plt.fill_between(np.linspace(0, len(y_train), num=len(y_train)), res-res_1, res+res_1, color="red", alpha=0.5, label='confidance region')

    plt.plot(np.arange(y_train.shape[0]), y_train, '-b' , linewidth=0.5,label='Data')

    
    plt.title("Mdn Model(Train set)")
    plt.xlabel("point[i], t")
    plt.ylabel("output")

    pis, mus, sigmas = model.eval_network(X_test)
    res = np.sum(pis.T*mus.T, axis=0)
    res_1 = np.sum(pis.T*sigmas.T, axis=0)
    plt.subplot(2,1,2)


    plt.plot(np.linspace(0, len(y_test), num=len(y_test)), res, '-g', color="green", linewidth=2.4,label='mean')
    plt.fill_between(np.linspace(0, len(y_test), num=len(y_test)), res-res_1, res+res_1, color="red", alpha=0.5, label='confidance region')

    plt.plot(np.arange(y_test.shape[0]), y_test, '-b' , linewidth=0.5,label='Data')

    
    plt.title("Mdn Model(Train set)")
    plt.xlabel("point[i], t")
    plt.ylabel("output")


    
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()



    









    

# line, = plt.plot(np.arange(len(outputs[0].reshape(-1))), np.mean(outputs, 0).reshape(-1),'r', lw=2, label="posterior mean")
# plt.fill_between(np.arange(len(outputs[0].reshape(-1))),
#                 np.percentile(outputs, 5, axis=0),
#                 np.percentile(outputs, 95, axis=0),
#                 color=line.get_color(), alpha = 0.3, label="confidence_region")    
# plt.plot(np.arange(y_train.shape[0]), y_train, '-b' , linewidth=0.5,label='Data')



# plt.title("Bnn Model(Train set)")
# plt.xlabel("point[i], t")
# plt.ylabel("output")



# plt.subplot(2,1,2)

# samples = 100
# outputs = model.evaluate(X_test, samples)
# outputs = outputs.reshape(samples,X_test.shape[0])

    

# line, = plt.plot(np.arange(len(outputs[0].reshape(-1))), np.mean(outputs, 0).reshape(-1),'r', lw=2, label="posterior mean")
# plt.fill_between(np.arange(len(outputs[0].reshape(-1))),
#                 np.percentile(outputs, 5, axis=0),
#                 np.percentile(outputs, 95, axis=0),
#                 color=line.get_color(), alpha = 0.3, label="confidence_region")

# plt.plot(np.arange(y_test.shape[0]), y_test, '-b' , linewidth=0.5,label='Data')



# plt.title("Bnn Model(Test set)")
# plt.xlabel("point[i], t")
# plt.ylabel("output")












