#!/home/arnaud/anaconda3/bin/python3

import os, sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from bnn_model import *
from mdn_model import *
from empirica_model import *
import utils as ut
from utils import test_train_split
from utils import select_data

import edward as ed
from scipy.stats import norm
import tensorflow as tf

from sklearn import model_selection

from model_evaluation import Evaluator

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



def main():
    station = "SBC"
    value="P1"
    period="1D"
    train_per = 0.75
    take_lu_bw = True
    out_value="P1"
    
    
    X, y, col_names, out_name = select_data(station, value, period,include_lu_bw=take_lu_bw, output_value=out_value)
    X_train, X_test, y_train, y_test = test_train_split(X, y,train_size=train_per, random=False)
    y_train = y_train.reshape(y_train.shape[0],1)
    y_test = y_test.reshape(y_test.shape[0],1)


    print("Period: " + period)
    print("Training samples: ", X_train.shape[0])
    print("Test samples: ", X_train.shape[0])
    print("Number of features: ", X_train.shape[1])
    print("Target value: " + value)
    print("Target station", station)
    print("Input features:", col_names)
    print("Outpute value", out_name)
    print("-------------")

    mdn_model = Mdn("MDN Model", X_train, y_train, inner_dims=[100, 100, 50], num_mixtures=1)
    bnn_model = Bnn("BNN Model")
    # mdn_model.fit(num_iter=600000)
    
    ev = Evaluator("/home/arnaud/code/pollution/test_eval", "Description", out_value)
    ev.set_test_train_split(X_train, X_test, y_train, y_test)
    ev.set_names(col_names, out_name)


    
    # ev.evaluate_empirical()
    ev.evaluate_mdn(mdn_model)
    

if __name__ == '__main__':
    main()
