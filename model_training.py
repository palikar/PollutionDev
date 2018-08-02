#!/home/arnaud/anaconda3/bin/python3

import os, sys, argparse

from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn import model_selection

import numpy as np
import pandas as pd
import edward as ed
import tensorflow as tf

from model_evaluation import Evaluator
from bnn_model import *
from mdn_model import *
from empirica_model import *
from utils import test_train_split
from utils import select_data
import utils as ut






def main():


    parser = argparse.ArgumentParser(description='tain model, save it, evaluate it!')

    parser.add_argument('--model', dest='model', action='store',
                    help='the model to be trained')

    parser.add_argument('--station', dest='station', action='store', default="SBC",
                        help='the model to be trained')

    parser.add_argument('--predictor', dest='pred_value', action='store', default="SBC",
                        help='the value(s) that should be used as features (P1, P2 of P1P2)')


    parser.add_argument('--period', dest='period', action='store', default="1D",
                        help='integration period for the data')

    parser.add_argument('--outvalue', dest='out_value', action='store', default="P1",
                        help='output value of the model(P1 or P2)')

    parser.add_argument('--take_lubw', dest='take_lu_bw', action='store_true', default=True,
                        help='should the LU BW station be taken as feature')
    
    parser.add_argument('--random_split', dest='random_split', action='store_true', default=False,
                        help='should the LU BW station be taken as feature')

    parser.add_argument('--dest', dest='dest', action='store', required=False, default="/home/arnaud/code/pollution/test_eval",
                        help='destination for the evaluation and for the build models')

    
    args = parser.parse_args()

    
    station = args.station
    value = args.out_value
    period = args.period
    out_value = args.out_value
    train_per = 0.75
    take_lu_bw = args.take_lu_bw
    random_split = args.random_split
    
    
    X, y, col_names, out_name = select_data(station, value, period,include_lu_bw=take_lu_bw, output_value=out_value)
    X_train, X_test, y_train, y_test = test_train_split(X, y,train_size=train_per, random=random_split)
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

    samples_cnt = 10000
    
    mdn_iter = 100000
    mdn_layers = [10, 10]
    mdn_mixture_cnt = 20
    
    desc = ""
    desc += "\nPeriod: " + str(period)
    desc += "\nTraining samples: " + str(X_train.shape[0])
    desc += "\nTest samples: " + str( X_train.shape[0])
    desc += "\nNumber of features: " + str(X_train.shape[1])
    desc += "\nTaking LU BW as feature: " + str(take_lu_bw)
    desc += "\nTarget value: " + str(value)               
    desc += "\nTarget station " + str(station)           
    desc += "\nInput features: " + str(col_names)
    desc += "\nOutpute value" + str(out_name)
    desc += "\nTest-train split ratio" + str(train_per)    
    desc += "\n-------------\n"

    desc += "\nMDN Configuration: "
    desc += "\nIterations: " + str(mdn_iter)
    desc += "\nLayers: " + str(mdn_layers)
    desc += "\nMixtures Count: " + str(mdn_mixture_cnt)
    desc += "\n-------------\n"


    desc += "\nEvaluation Configuration"
    desc += "\nSamples drawn from models for each observation" + str(mdn_iter)
    


    
    mdn_model = Mdn("MDN Model", X_train, y_train, inner_dims=mdn_layers, num_mixtures=mdn_mixture_cnt)
    # mdn_model.fit(num_iter=mdn_iter)
    # mdn_model.save("/home/arnaud/code/pollution/test_eval/mdn_model")
    
    ev = Evaluator("/home/arnaud/code/pollution/test_eval", desc, out_value)
    ev.set_test_train_split(X_train, X_test, y_train, y_test)
    ev.set_names(col_names, out_name)
    
    # ev.evaluate_empirical(samples=samples_cnt)
    ev.evaluate_mdn(mdn_model, samples=samples_cnt)
    

if __name__ == '__main__':
    main()
