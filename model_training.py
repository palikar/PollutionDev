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
import json




#rm -rf test_eval/ && ./model_training.py --config ./model_config.json --model mdn --station SBC --predictor P1P2 --period 1D --outvalue P1  --dest "/home/arnaud/code/pollution/test_eval" --base-dir "./env/data_frames"
# ./model_training.py --config ./model_config.json --model both --station SBC --predictor P1 --period 1D --outvalue P1 --take_lubw  --dest "/home/arnaud/code/pollution/test_eval_1" --base-dir "./env/data_frames" --load-mdn ./test_eval/mdn_model/model --load-bnn ./test_eval/bnn_model/



def plot_bnn(X_train, X_test, y_train, y_test, directory,  model, i):
    samples = 50000
    res_train = model.evaluate(X_train, samples)
    res_train = res_train.reshape(samples, X_train.shape[0])

    res_test = model.evaluate(X_test, samples)
    res_test = res_test.reshape(samples, X_test.shape[0])

    plt.figure(figsize=(15,13), dpi=100)
    plt.subplot(2,1,1)
    plt.plot(np.arange(y_train.shape[0]), y_train, '-b', linewidth=1.0,label='Data')
    plt.plot(np.arange(y_train.shape[0]), np.mean(res_train, 0).reshape(-1), 'r-', lw=2, label="Posterior mean")
    plt.fill_between(np.arange(y_train.shape[0]),
                     np.percentile(res_train, 5, axis=0),
                     np.percentile(res_train, 95, axis=0),
                     color = "red", alpha = 0.5, label="90% confidence region")
    plt.ylim(y_train.min() - 10, y_train.max() + 10)
    plt.legend()
    plt.title("Bayesian Neural Network(train set), it: " + str(i))
    plt.xlabel("t")
    plt.ylabel("Value")        
    plt.subplot(2,1,2)
    plt.plot(np.arange(y_test.shape[0]), y_test, '-b', linewidth=1.0,label='Data')
    plt.plot(np.arange(y_test.shape[0]), np.mean(res_test, 0).reshape(-1), 'r-', lw=2, label="Posterior mean")
    plt.fill_between(np.arange(y_test.shape[0]),
                     np.percentile(res_test, 5, axis=0),
                     np.percentile(res_test, 95, axis=0),
                     color = "red", alpha = 0.5, label="90% confidence region")
    plt.ylim(y_test.min() - 10, y_test.max() + 10)
    plt.legend()
    plt.title("Bayesian Neural Network(test set), it: " + str(i))
    plt.xlabel("t")
    plt.ylabel("Value")
    plt.savefig(directory+"/bnn_data_plot_"+str(i)+".png", bbox_inches='tight')


    
def plot_mdn(X_train, X_test, y_train, y_test, directory,  model, i):
    samples = 50000

    def sample_mixed( pis, mus, sigmas, j, size=1):
        choice = np.random.choice(np.arange(0, pis.shape[1]), p=pis[j])
        return norm.rvs(size=size, loc=mus[j][choice], scale=sigmas[j][choice])

    
    pis_train, mus_train, sigmas_train = model.eval_network(X_train)
    res_train_mu = np.sum(pis_train.T*mus_train.T, axis=0)
    sampled_train = np.array([ sample_mixed(pis_train, mus_train, sigmas_train, j, size=samples) for j in range(y_train.shape[0])])
    
    pis_test, mus_test, sigmas_test = model.eval_network(X_test)
    res_test_mu = np.sum(pis_test.T*mus_test.T, axis=0)
    sampled_test = np.array([ sample_mixed(pis_test, mus_test, sigmas_test, j, size=samples) for j in range(y_test.shape[0])])
    
    plt.figure(figsize=(15,13), dpi=100)
    plt.subplot(2,1,1)
    plt.plot(np.arange(y_train.shape[0]), y_train, '-b', linewidth=1.0,label='Data')
    plt.plot(np.arange(y_train.shape[0]), res_train_mu, '-r', color="green", linewidth=2.4,label='Distribution mean')        
    plt.fill_between(np.arange(y_train.shape[0]),
                     np.percentile(sampled_train, 5, axis=1),
                     np.percentile(sampled_train, 95, axis=1),
                     color="red", alpha=0.5, label="90 confidence region")
    plt.ylim(y_train.min() - 10, y_train.max() + 10)
    plt.legend()
    plt.title("Mixture Density Network(train set)")
    plt.xlabel("t")
    plt.ylabel("Value")
    plt.subplot(2,1,2)
    plt.plot(np.arange(y_test.shape[0]), y_test, '-b', linewidth=1.0,label='Data')
    plt.plot(np.arange(y_test.shape[0]), res_test_mu, '-r', color="green", linewidth=2.4,label='Distribution mean')
    plt.fill_between(np.arange(y_test.shape[0]),
                     np.percentile(sampled_test, 5, axis=1),
                     np.percentile(sampled_test, 95, axis=1),
                     color="red", alpha=0.5, label="90 confidence region")
    plt.ylim(y_test.min() - 10, y_test.max() + 10)
    plt.legend()
    plt.title("Mixture Density Network(test set)")
    plt.xlabel("t")
    plt.ylabel("Value")
    plt.savefig(directory + "/mdn_data_plot_" + str(i) + ".png", bbox_inches='tight')


    
def main():
    
    parser = argparse.ArgumentParser(description='tain model, save it, evaluate it!')
    
    parser.add_argument('--model', dest='model', action='store',
                    help='the model to be trained')

    parser.add_argument('--config', dest='config', action='store', required=True,
                    help='the configuration file with setting for the architecture of the models')

    parser.add_argument('--station', dest='station', action='store', default="SBC",
                        help='the model to be trained')

    parser.add_argument('--predictor', dest='pred_value', action='store', default="P1",
                        help='the value(s) that should be used as features (P1, P2 of P1P2)')

    parser.add_argument('--period', dest='period', action='store', default="1D",
                        help='integration period for the data (1D, 1H, 12H)')

    parser.add_argument('--outvalue', dest='out_value', action='store', default="P1",
                        help='output value of the model(P1 or P2)')

    parser.add_argument('--take_lubw', dest='take_lu_bw', action='store_true', default=False,
                        help='should the LU BW station be taken as feature')
    
    parser.add_argument('--random_split', dest='random_split', action='store_true', default=False,
                        help='should the LU BW station be taken as feature')

    parser.add_argument('--dest', dest='dest', action='store', required=False, default="/home/arnaud/code/pollution/test_eval",
                        help='destination for the evaluation and for the build models')

    parser.add_argument('--base-dir', dest='base_dir', action='store', required=False, default="/home/arnaud/code/pollution/env/data_frames",
                        help='The directory where the data frames reside')

    parser.add_argument('--load-mdn', dest='load_mdn', action='store', required=False, default=None,
                        help='Load the MDB model from specific folder and dont train a new one')

    parser.add_argument('--load-bnn', dest='load_bnn', action='store', required=False, default=None,
                        help='Load the BNN model from specific folder and dont train a new one')

    
    args = parser.parse_args()

    
    station = args.station
    in_value = args.pred_value
    period = args.period
    out_value = args.out_value
    train_per = 0.75
    take_lu_bw = args.take_lu_bw
    random_split = args.random_split
    base_dir = args.base_dir
    dest = args.dest
    
    
    X, y, col_names, out_name = select_data(station, in_value, period, include_lu_bw=take_lu_bw, output_value=out_value, base_dir=base_dir)
    X_train, X_test, y_train, y_test = test_train_split(X, y,train_size=train_per, random=random_split)
    y_train = y_train.reshape(y_train.shape[0],1)
    y_test = y_test.reshape(y_test.shape[0],1)

    config_file = open(args.config, "r")
    config = json.load(config_file)
    config_file.close()

    
    print("Period: " + period)
    print("Training samples: ", X_train.shape[0])
    print("Test samples: ", X_test.shape[0])
    print("Number of features: ", X_train.shape[1])
    print("Input features: " + in_value)
    print("Target station", station)
    print("Input features:", col_names)
    print("Outpute value", out_name)
    print("-------------")


    
    ev_samples_cnt = config["ev_samples_cnt"]
    
    mdn_iter = config["mdn"]["mdn_iter"]
    mdn_layers = config["mdn"]["mdn_layers"]
    mdn_mixture_cnt = config["mdn"]["mdn_mixture_cnt"]
    mdn_id = "mdn_l"+str(mdn_layers)+"_i"+str(mdn_iter)+"_mc"+str(mdn_mixture_cnt)


    bnn_samples = config["bnn"]["bnn_samples"]
    bnn_iter = config["bnn"]["bnn_iter"]
    bnn_layers = config["bnn"]["bnn_layers"]
    bnn_id = "bnn_l"+str(bnn_layers)+"_i"+str(bnn_iter)+"_s"+str(bnn_samples)

    
    desc = ""
    desc += "\nPeriod: " + str(period)
    desc += "\nTraining samples: " + str(X_train.shape[0])
    desc += "\nTest samples: " + str( X_test.shape[0])
    desc += "\nNumber of features: " + str(X_train.shape[1])
    desc += "\nTaking LU BW as feature: " + str(take_lu_bw)
    desc += "\nInput value: " + str(in_value)               
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

    desc += "\nBNN Configuration: "
    desc += "\nIterations: " + str(bnn_iter)
    desc += "\nLayers: " + str(bnn_layers)
    desc += "\nSamples for vatiational inference " + str(bnn_samples)
    desc += "\n-------------\n"


    desc += "\nEvaluation Configuration"
    desc += "\nSamples drawn from models for each observation: " + str(ev_samples_cnt)
    



    ev = Evaluator(dest, desc, out_value)
    ev.set_test_train_split(X_train, X_test, y_train, y_test)
    ev.set_names(col_names, out_name)

    os.makedirs(dest+"/bnn_train_plots")
    os.makedirs(dest+"/mdn_train_plots")


    
    def get_mdn():
        if args.load_mdn is None:
            mdn_model = Mdn("MDN Model", X_train, y_train, inner_dims=mdn_layers, num_mixtures=mdn_mixture_cnt)
            mdn_model.fit(num_iter=mdn_iter,
                          callback = lambda mod,j: plot_mdn(X_train, X_test, y_train, y_test, dest+"/mdn_train_plots", mod, j) )
            mdn_model.save(dest + "/mdn_model")
            return mdn_model
        else:
            print("Loading MDN from file")
            mdn_model = Mdn("MDN Model", X_train, y_train, inner_dims=mdn_layers, num_mixtures=mdn_mixture_cnt, model_file=args.load_mdn)
            mdn_model.save(dest + "/mdn_model")
            return mdn_model


    def get_bnn():
        if args.load_bnn is None:
            bnn_model = Bnn("BNN Model")
            bnn_model.build(X_train.shape[1],1,layers_defs=bnn_layers, examples = X_train.shape[0])
            bnn_model.fit(X_train, np.squeeze(y_train), epochs=bnn_iter, samples=bnn_samples,
                          callback = lambda mod,j: plot_bnn(X_train, X_test, y_train, y_test, dest+"/bnn_train_plots", mod, j))
            bnn_model.save(dest + "/bnn_model", "bnn_model")
            return bnn_model
        else:
            print("Loading BNN from file")
            bnn_model = Bnn("BNN Model")
            bnn_model.load(args.load_bnn, name="bnn_model")
            bnn_model.save(dest + "/bnn_model", "bnn_model")
            return bnn_model



    if args.model == "bnn":
        print("Fitting the BNN")
        bnn_model = get_bnn()
        ev.evaluate_bnn(bnn_model, bnn_id, samples=ev_samples_cnt)
    elif args.model == "mdn":
        print("Fitting the MDN")
        mdn_model = get_mdn()
        ev.evaluate_mdn(mdn_model, mdn_id, samples=ev_samples_cnt)
    else:
        print("Fitting the MDN")
        mdn_model = get_mdn()
        ev.evaluate_mdn(mdn_model, mdn_id, samples=ev_samples_cnt)

        tf.reset_default_graph()
        
        print("Fitting the BNN")
        bnn_model = get_bnn()
        ev.evaluate_bnn(bnn_model,bnn_id, samples=ev_samples_cnt)
        

    ev.evaluate_empirical(samples=ev_samples_cnt)



    
    
    

if __name__ == '__main__':
    main()
