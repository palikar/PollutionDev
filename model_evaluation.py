#!/home/arnaud/anaconda3/bin/python3.6



import sys, os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.stats import norm
import properscoring as ps
import scoring as sc
from scorer import Scorer
from sklearn.utils import shuffle


class Evaluator:



    def __init__(self, directory, description):
        if not os.path.isdir(directory):
            os.mkdir(directory)
            with open("desc_file.txt", "w") as desc:
                desc.write(description)


    def set_names(self, col_name, res_name):
        self.col_name = col_name
        self.res_name = res_name

    def set_test_train_split(self, X_train, X_test, y_train, y_split ):
       self.X_train = X_train
       self.X_test  = X_test 
       self.y_train = y_train
       self.y_split = y_split
        


    def gen_feature_importance_data(self, data, col):
        X = data.T
        X[col] = shuffle(X[col])
        return X.T

        
        
       

    def evaluate_bnn(self, model):
        pass


    def evaluate_mdn(self, model):
        pass
        
        
    def evaluate_empirical(self, model):
        pass

    

# plots with the mean and std proper scoring rules in the dataframe
# feature importance - interate columns, use scorerm generate rule
# vecors, substract them from the original(unpermuted), log results in
# DF (model(name, id), rule_col )




def main():
    # scorer = Scorer(min_samples=5, max_samples=10000, samples_cnt_step=30)
    # scorer.set_sampler("nomral_uo_std5", lambda n: norm.rvs(size=int(n), loc=0, scale=5))
    # res = scorer.single_model_evaluation(np.array([0.1]),"nomral_uo_std5",
    #                                      data_frame_file="./evaluation_results.csv")
    # print(res)





    

if __name__ == '__main__':
    main()
