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

from bnn_model import *
from mdn_model import *
from empirica_model import *


class Evaluator:



    def __init__(self, directory, description, out_val):
        self.desc = directory + "/desc_file.txt"
        self.directory = directory
        self.out_val = out_val

        if not os.path.isdir(directory):
            os.mkdir(directory)
        with open(self.desc, "w") as desc_f:
            desc_f.write(description)
            desc_f.write("--------")
            


    def set_names(self, col_name, res_name):
        self.col_name = col_name
        self.res_name = res_name

    def set_test_train_split(self, X_train, X_test, y_train, y_test ):
       self.X_train = X_train
       self.X_test  = X_test 
       self.y_train = y_train
       self.y_test = y_test
        


    def gen_feature_importance_data(self, data, col):
        X = data.T
        X[col] = shuffle(X[col])
        return X.T

        
        
       

    def evaluate_bnn(self, model):
        pass


    def evaluate_mdn(self, model):

        print("Evaluating MDN model")
        
        pis, mus, sigmas = model.eval_network(self.X_train)
        res_train_mu = np.sum(pis.T*mus.T, axis=0)
        res_train_std = np.sum(pis.T*sigmas.T, axis=0)

        pis, mus, sigmas = model.eval_network(self.X_test)
        res_test_mu = np.sum(pis.T*mus.T, axis=0)
        res_test_std = np.sum(pis.T*sigmas.T, axis=0)



        print("Generating plots")
        plt.figure(figsize=(15,13), dpi=100)

        plt.subplot(2,1,1)
        plt.plot(np.arange(self.y_train.shape[0]), self.y_train, '-b', linewidth=1.0,label='Station ' + self.res_name)
        plt.plot(np.arange(self.y_train.shape[0]), res_train_mu, '-r', color="green", linewidth=2.4,label='training data')
        plt.fill_between(np.arange(self.y_train.shape[0]),
                         res_train_mu - res_train_std,
                         res_train_mu + res_train_std,
                         color="red", alpha=0.5, label="90 confidence region")
        plt.legend()
        plt.title("Mixture Density Network")
        plt.xlabel("t")
        plt.ylabel(self.res_name)

        plt.subplot(2,1,2)
        plt.plot(np.arange(self.y_test.shape[0]), self.y_test, '-b', linewidth=1.0,label='Station ' + self.res_name)
        plt.plot(np.arange(self.y_test.shape[0]), res_test_mu, '-r', color="green", linewidth=2.4,label='training data')
        plt.fill_between(np.arange(self.y_test.shape[0]),
                         res_test_mu - res_test_std,
                         res_test_mu + res_test_std,
                         color="red", alpha=0.5, label="90% confidence region")
        plt.legend()
        plt.title("Mixture Density Network")
        plt.xlabel("t")
        plt.ylabel(self.res_name)
        
        
        plt.savefig(self.directory+"/mdn_data_plot.png", bbox_inches='tight')
        # plt.show()

        
        sc = Scorer(max_samples=5)
        
        print("Calculating rules on the test set")
        rules_val_test = {}
        for i in range(self.y_test.shape[0]):
            print("Looking at example", i)
            sc.set_sampler("mdn_test", lambda n: norm.rvs(size=int(n), loc=res_test_mu[i], scale=res_test_std[i]))
            rules = sc.single_model_evaluation(np.array(self.y_test[i]),"mdn_test",
                                               data_frame_file=self.directory+"/evaluation_results_df.csv")
            for rule, val in rules.items():
                if rule not in rules_val_test.keys():
                    rules_val_test[rule] = np.array([])
                rules_val_test[rule] = np.append(rules_val_test[rule], val)
            
        with open(self.desc, "w+") as desc_f:
            for rule, vals in rules_val_test.items():
                print(rule,":", vals.mean())
                desc_f.write("Results of MDN")
                desc_f.write(str(rule)+":"+str(vals.mean()))
                desc_f.write("------------")

        
        print("Calculating rules on the train set")
        rules_val_train = {}
        for i in range(self.y_test.shape[0]):
            print("Looking at example", i)
            sc.set_sampler("mdn_train", lambda n: norm.rvs(size=int(n), loc=res_train_mu[i], scale=res_train_std[i]))
            rules = sc.single_model_evaluation(np.array(self.y_train[i]),"mdn_train",
                                               data_frame_file=self.directory+"/evaluation_results_train_df.csv")
            for rule, val in rules.items():
                if rule not in rules_val_train.keys():
                    rules_val_train[rule] = np.array([])
                rules_val_train[rule] = np.append(rules_val_train[rule], val)
                                    
        with open(self.desc, "w+") as desc_f:
            for rule, vals in rules_val_train.items():
                print(rule,":", vals.mean())
                desc_f.write("Results of MDN")
                desc_f.write(str(rule)+":"+str(vals.mean()))
                desc_f.write("------------")

        print("Calcualting feature importance on the test set")
        featrue_imp = list(range(self.X_test.shape[1]))
        for i in range(self.X_test.shape[1]):
            featrue_imp[i] = {}

            X_shuf = self.gen_feature_importance_data(self.X_test, i)            
            pis, mus, sigmas = model.eval_network(X_shuf)
            mu = np.sum(pis.T*mus.T, axis=0)
            std = np.sum(pis.T*sigmas.T, axis=0)
            print("Col: ", i)
            rules_val = {}
            for j in range(self.y_test.shape[0]):
                print("Example: ", j)
                sc.set_sampler("mdn_im", lambda n: norm.rvs(size=int(n), loc=res_test_mu[i], scale=res_test_std[i]))                
                rules = sc.single_model_evaluation(np.array(self.y_test[j]),"mdn_im",data_frame_file=self.directory+"/evaluation_results_df.csv")
                for rule, val in rules.items():
                    if rule not in rules_val.keys():
                        rules_val[rule] = np.array([])
                    rules_val[rule] = np.append(rules_val[rule], val)

            
            for rule, vals in rules_val.items():
                diff = vals - rules_val_test[rule]
                print("Diff: ",rule,"",diff.mean())
                featrue_imp[i][rule] = diff

            
        #save to df! (model, rule, col, score)
            
            

        


                
        
    def evaluate_empirical(self,samples=10000):
        print("Evaluating empirical model")
        empirical_model = Emp("Empirical model")

        y = np.concatenate([self.y_train, self.y_test])

        start_pos = self.y_train.shape[0]
        end_pos = y.shape[0]

        res = np.array([
            empirical_model.build(y[0:i]).evaluate(samples).reshape(samples)
            for i in range(start_pos,end_pos)
        ], dtype=np.float32)
        mus = res.mean(axis=1)        
        np.savetxt(self.directory+"/empirical_result.txt", res)
        print("Generating plot")
        #generate plot
        plt.figure(figsize=(15,13), dpi=100)
        plt.title("Empirical Model")
        plt.xlabel("t")
        plt.ylabel(self.out_val)
        plt.plot(np.arange(y.shape[0]), y , '-b' , linewidth=1.0, label="Station: " + self.res_name)
        plt.plot(np.arange(start_pos, end_pos) ,mus , '-r', linewidth=1.1, label='mean of posterior')
        plt.fill_between(np.arange(start_pos, end_pos), np.percentile(res, 5, axis=1), np.percentile(res, 95, axis=1) , color="red", alpha=0.2, label="90% confidence region")
        plt.legend()
        plt.savefig(self.directory+"/epirical_data_plot.png", bbox_inches='tight')
        
        
        print("Calculating scoring rules")
        # calculate scoring rules
        sc = Scorer(max_samples=10000)
        rules_val = {}
        for i in range(start_pos,end_pos):
            print("Looking at example", i-start_pos)
            sc.set_sampler("empirical", lambda n: res[i - start_pos])
            rules = sc.single_model_evaluation(np.array(y[i]),"empirical",
                                                 data_frame_file=self.directory+"/evaluation_results_df.csv")
            for rule, val in rules.items():
                if rule not in rules_val.keys():
                    rules_val[rule] = np.array([])
                rules_val[rule] = np.append(rules_val[rule], [val])

        with open(self.desc, "w+") as desc_f:
            for rule, vals in rules_val.items():
                print(rule,":", vals.mean())
                desc_f.write("Results of Empirical")
                desc_f.write(str(rule)+":"+str(vals.mean()))
                desc_f.write("------------")
                

            
        

        

    

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
    pass





    

if __name__ == '__main__':
    main()
