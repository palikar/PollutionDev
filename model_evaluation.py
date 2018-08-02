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
        if not os.path.isdir(directory):
            os.mkdir(directory)
            
        self.directory = directory
        self.desc = directory + "/desc_file.txt"
        self.out_val = out_val



        with open(self.desc, "w") as desc_f:
            desc_f.write(description+"\n")
            desc_f.write("--------\n")
            


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


    
    def calculate_rules(self, y, scorer, model_name, sample_data, res_file):
        scores = scorer.simple_evaluation(sample_data, y, model_name, df_file=res_file)
        return scores

            

    def calculate_feature_imp(self, scorer, X, y, generate_sample_data, model_name, true_vals):
        feature_imp = list(range(X.shape[1]))
        for i in range(X.shape[1]):
            print("Column: ", self.col_name[i], "(", i,"/",X.shape[1],")")
            feature_imp[i] = {}
            
            X_shuf = self.gen_feature_importance_data(X, i)
            sample_data = generate_sample_data(X_shuf)
            
            rules_val = self.calculate_rules(
                y, scorer, model_name,
                sample_data,
                None)
            for rule, vals in rules_val.items():
                diff = vals - true_vals[rule]
                print("Diff: ",rule," - ", diff.mean())
                feature_imp[i][rule] = diff
        return feature_imp

    def log_rules(self, rules_val, header):
        with open(self.desc, "a") as desc_f:
            desc_f.write(header+"\n")
            for rule, vals in rules_val.items():
                print(rule,":", vals.mean())
                desc_f.write(str(rule)+":"+str(vals.mean())+"\n")
            desc_f.write("------------\n")            

        

    def log_feature_importance(self, df_file, feature_imp, model_name):
        cols = ["Model", "feature", "rule", "diff_value"]
        feature_imp_df = pd.DataFrame(columns=cols)
        
        for i in range(len(feature_imp)):
            feature = self.col_name[i]
            for rule, diff_val in feature_imp[i].items():
                feature_imp_df = feature_imp_df.append(
                    {
                        cols[0] : model_name,
                        cols[1] : feature,
                        cols[2] : rule,
                        cols[3] : diff_val.mean()
                    }, ignore_index=True)
        if not os.path.isfile(df_file):
            feature_imp_df.to_csv(df_file, sep=";")
        else:
            feature_imp_df.to_csv(df_file, sep=";", mode="a", header=False)



    
    def mdn_samples(self, model, X, samples):
        pis, mus, sigmas = model.eval_network(X)
        mu = np.sum(pis.T*mus.T, axis=0)
        std = np.sum(pis.T*sigmas.T, axis=0)
        sample_data = np.array([
            norm.rvs(size=samples, loc=mu[i], scale=std[i])
            for i in range(mu.shape[0])
        ])
        return sample_data




    
    def evaluate_bnn(self, model):
        pass


    def evaluate_mdn(self, model, samples=10000):
        print("Evaluating MDN model")
        
        pis, mus, sigmas = model.eval_network(self.X_train)
        res_train_mu = np.sum(pis.T*mus.T, axis=0)
        res_train_std = np.sum(pis.T*sigmas.T, axis=0)

        pis, mus, sigmas = model.eval_network(self.X_test)
        res_test_mu = np.sum(pis.T*mus.T, axis=0)
        res_test_std = np.sum(pis.T*sigmas.T, axis=0)

        sc = Scorer(max_samples=samples)

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
        


        print("Calculating rules on the test set")
        sample_data = np.array([
            norm.rvs(size=samples, loc=res_test_mu[i], scale=res_test_std[i])
            for i in range(res_test_mu.shape[0])
        ])
        rules_val_test = self.calculate_rules(
            self.y_test, sc, "mdn_test",
            sample_data,
            self.directory+"/evaluation_results_df.csv")
        self.log_rules(rules_val_test, "Results of MDN on test")


        
        print("Calculating rules on the train set")
        sample_data = np.array([
            norm.rvs(size=samples, loc=res_train_mu[i], scale=res_train_std[i])
            for i in range(res_train_mu.shape[0])
        ])
        rules_val_train = self.calculate_rules(
            self.y_train, sc, "mdn_train",
            sample_data,
            self.directory+"/evaluation_results_train_df.csv")
        self.log_rules(rules_val_train, "Results of MDN on train")


        
        print("Calcualting feature importance on the test set")
        feature_imp = self.calculate_feature_imp(sc, self.X_test, self.y_test, lambda X: self.mdn_samples(model, X, samples) , "mdn_feature_imp", rules_val_test)
        self.log_feature_importance(self.directory+"/feature_importance.csv", feature_imp, "mdn_test")
                
        

                
        
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
        sc = Scorer(max_samples=samples)

        rules_val = {}
        for i in range(start_pos,end_pos):
            sc.set_sampler("empirical", lambda n: res[i - start_pos])
            rules = sc.single_model_evaluation(np.array(y[i]),"empirical",
                                                 data_frame_file=self.directory+"/evaluation_results_df.csv")
            for rule, val in rules.items():
                if rule not in rules_val.keys():
                    rules_val[rule] = np.array([])
                rules_val[rule] = np.append(rules_val[rule], [val])

        with open(self.desc, "a") as desc_f:
            desc_f.write("Results of Empirical on test\n")
            for rule, vals in rules_val.items():
                print(rule,":", vals.mean())
                desc_f.write(str(rule)+":"+str(vals.mean())+"\n")
            desc_f.write("------------\n")
                

            
        

        




def main():
    pass






    

if __name__ == '__main__':
    main()
