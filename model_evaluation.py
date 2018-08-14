#!/home/arnaud/anaconda3/bin/python3.6



import sys, os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.stats import norm
from scipy.stats import gaussian_kde
import properscoring as ps
import scoring as sc
from scorer import Scorer
from sklearn.utils import shuffle

from bnn_model import *
from mdn_model import *
from empirica_model import *

import properscoring as ps


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
        

    def calculate_feature_imp(self, X, fun, true_vals):
        feature_imp = list(range(X.shape[1]))
        for i in range(X.shape[1]):
            print("Column: ", self.col_name[i], "(", i,"/",X.shape[1],")")
            feature_imp[i] = {}
            
            X_shuf = self.gen_feature_importance_data(X, i)
            scores_dict = fun(X_shuf)
            

            for rule, vals in scores_dict.items():
                diff = vals - true_vals[rule]
                print("Diff: ",rule," - ", diff.mean())
                feature_imp[i][rule] = diff
        return feature_imp


    def log_scores(self, model_id, scores_dict, df_file, header):
        with open(self.desc, "a") as desc_f:
            desc_f.write(header+"\n")
            for rule, val in scores_dict.items():
                print(rule,":", val)
                desc_f.write(str(rule)+" : "+str(val)+"\n")
            desc_f.write("------------\n")
        
        if df_file is not None:
            data = {}
            data["model_id"] = [model_id]
            for rule, score in scores_dict.items():
                data[rule] = [score]
            df = pd.DataFrame(data)
            if os.path.isfile(df_file):
                df.to_csv(df_file,
                          sep=";",
                          mode="a",
                          index=False,
                          header=False)
            else:
                df.to_csv(df_file,
                          index=False,
                          sep=";")

        
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
            feature_imp_df.to_csv(df_file, sep=";", index=False)
        else:
            feature_imp_df.to_csv(df_file, sep=";", mode="a", header=False, index=False)

        with open(self.desc, "a") as desc_f:
            desc_f.write("Feature importance for " + model_name +"\n")
            for i in range(len(feature_imp)):
                feature = self.col_name[i]
                desc_f.write(feature + ":\n")
                for rule, diff_val in feature_imp[i].items():
                    desc_f.write("     Mean diff in " + rule + " : "+ str(diff_val.mean()) +"\n")
                    
            desc_f.write("------------\n")

            






    def get_ranks(self, obs, samples):
        sorted = np.sort(samples, axis=1, kind='quicksort')
        return np.array([
            np.searchsorted(sorted[j], obs[j])
            for j in np.arange(obs.shape[0])
        ])


    def generate_rank_hist(self, y, samples, image_file, heading ):
        poss = self.get_ranks(y, samples).squeeze()
        plt.figure(figsize=(15,13), dpi=100)
        n_bins = 11
        plt.hist(poss, bins=n_bins, density=True, facecolor='green', alpha=0.75, histtype='stepfilled', edgecolor='black', linewidth=1.0, rwidth=0.9)
        plt.title(heading)
        plt.xlabel("Rank")
        plt.ylabel("Fraction")
        plt.grid(True)
        plt.savefig(image_file, bbox_inches='tight')

        

    
    def mixed_desnity(self, pis, mus, sigmas, y, j):
        val = 0.0
        for i in np.arange(pis.shape[1]):
            val = val + pis[j][i] * norm.pdf(y, mus[j][i], sigmas[j][i])
        return val


    def sample_mixed(self, pis, mus, sigmas, j, size=1):
        choice = np.random.choice(np.arange(0, pis.shape[1]), p=pis[j])
        return norm.rvs(size=size, loc=mus[j][choice], scale=sigmas[j][choice])


    def mdn_rules(self, model, X, y, samples):
        pis, mus, sigmas = model.eval_network(X)
        res_mu = np.sum(pis.T*mus.T, axis=0)
        sampled = np.array([ self.sample_mixed(pis, mus, sigmas, j, size=samples) for j in range(y.shape[0])])
        
        log_scores = -np.log(np.array([self.mixed_desnity(pis, mus, sigmas, y, j) for j, y in enumerate(y)]).clip(0.001))

        crps_scores = np.array([ ps.crps_ensemble(y_val, sampled[j]) for j, y_val in enumerate(y.squeeze())]) #fixed
        dss_scores = np.array([sc.dss_norm(y, loc=res_mu[j], scale=sampled[j,:].std()) for j, y in enumerate(y)])

        scores = dict()
        scores['CRPS'] = crps_scores.mean()
        scores['LS'] = log_scores.mean()
        scores['DSS'] = dss_scores.mean()
        # print(scores['DSS'])
        
        return scores


    def evaluate_mdn(self, model, model_id, samples=10000):
        print("Evaluating MDN model")

        pis_train, mus_train, sigmas_train = model.eval_network(self.X_train)
        res_train_mu = np.sum(pis_train.T*mus_train.T, axis=0)
        sampled_train = np.array([ self.sample_mixed(pis_train, mus_train, sigmas_train, j, size=samples) for j in range(self.y_train.shape[0])])
        
        pis_test, mus_test, sigmas_test = model.eval_network(self.X_test)
        res_test_mu = np.sum(pis_test.T*mus_test.T, axis=0)
        sampled_test = np.array([ self.sample_mixed(pis_test, mus_test, sigmas_test, j, size=samples) for j in range(self.y_test.shape[0])])
        
        # print("Log results to file")
        # np.savetxt(self.directory +'/mdn_test_samples.out', sampled_test, delimiter=',')
        # np.savetxt(self.directory +'/mdn_train_samples.out', sampled_train, delimiter=',')
        # np.savetxt(self.directory +'/mdn_train_pis.out', pis_train, delimiter=',')
        # np.savetxt(self.directory +'/mdn_test_pis.out', pis_test, delimiter=',')
        # np.savetxt(self.directory +'/X_train.out', self.X_train, delimiter=',')
        # np.savetxt(self.directory +'/X_test.out', self.X_test, delimiter=',')
        # np.savetxt(self.directory +'/y_train.out', self.y_train, delimiter=',')
        # np.savetxt(self.directory +'/y_test.out', self.y_test, delimiter=',')

        print("Generating plots")
        plt.figure(figsize=(15,13), dpi=100)
        plt.subplot(2,1,1)
        plt.plot(np.arange(self.y_train.shape[0]), self.y_train, '-b', linewidth=1.0,label='Station ' + self.res_name)
        plt.plot(np.arange(self.y_train.shape[0]), res_train_mu, '-r', color="green", linewidth=2.4,label='Distribution mean')        
        plt.fill_between(np.arange(self.y_train.shape[0]),
                         np.percentile(sampled_train, 5, axis=1),
                         np.percentile(sampled_train, 95, axis=1),
                         color="red", alpha=0.5, label="90 confidence region")
        plt.ylim(self.y_train.min() - 10, self.y_train.max() + 10)
        plt.legend()
        plt.title("Mixture Density Network(train set)")
        plt.xlabel("t")
        plt.ylabel(self.res_name)
        plt.subplot(2,1,2)
        plt.plot(np.arange(self.y_test.shape[0]), self.y_test, '-b', linewidth=1.0,label='Station ' + self.res_name)
        plt.plot(np.arange(self.y_test.shape[0]), res_test_mu, '-r', color="green", linewidth=2.4,label='Distribution mean')
        plt.fill_between(np.arange(self.y_test.shape[0]),
                         np.percentile(sampled_test, 5, axis=1),
                         np.percentile(sampled_test, 95, axis=1),
                         color="red", alpha=0.5, label="90 confidence region")
        plt.ylim(self.y_test.min() - 10, self.y_test.max() + 10)
        plt.legend()
        plt.title("Mixture Density Network(test set)")
        plt.xlabel("t")
        plt.ylabel(self.res_name)
        plt.savefig(self.directory + "/mdn_data_plot.png", bbox_inches='tight')


        print("Generating rank histograms")
        self.generate_rank_hist(self.y_test, sampled_test, self.directory+"/mdn_rank_hist_test.png" , "MDN rank histogram on test set")
        self.generate_rank_hist(self.y_train, sampled_train, self.directory+"/mdn_rank_hist_train.png" , "MDN rank histogram on train set")

        print("Calculating rules on train set")
        scores_train = self.mdn_rules(model, self.X_train, self.y_train, samples)
        self.log_scores(model_id+"_test", scores_train, self.directory + "/rules_scores_train.csv", "Results of MDN on train set\n")

        
        print("Calculating rules on test set")
        scores_test = self.mdn_rules(model, self.X_test, self.y_test, samples)
        self.log_scores(model_id+"_train", scores_test, self.directory + "/rules_scores.csv", "Results of MDN on test set\n")


        print("Calcualting feature importance on the test set")
        feature_imp = self.calculate_feature_imp(self.X_test, lambda X: self.mdn_rules(model, X, self.y_test, samples), scores_test)
        self.log_feature_importance(self.directory+"/feature_importance.csv", feature_imp, model_id + "_test")

        print("Calcualting feature importance on the train set")
        feature_imp = self.calculate_feature_imp(self.X_train, lambda X: self.mdn_rules(model, X, self.y_train, samples), scores_train)
        self.log_feature_importance(self.directory+"/feature_importance_train.csv", feature_imp, model_id + "_test")

        
                
        
    def bnn_rules(self, model, X, y, samples):
        res_train = model.evaluate(X, samples)
        res_train = res_train.reshape(samples, X.shape[0])
        sampled = res_train.T


        
        
        log_scores = -np.log(np.array([gaussian_kde(sampled[j]).pdf(y)  for j, y in enumerate(y)]).clip(0.001)) #fixed    
        crps_scores = np.array([ ps.crps_ensemble(y_val, sampled[j]) for j, y_val in enumerate(y.squeeze())]) #fixed    
        dss_scores = np.array([sc.dss_norm(y, loc=sampled[j].mean(), scale=sampled[j].std()) for j, y in enumerate(y)])

        scores = dict()
        scores['CRPS'] = crps_scores.mean()
        scores['LS'] = log_scores.mean()
        scores['DSS'] = dss_scores.mean()
        
        return scores


    def evaluate_bnn(self, model, model_id, samples=10000):

        res_train = model.evaluate(self.X_train, samples)
        res_train = res_train.reshape(samples, self.X_train.shape[0])

        res_test = model.evaluate(self.X_test, samples)
        res_test = res_test.reshape(samples, self.X_test.shape[0])

        # print("Log results to file")
        # np.savetxt(self.directory + '/bnn_test_samples.out', res_test, delimiter=',')
        # np.savetxt(self.directory + '/bnn_train_samples.out', res_train, delimiter=',')
        # np.savetxt(self.directory + '/X_train.out', self.X_train, delimiter=',')
        # np.savetxt(self.directory + '/X_test.out', self.X_test, delimiter=',')
        # np.savetxt(self.directory + '/y_train.out', self.y_train, delimiter=',')
        # np.savetxt(self.directory + '/y_test.out', self.y_test, delimiter=',')
        
        print("Generating plots")
        plt.figure(figsize=(15,13), dpi=100)
        plt.subplot(2,1,1)
        plt.plot(np.arange(self.y_train.shape[0]), self.y_train, '-b', linewidth=1.0,label='Station ' + self.res_name)
        plt.plot(np.arange(self.y_train.shape[0]), np.mean(res_train, 0).reshape(-1), 'r-', lw=2, label="Posterior mean")
        plt.fill_between(np.arange(self.y_train.shape[0]),
                         np.percentile(res_train, 5, axis=0),
                         np.percentile(res_train, 95, axis=0),
                         color = "red", alpha = 0.5, label="90% confidence region")
        plt.ylim(self.y_train.min() - 10, self.y_train.max() + 10)
        plt.legend()
        plt.title("Bayesian Neural Network(train set)")
        plt.xlabel("t")
        plt.ylabel(self.res_name)        
        plt.subplot(2,1,2)
        plt.plot(np.arange(self.y_test.shape[0]), self.y_test, '-b', linewidth=1.0,label='Station ' + self.res_name)
        plt.plot(np.arange(self.y_test.shape[0]), np.mean(res_test, 0).reshape(-1), 'r-', lw=2, label="Posterior mean")
        plt.fill_between(np.arange(self.y_test.shape[0]),
                         np.percentile(res_test, 5, axis=0),
                         np.percentile(res_test, 95, axis=0),
                         color = "red", alpha = 0.5, label="90% confidence region")
        plt.ylim(self.y_test.min() - 10, self.y_test.max() + 10)
        plt.legend()
        plt.title("Bayesian Neural Network(test set)")
        plt.xlabel("t")
        plt.ylabel(self.res_name)
        plt.savefig(self.directory+"/bnn_data_plot.png", bbox_inches='tight')
        
        print("Generating rank histograms")
        self.generate_rank_hist(self.y_test, res_test.T, self.directory+"/bnn_rank_hist_test.png" , "BNN rank histogram on test set")
        self.generate_rank_hist(self.y_train, res_train.T, self.directory+"/bnn_rank_hist_train.png" , "BNN rank histogram on train set")
        
        print("Calculating rules on the test set")
        scores_test = self.bnn_rules(model,self.X_test, self.y_test ,samples)
        self.log_scores(model_id+"_test", scores_test, self.directory + "/rules_scores.csv", "Results of BNN on test set\n")

        print("Calculating rules on the test set")
        scores_train = self.bnn_rules(model,self.X_train, self.y_train ,samples)
        self.log_scores(model_id+"_train", scores_train, self.directory + "/rules_scores_train.csv", "Results of BNN on train set\n")

        print("Calcualting feature importance on the test set")
        feature_imp = self.calculate_feature_imp(self.X_test, lambda X: self.bnn_rules(model, X, self.y_test, samples), scores_test)
        self.log_feature_importance(self.directory+"/feature_importance.csv", feature_imp, model_id+"_test")

        print("Calcualting feature importance on the train set")
        feature_imp = self.calculate_feature_imp(self.X_train, lambda X: self.bnn_rules(model, X, self.y_train, samples), scores_train)
        self.log_feature_importance(self.directory+"/feature_importance_train.csv", feature_imp, model_id+"_test")
        
        


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
        sigmas = res.std(axis=1)        

        # np.savetxt(self.directory+"/empirical_result.txt", res)

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
        log_scores = -np.log(np.array([norm.pdf(y, loc=mus[j], scale=sigmas[j]) for j, y in enumerate(self.y_test)]))
        crps_scores = np.array([ ps.crps_gaussian(y, mu=mus[j], sig=sigmas[j]) for j, y in enumerate(self.y_test)])
        dss_scores = np.array([sc.dss_norm(y, loc=mus[j], scale=sigmas[j]) for j, y in enumerate(self.y_test)])

        scores = dict()
        scores['CRPS'] = crps_scores.mean()
        scores['LS'] = log_scores.mean()
        scores['DSS'] = dss_scores.mean()
        
        self.log_scores("empirical", scores, self.directory + "/rules_scores.csv", "Results of Empirical on test set\n")

        
        self.generate_rank_hist(self.y_test, res, self.directory+"/empirical_rank_hist_test.png" , "Empirical model rank histogram on test set")
                        

            
        

        




def main():
    pass






    

if __name__ == '__main__':
    main()
