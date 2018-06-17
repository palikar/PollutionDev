#!/home/arnaud/anaconda3/bin/python3.6



import sys, os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.stats import norm
import properscoring as ps
import scoring as sc



class Scorer:
    "Conveniance class for easier scoring of generated models based on\
samples drawn from them"

    def __init__(self, rules=None, min_samples=5, max_samples=2000, samples_cnt_step=5):
        ""
        self.scoring_rules = {}
        self.scoring_rules["DSS"] = sc.dss_edf_samples
        self.scoring_rules["LOG"] = sc.log_edf_samples
        self.scoring_rules["CRPS"] = sc.crps_edf_samples
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.samples_cnt_step = samples_cnt_step
        self.samplers = {}
        

    def set_sampler(self, model_id, sampler):
        "sampler must be a funtion taking and integer n and returning\
        an arraylike with size n containing samples from some\
        distrubuition that is to be evaluated"
        self.samplers[model_id] = sampler
        
                    

    def def_log_model_results_in_file(self, log_file, scoring_results, y , model_id):
        print("Logging results to file " + str(log_file))
        with open(log_file, "w") as out:
            out.write("Report for model " + str(model_id) + "\n")
            out.write("-----------------------\n")
            out.write("Scoring rules were applied to " + str(y.shape[0]) + " obseravtions\n")
            out.write("-----------------------\n")
            out.write("Final numbers: \n")
            for rule, score in scoring_results.items():
                out.write(str(rule) + " : " + str(score[-1]) + "\n")

            out.write("-----------------------\n")
            out.write("Means: \n")
            for rule, score in scoring_results.items():
                out.write(str(rule) + " : " + str(score[-1].mean()) + "\n")

            out.write("-----------------------\n")
            out.write("Max: \n")
            for rule, score in scoring_results.items():
                out.write(str(rule) + " : " + str(score[-1].max()) + "\n")

            out.write("-----------------------\n")
            out.write("Min: \n")
            for rule, score in scoring_results.items():
                out.write(str(rule) + " : " + str(score[-1].min()) + "\n")
            out.write("-----------------------\n")
            out.write("Std: \n")
            for rule, score in scoring_results.items():
                out.write(str(rule) + " : " + str(score[-1].std()) + "\n")

            out.write("-----------------------\n")


    def create_evaluation_data_frame(self, data_frame_file, scoring_results, y , model_id):
        print("Creating data drame in file " + str(data_frame_file))
        data = {}
        data["observation"] = y
        for rule, score in scoring_results.items():
            data[rule] = score[-1]
            df = pd.DataFrame(data)
            df.index.name = "num"
            df.to_csv(data_frame_file, sep=";")


    def create_evalation_plots(self, plots_path, scoring_results, y , model_id, samples_range):
        print("Ploting results in directory " + str(plots_path))
        for y_indx, y_val in enumerate(np.nditer(y)):
            plt.cla()
            plt.close()
            plt.clf()
            plt.figure(figsize=(10,7), dpi=100)
            for rule, score in scoring_results.items():
                plt.plot(samples_range, score.T[y_indx], linewidth = 1.0, label=str(rule))
                plt.title('Scoring rules for observation ' + str(y_val) + " model " + model_id)
                plt.ylabel("Rule score")
                plt.xlabel("Number of samples")
                plt.legend()
                plt.savefig(plots_path + "/scroring_"+model_id+"_"+str(y_indx)+".png", bbox_inches='tight')

            
        plt.cla()
        plt.close()
        plt.clf()
        plt.figure(figsize=(10,7), dpi=100)
        index = np.arange(y.shape[0])
        bar_width = 0.05
        opacity = 0.8
        for indx, (rule, score) in enumerate(scoring_results.items()):
            rects = plt.bar(index + indx*bar_width, score[-1], bar_width,
                    alpha=opacity,
                    label=rule
                    )
            for rect in rects:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                        str(round(height,2)),
                        ha='center', va='bottom')
        plt.xlabel('Observation')
        plt.ylabel('Score value')
        plt.title('Scores by observation for model' + model_id)
        plt.xticks(index + bar_width, y.astype(str))
        plt.legend()
            
        plt.tight_layout()
        plt.savefig(plots_path + "/scroring_"+model_id+"_per_observation.png", bbox_inches='tight')

    

    def evaluate_samples(self, y, samples, sampler):
        scoring_results = {}
        samples_range = None
        if samples is None:
            print("Using the defined sampler function")
            samples_range = range(self.min_samples, self.max_samples, self.samples_cnt_step)
            for i in samples_range:
                new_samples = sampler(i)
                for rule_name, rule_fun in self.scoring_rules.items():
                    if rule_name not in scoring_results:
                        scoring_results[rule_name] = np.array([rule_fun(y, new_samples)])
                    else:
                        scoring_results[rule_name] = np.append(scoring_results[rule_name], [rule_fun(y, new_samples)], axis=0)

        else:
            print("Using user provided samples")
            samples_range = np.array([])
            for new_samples in np.nditer(samples):
                samples_range = np.append(samples_range, len(new_samples))
                for rule_name, rule_fun in self.scoring_rules.items():
                    if rule_name not in scoring_results:
                        scoring_results[rule_name] = np.array([rule_fun(y, new_samples)])
                    else:
                        scoring_results[rule_name] = np.append(scoring_results[rule_name], [rule_fun(y, new_samples)], axis=0)

        return scoring_results, samples_range
        

    def single_model_evaluation(self,y, model_id, samples=None, plots_path=None, log_file=None, data_frame_file=None):

        if not isinstance(y, np.ndarray):
            y = np.array([y])

        scoring_results, samples_range = self.evaluate_samples(y, samples, self.samplers[model_id])
                    
            
        print("Evaluation of " + model_id + " complete")
        print("Final scores")
        for rule, score in scoring_results.items():
            print(str(rule) + " : " + str(score[-1]))

        if log_file is not None:
            self.def_log_model_results_in_file(log_file, scoring_results, y, model_id)

        if data_frame_file is not None:
            self.create_evaluation_data_frame(data_frame_file, scoring_results, y, model_id)                


        if plots_path is not None:
            self.create_evalation_plots(plots_path, scoring_results, y, model_id, samples_range)

                

    def cross_model_log_file(self, log_file, res, y , model_ids):
        with open(log_file, "w") as out:
                out.write("Logging for evaluation of models: " + str(model_ids) + "\n")
                out.write("Scores per observation\n")
                out.write("----------\n")
                for rule in self.scoring_rules:
                    out.write("Rule: " + str(rule) + "\n")
                    for model_id in model_ids:
                        out.write(model_id+" : "+str(res[model_id][0][rule][-1]) +"\n")
                    out.write("----------\n")
                    out.write("Averrage scores for all obseravtions\n")
                    for model_id in model_ids:
                        out.write(model_id+" : "+str(res[model_id][0][rule][-1].mean()) +"\n")
                    out.write("----------\n")
                    out.write("STDs of scores for all obseravtions\n")
                    for model_id in model_ids:
                        out.write(model_id+" : "+str(res[model_id][0][rule][-1].std()) +"\n")
                    out.write("----------\n")


    def cross_model_data_frame(self, data_frame_file, res, y , model_ids):
        data = {}
        data["observation"] = y
        for rule in self.scoring_rules.keys():
            for model_id in model_ids:
                data[rule+"_"+model_id] = res[model_id][0][rule][-1]
        df = pd.DataFrame(data)
        df.index.name = "num"
        df.to_csv(data_frame_file, sep=";")


    def cross_model_plots(self, plots_path, res, y , model_ids):
        for indx_y, y_val in enumerate(np.nditer(y)):
            plt.cla()
            plt.close()
            plt.clf()
            plt.figure(figsize=(12,10), dpi=100)
            for indx, rule in enumerate(self.scoring_rules):
                plt.subplot(3,1,indx+1)
                for model_id in model_ids:
                    plt.plot(res[model_id][1],res[model_id][0][rule].T[indx_y], linewidth = 1.0, label=model_id)

                plt.ylabel("Rule score")
                plt.xlabel("Number of samples")
                plt.title(rule + " for all models and observation " + str(indx_y) + " (" + str(round(float(y_val),2)) + ")")
                plt.legend()
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(plots_path + "/scroring_models_" + str(indx_y) + ".png")


        index = np.arange(len(self.scoring_rules))
        bar_width = 0.1
        opacity = 0.8
        for y_indx, y_val in enumerate(np.nditer(y)):
            plt.cla()
            plt.close()
            plt.clf()
            plt.figure(figsize=(10,7), dpi=100)
            for indx, model_id in enumerate(model_ids):

                                
                rects = plt.bar(index + indx*bar_width,
                                list(map(lambda rule_res: rule_res[-1][y_indx],res[model_id][0].values())),
                                bar_width,
                                alpha=opacity,
                                label=model_id)
                for rect in rects:
                    height = rect.get_height()
                    plt.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                             str(round(height,2)),
                             ha='center', va='bottom')
                    
            plt.xlabel('Rule')
            plt.ylabel('Score value')
            plt.title('Scores by rules per models for observation ' + str(y_indx) + " (" + str(round(float(y_val),2)) + ")" )
            plt.xticks(index + bar_width, self.scoring_rules.keys())
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(plots_path + "/cross_model_scroring_" + str(y_indx) +".png", bbox_inches='tight')
        
        
        

    
    def cross_model_evaluation(self,y, model_ids, samples=None, plots_path=None, log_file=None, data_frame_file=None ):

        if not isinstance(y, np.ndarray):
            y = np.array([y])


        res = {}
        for model_id in model_ids:
            if samples is not None and model_id in samples:
                res[model_id] = self.evaluate_samples(y, samples[model_id], None)
            elif model_id in self.samplers:
                res[model_id] = self.evaluate_samples(y, None, self.samplers[model_id])
            else:
                print("Model id must be either in the provided samples or a sampler must be added")

            print("Evaluation of " + model_id + " complete")


        
        for rule in self.scoring_rules.keys():
            print("----------")
            print("Rule: " + str(rule))
            for model_id in model_ids:
                print(model_id+" : "+str(res[model_id][0][rule][-1]))
        print("----------")



        if log_file is not None:
            self.cross_model_log_file(log_file, res,y, model_ids)

        if data_frame_file is not None:
            self.cross_model_data_frame(data_frame_file, res,y, model_ids)

        if plots_path is not None:
            self.cross_model_plots(plots_path, res,y, model_ids)
                


def main():
    scorer = Scorer(min_samples=5, max_samples=100, samples_cnt_step=5)


    scorer.set_sampler("nomral_uo_std5", lambda n: norm.rvs(size=int(n), loc=0, scale=5))
    scorer.set_sampler("nomral_u2_std7", lambda n: norm.rvs(size=int(n), loc=2, scale=7))


    scorer.cross_model_evaluation(np.array([0.1,6,15]),
                                   ["nomral_uo_std5", "nomral_u2_std7"],
                                   log_file="/home/arnaud/code/pollution/scores_log.txt",
                                   data_frame_file="/home/arnaud/code/pollution/scores_df.txt",
                                   plots_path="/home/arnaud/code/pollution/scores_plots"
                                   )

    scorer.single_model_evaluation(np.array([0.1,6,15]),
                                   "nomral_uo_std5",
                                   log_file="/home/arnaud/code/pollution/scores_log.txt",
                                   data_frame_file="/home/arnaud/code/pollution/scores_df.txt",
                                   plots_path="/home/arnaud/code/pollution/scores_plots"
                                   )


if __name__ == '__main__':
    main()
