#!/home/arnaud/anaconda3/bin/python3


import os, sys, argparse
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import utils as ut
import json, re
from scipy import stats
from collections import defaultdict
import collections





class ResSelector:

    def __init__(self, main_folders):
        self.folders = main_folders
        

    def isStation(self,folder, station ):
        desc_file_name = os.path.join(folder, "desc_file.txt")
        data = None
        with open(desc_file_name,'r') as desc_file:
            data = desc_file.read()
        reg = "\s*Target station\s*:?\s*(\w*)\s*"
        m = re.search(reg, data)
        if m.group(1) == station:
            return True
        else:
            return False
            

    def isValue(self,folder, value):
        desc_file_name = os.path.join(folder, "desc_file.txt")
        data = None
        with open(desc_file_name,'r') as desc_file:
            data = desc_file.read()
            reg = "\s*Input value\s*:\s*(\w*)\s*"
        m = re.search(reg, data)
        if m.group(1) == value:
            return True
        else:
            return False
        pass


    def isLUBW(self,folder):
        desc_file_name = os.path.join(folder, "desc_file.txt")
        data = None
        with open(desc_file_name,'r') as desc_file:
            data = desc_file.read()
        reg = "\s*Taking LU BW as feature\s*:\s*(\w*)\s*"
        m = re.search(reg, data)
        if m.group(1) == "True":
            return True
        else:
            return False
        pass


    def importance(self, station, lu_bw, value, rule, test=True, limit=10):
        res = dict()
        for main_folder in self.folders:
            sub_folders = [os.path.join(main_folder, f) for f in os.listdir(main_folder)
                  if os.path.isdir(os.path.join(main_folder, f))]


            
            fold_filt = filter(lambda fold: self.isStation(fold, station) ,sub_folders)
            fold_filt = filter(lambda fold: self.isValue(fold, value) ,fold_filt)
            fold_filt = list(filter(lambda fold: lu_bw == self.isLUBW(fold) ,fold_filt))

            
            if len(fold_filt) == 0:
                print("Something is wrong")
                print(fold_filt)
                return None

            
            for folder in fold_filt:
                df = None
                if test:
                    df =  pd.read_csv(os.path.join(folder, "feature_importance.csv"), sep=';')
                else:
                    df =  pd.read_csv(os.path.join(folder, "feature_importance_train.csv"), sep=';')

                # print(df)
                df = df.loc[ df['rule'] == rule]
                df = df.sort_values(["feature"], ascending=False)
                model_name = df.iloc[0]["Model"]
                df = df.iloc[0 : ][["feature","diff_value"]]
                res[model_name] = df.to_dict('list')
        return collections.OrderedDict(sorted(res.items()))
        # return res
            
        

    def query(self,station, lu_bw, value, rule, test=True):
        res = dict()
        for main_folder in self.folders:
            sub_folders = [os.path.join(main_folder, f) for f in os.listdir(main_folder)
                  if os.path.isdir(os.path.join(main_folder, f))]


            
            fold_filt = filter(lambda fold: self.isStation(fold, station) ,sub_folders)
            fold_filt = filter(lambda fold: self.isValue(fold, value) ,fold_filt)
            fold_filt = list(filter(lambda fold: lu_bw == self.isLUBW(fold) ,fold_filt))

            
            if len(fold_filt) == 0:
                print("Something is wrong")
                print(fold_filt)
                return None

            # folder = fold_filt[0]

            for folder in fold_filt:
                # print("Proc", folder)
                df = None
                if test:
                    df =  pd.read_csv(os.path.join(folder, "rules_scores.csv"), sep=';')
                else:
                    df =  pd.read_csv(os.path.join(folder, "rules_scores_train.csv"), sep=';')

                value_model = df.iloc[0][rule]
                value_emp = df.iloc[1][rule]
                model_name = df.iloc[0]["model_id"]
                res[model_name] = value_model
                if "Empirical" not in res.keys():
                   res["Empirical"] = value_emp

        # print(collections.OrderedDict(sorted(res.items())))
        return collections.OrderedDict(sorted(res.items()))
        # return res


    def query_l(self,station, lu_bw, value, rule, test=True):

        res = dict()
        for main_folder in self.folders:
            sub_folders = [os.path.join(main_folder, f) for f in os.listdir(main_folder)
                  if os.path.isdir(os.path.join(main_folder, f))]


            
            fold_filt = filter(lambda fold: self.isStation(fold, station) ,sub_folders)
            fold_filt = filter(lambda fold: self.isValue(fold, value) ,fold_filt)
            fold_filt = list(filter(lambda fold: lu_bw == self.isLUBW(fold) ,fold_filt))

            
            if len(fold_filt) == 0:
                print("Something is wrong")
                print(fold_filt)
                return None

            # folder = fold_filt[0]

            for folder in fold_filt:
                # print("Proc", folder)
                df =  pd.read_csv(os.path.join(folder, "rules_scores_l.csv"), sep=';')

                models = df["model_id"].unique()
                for model in models:
                    value_l = df[df.model_id == model][rule]
                    model_capt = model.capitalize() if "empi" in model else model
                    res[model_capt] = value_l.values

        # print(collections.OrderedDict(sorted(res.items())).keys())
        return collections.OrderedDict(sorted(res.items()))
    
        # return res



def basic_res(sec, dest):
    stations = ['SBC', 'SAKP', 'SNTR']
    values = ['P1', 'P2']
    lu_bw  = [True, False]
    rules = ['CRPS', 'LS', 'DSS']

    for rule in rules:
        plt.figure(figsize=(10,12), dpi=100)
        i = 1
        

        for stat in stations:
            for val in values:
                ax =plt.subplot(3,2,i)
                i = i + 1
                n_groups = 2
                index = np.arange(n_groups)
                bar_width = 0.1
                opacity = 0.75

                with_lu_bw  = sec.query(stat, True, val, rule)
                without_lu_bw  = sec.query(stat, False, val, rule)
                if with_lu_bw is None or without_lu_bw is None:
                    continue

                
                for j,mod in enumerate(with_lu_bw.keys()):
                    plt.bar(index + j*bar_width, 
                            [with_lu_bw[mod], without_lu_bw[mod]],
                            bar_width,
                            alpha=opacity,
                            label=mod)                



                y_factor = 1.7
                plt.xlim(min(index)-bar_width*4, max(index)+bar_width*4)
                plt.ylim([0, max(
                    [val for key, val in with_lu_bw.items()] + [val for key, val in without_lu_bw.items()]
                )*y_factor])


                ax.set_xticks(np.linspace(min(index)-bar_width*4, max(index)+bar_width*4, 15), minor=True)
                ax.set_yticks(np.linspace(0, max(
                    [val for key, val in with_lu_bw.items()] + [val for key, val in without_lu_bw.items()]
                )*y_factor, 7), minor=True)
                plt.grid(which='minor')
                plt.grid(which='minor', alpha=0.5)
                
                plt.xticks(index + bar_width, ('With LU BW', 'Without LU BW'))
                plt.legend(loc='upper left')
                val_t = "PM10" if val is "P1" else "PM2.5"
                plt.title("Station: "+ stat + ", Predicted value:" + val_t)
                plt.ylabel(rule)

                
                plt.tight_layout()

        plt.savefig(os.path.join(dest, "results_plot_"+rule+".png"))      
            



def translate(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
 
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)
        
def feat_importance(sec, dest):
    stations = ['SBC', 'SAKP', 'SNTR']
    values = ['P1', 'P2']
    lu_bws  = [True, False]
    rules = ['CRPS']

 
    for rule in rules:
        for stat in stations:
            for val in values:
                plt.figure(figsize=(10,12), dpi=100)
                i = 1
                for lu_bw in lu_bws:
                
                    n_groups = 10
                    data = sec.importance(stat, lu_bw, val, rule, limit=n_groups)
                    n_groups = len(next(iter(data.values()))["feature"])

                    # print(data)
                    
                    index = np.arange(n_groups)
                    bar_width = 0.2
                    opacity = 0.75

                    ax =plt.subplot(2,1,i)
                    i = i +1
                    
                    for j,mod in enumerate(data.keys()):
                        max_d = max(data[mod]["diff_value"])
                        min_d = min(data[mod]["diff_value"])

                        # map_l = 0 if min_d > 0 else 10
                        map_l = 0
                        map_u = 100

                        vals = list(map(lambda x: translate(x,min_d,max_d,map_l,map_u), data[mod]["diff_value"]))
                        data[mod]["diff_value"] = vals
                        plt.bar(index + j*bar_width, 
                                vals,
                                bar_width,
                                alpha=opacity,
                                label=mod)

                        
                    # for vla in data.values():
                    #     print(vla)


                    labs = next(iter(data.values()))["feature"]
                    labs = list(map(lambda l: l.replace("P1", "PM10").replace("P2", "PM2.5") , labs))
                    plt.xticks(rotation=45)
                    plt.xticks(index + bar_width, labs)
                    plt.legend()
                    val_t = "PM10" if val is "P1" else "PM2.5"
                    if lu_bw:
                        plt.title("Feature importance. Predicted value - " + val_t + ", Station - " + stat + ", with LUBW")
                    else:
                        plt.title("Feature importance. Predicted value - " + val_t + ", Station - " + stat + ", without LUBW")
                    
                    plt.ylabel("Avrg. diff. in " + rule)
                
                    y_factor = 1.5
                    plt.xlim(min(index)-bar_width*2, max(index)+bar_width*2)
                    plt.ylim([
                        min([0] + np.array([vals["diff_value"] for mod, vals in data.items()]).flatten()),
                        max(np.array([vals["diff_value"] for mod, vals in data.items()]).flatten())*y_factor])


                    
                    ax.set_xticks(np.linspace(min(index)-bar_width*2, max(index)+bar_width*2, 15), minor=True)
                    ax.set_yticks(np.linspace(min([0] + np.array([vals["diff_value"] for mod, vals in data.items()]).flatten()),
                                              max(np.array([vals["diff_value"] for mod, vals in data.items()]).flatten())*y_factor,
                                              7), minor=True)
                    plt.grid(which='minor')
                    plt.grid(which='minor', alpha=0.5)
                    ax.axhline(y=0, color='k', linewidth=0.7)
                    
                plt.tight_layout()
                plt.savefig(os.path.join(dest, "feature_importance_"+rule+"_"+stat+"_"+val+".png"))


def gen_tables(sec, dest):
    latex_table = "\\captionsetup{{width=0.7\\linewidth,justification=raggedright}}\n\
    \\begin{{tabular}}{{c V{{2.6}}c V{{0.3}}cc||cc||cc||cc||cc||cc}} \n\
    \hline \n\
    \hline \n\
    && \multicolumn{{2}}{{c||}}{{SBC PM10}}& \multicolumn{{2}}{{c||}}{{SBC PM2.5}}& \multicolumn{{2}}{{c||}}{{SNTR PM10}}& \multicolumn{{2}}{{c||}}{{SNTR PM2.5}}&\multicolumn{{2}}{{c||}}{{SAKP PM10}}& \multicolumn{{2}}{{c}}{{SAKP PM2.5}}\\\\ \n\
    \Xhline{{2.3\\arrayrulewidth}} \n\
    &&with&without&with&without&with&without&with&without&with&without&with&without\\\\ \n\
    &&LUBW&LUBW&LUBW&LUBW&LUBW&LUBW&LUBW&LUBW&LUBW&LUBW&LUBW&LUBW\\\\ \n\
    \Xhline{{2.6\\arrayrulewidth}} \n\
    CRPS&&&&&&&&&&&\\\\ \n\
    &{m1}&{m1_d[CRPS][SBC][P1][True]}&{m1_d[CRPS][SBC][P1][False]}&{m1_d[CRPS][SBC][P2][True]}&{m1_d[CRPS][SBC][P2][False]}&{m1_d[CRPS][SNTR][P1][True]}&{m1_d[CRPS][SNTR][P1][False]}&{m1_d[CRPS][SNTR][P2][True]}&{m1_d[CRPS][SNTR][P2][False]}&{m1_d[CRPS][SAKP][P1][True]}&{m1_d[CRPS][SAKP][P1][False]}&{m1_d[CRPS][SAKP][P2][True]}&{m1_d[CRPS][SAKP][P2][False]}\\\\ \n\
    &{m2}&{m2_d[CRPS][SBC][P1][True]}&{m2_d[CRPS][SBC][P1][False]}&{m2_d[CRPS][SBC][P2][True]}&{m2_d[CRPS][SBC][P2][False]}&{m2_d[CRPS][SNTR][P1][True]}&{m2_d[CRPS][SNTR][P1][False]}&{m2_d[CRPS][SNTR][P2][True]}&{m2_d[CRPS][SNTR][P2][False]}&{m2_d[CRPS][SAKP][P1][True]}&{m2_d[CRPS][SAKP][P1][False]}&{m2_d[CRPS][SAKP][P2][True]}&{m2_d[CRPS][SAKP][P2][False]}\\\\ \n\
    &{m3}&{m3_d[CRPS][SBC][P1][True]}&{m3_d[CRPS][SBC][P1][False]}&{m3_d[CRPS][SBC][P2][True]}&{m3_d[CRPS][SBC][P2][False]}&{m3_d[CRPS][SNTR][P1][True]}&{m3_d[CRPS][SNTR][P1][False]}&{m3_d[CRPS][SNTR][P2][True]}&{m3_d[CRPS][SNTR][P2][False]}&{m3_d[CRPS][SAKP][P1][True]}&{m3_d[CRPS][SAKP][P1][False]}&{m3_d[CRPS][SAKP][P2][True]}&{m3_d[CRPS][SAKP][P2][False]}\\\\ \n\
    &{m4}&{m4_d[CRPS][SBC][P1][True]}&{m4_d[CRPS][SBC][P1][False]}&{m4_d[CRPS][SBC][P2][True]}&{m4_d[CRPS][SBC][P2][False]}&{m4_d[CRPS][SNTR][P1][True]}&{m4_d[CRPS][SNTR][P1][False]}&{m4_d[CRPS][SNTR][P2][True]}&{m4_d[CRPS][SNTR][P2][False]}&{m4_d[CRPS][SAKP][P1][True]}&{m4_d[CRPS][SAKP][P1][False]}&{m4_d[CRPS][SAKP][P2][True]}&{m4_d[CRPS][SAKP][P2][False]}\\\\ \n\
    \Xhline{{2.6\\arrayrulewidth}} \n\
    LS&&&&&&&&&&&\\\\ \n\
    &{m1}&{m1_d[LS][SBC][P1][True]}&{m1_d[LS][SBC][P1][False]}&{m1_d[LS][SBC][P2][True]}&{m1_d[LS][SBC][P2][False]}&{m1_d[LS][SNTR][P1][True]}&{m1_d[LS][SNTR][P1][False]}&{m1_d[LS][SNTR][P2][True]}&{m1_d[LS][SNTR][P2][False]}&{m1_d[LS][SAKP][P1][True]}&{m1_d[LS][SAKP][P1][False]}&{m1_d[LS][SAKP][P2][True]}&{m1_d[LS][SAKP][P2][False]}\\\\ \n\
    &{m2}&{m2_d[LS][SBC][P1][True]}&{m2_d[LS][SBC][P1][False]}&{m2_d[LS][SBC][P2][True]}&{m2_d[LS][SBC][P2][False]}&{m2_d[LS][SNTR][P1][True]}&{m2_d[LS][SNTR][P1][False]}&{m2_d[LS][SNTR][P2][True]}&{m2_d[LS][SNTR][P2][False]}&{m2_d[LS][SAKP][P1][True]}&{m2_d[LS][SAKP][P1][False]}&{m2_d[LS][SAKP][P2][True]}&{m2_d[LS][SAKP][P2][False]}\\\\ \n\
    &{m3}&{m3_d[LS][SBC][P1][True]}&{m3_d[LS][SBC][P1][False]}&{m3_d[LS][SBC][P2][True]}&{m3_d[LS][SBC][P2][False]}&{m3_d[LS][SNTR][P1][True]}&{m3_d[LS][SNTR][P1][False]}&{m3_d[LS][SNTR][P2][True]}&{m3_d[LS][SNTR][P2][False]}&{m3_d[LS][SAKP][P1][True]}&{m3_d[LS][SAKP][P1][False]}&{m3_d[LS][SAKP][P2][True]}&{m3_d[LS][SAKP][P2][False]}\\\\ \n\
    &{m4}&{m4_d[LS][SBC][P1][True]}&{m4_d[LS][SBC][P1][False]}&{m4_d[LS][SBC][P2][True]}&{m4_d[LS][SBC][P2][False]}&{m4_d[LS][SNTR][P1][True]}&{m4_d[LS][SNTR][P1][False]}&{m4_d[LS][SNTR][P2][True]}&{m4_d[LS][SNTR][P2][False]}&{m4_d[LS][SAKP][P1][True]}&{m4_d[LS][SAKP][P1][False]}&{m4_d[LS][SAKP][P2][True]}&{m4_d[LS][SAKP][P2][False]}\\\\ \n\
    \Xhline{{2.6\\arrayrulewidth}} \n\
    DSS&&&&&&&&&&&\\\\ \n\
        &{m1}&{m1_d[DSS][SBC][P1][True]}&{m1_d[DSS][SBC][P1][False]}&{m1_d[DSS][SBC][P2][True]}&{m1_d[DSS][SBC][P2][False]}&{m1_d[DSS][SNTR][P1][True]}&{m1_d[DSS][SNTR][P1][False]}&{m1_d[DSS][SNTR][P2][True]}&{m1_d[DSS][SNTR][P2][False]}&{m1_d[DSS][SAKP][P1][True]}&{m1_d[DSS][SAKP][P1][False]}&{m1_d[DSS][SAKP][P2][True]}&{m1_d[DSS][SAKP][P2][False]}\\\\ \n\
    &{m2}&{m2_d[DSS][SBC][P1][True]}&{m2_d[DSS][SBC][P1][False]}&{m2_d[DSS][SBC][P2][True]}&{m2_d[DSS][SBC][P2][False]}&{m2_d[DSS][SNTR][P1][True]}&{m2_d[DSS][SNTR][P1][False]}&{m2_d[DSS][SNTR][P2][True]}&{m2_d[DSS][SNTR][P2][False]}&{m2_d[DSS][SAKP][P1][True]}&{m2_d[DSS][SAKP][P1][False]}&{m2_d[DSS][SAKP][P2][True]}&{m2_d[DSS][SAKP][P2][False]}\\\\ \n\
    &{m3}&{m3_d[DSS][SBC][P1][True]}&{m3_d[DSS][SBC][P1][False]}&{m3_d[DSS][SBC][P2][True]}&{m3_d[DSS][SBC][P2][False]}&{m3_d[DSS][SNTR][P1][True]}&{m3_d[DSS][SNTR][P1][False]}&{m3_d[DSS][SNTR][P2][True]}&{m3_d[DSS][SNTR][P2][False]}&{m3_d[DSS][SAKP][P1][True]}&{m3_d[DSS][SAKP][P1][False]}&{m3_d[DSS][SAKP][P2][True]}&{m3_d[DSS][SAKP][P2][False]}\\\\ \n\
    &{m4}&{m4_d[DSS][SBC][P1][True]}&{m4_d[DSS][SBC][P1][False]}&{m4_d[DSS][SBC][P2][True]}&{m4_d[DSS][SBC][P2][False]}&{m4_d[DSS][SNTR][P1][True]}&{m4_d[DSS][SNTR][P1][False]}&{m4_d[DSS][SNTR][P2][True]}&{m4_d[DSS][SNTR][P2][False]}&{m4_d[DSS][SAKP][P1][True]}&{m4_d[DSS][SAKP][P1][False]}&{m4_d[DSS][SAKP][P2][True]}&{m4_d[DSS][SAKP][P2][False]}\\\\ \n\
    \Xhline{{2.6\\arrayrulewidth}} \n\
  \end{{tabular}} \n\
  \\captionof{{table}}{{{m1} : \\texttt{{{m1_n}}} \\\\ {m2} : \\texttt{{{m2_n}}} \\\\ {m3} : \\texttt{{{m3_n}}} \\\\ {m4} : \\texttt{{{m4_n}}} }}"

    

    stations = ['SBC', 'SAKP', 'SNTR']
    values = ['P1', 'P2']
    lu_bws  = [True, False]
    rules = ['CRPS', 'LS', 'DSS']

    formater = dict()
    mod_names = defaultdict(dict)
    real_name = defaultdict(dict)
    i=1
    mdn_i=1
    bnn_i=1
    for rule in rules:
        for stat in stations:
            for val in values:
                for lu_bw in lu_bws:
                    res = sec.query(stat, lu_bw, val, rule)
                    for mod, value in res.items():
                        if mod not in mod_names.keys():
                            if "mdn" in mod:
                                mod_names["m"+str(i)] = "MDN$_"+str(mdn_i)+"$"
                                mdn_i = mdn_i + 1
                            elif "bnn" in mod:
                                mod_names["m"+str(i)] = "BNN$_"+str(bnn_i)+"$"
                                bnn_i = bnn_i + 1
                            else:
                                mod_names["m"+str(i)] = "Emp."
                            mod_names[mod] = "m"+str(i)                            
                            i = i + 1 
                        mod_name = mod_names[mod]
                        mod_names[mod_name+"_n"] = mod.replace("_","\\_")
                        formater.setdefault(
                            mod_name+"_d" ,{}).setdefault(
                                rule,{}).setdefault(
                                    stat, {}).setdefault(
                                        val, {}).setdefault(str(lu_bw), round(value,3))

    
    with open(os.path.join(dest, "latex_res_table.tex"),"w") as tex_file:
        tex_file.write(latex_table.format_map({**formater, **mod_names}))
    #print(latex_table.format_map({**formater, **mod_names}))
    # reg = "(?<={})[\S\s]*(?={})".format(beg, end)



def dm_test(p1, p2):
    p1_m = p1.mean()
    p2_m = p2.mean()
    n = float(p1.shape[0])
    sig = ((p1 - p2)**2).mean()
    t = ((np.sqrt(n))*(p1_m - p2_m))/np.sqrt(sig)
    p = stats.norm.pdf(abs(t))
    return (t,p)


def predictive_check(sec, dest):

    latex_table="\\begin{{tabular}}{{c|c|cccc||cccc}} \n\
  \\hline \n\
  \\hline \n\
  &&\\multicolumn{{3}}{{c}}{{PM10 with LUBW}} && \\multicolumn{{3}}{{c}}{{PM10 without LUBW}} \\\\ \n\
  \\hline \n\
  &&{m1}&{m2}&{m3}&{m4}&{m1}&{m2}&{m3}&{m4} \\\\ \n\
  \\hline \n\
  \\hline \n\
  \\textbf{{SBC}}&&&&&&& \\\\ \n\
  PM10&{m1}&{m1_d[m1_d][SBC][P1][True][True]}&{m1_d[m2_d][SBC][P1][True][True]}&{m1_d[m3_d][SBC][P1][True][True]}&{m1_d[m4_d][SBC][P1][True][True]}&{m1_d[m1_d][SBC][P1][False][True]}&{m1_d[m2_d][SBC][P1][False][True]}&{m1_d[m3_d][SBC][P1][False][True]}&{m1_d[m4_d][SBC][P1][False][True]} \\\\ \n\
  with&{m2}&{m2_d[m1_d][SBC][P1][True][True]}&{m2_d[m2_d][SBC][P1][True][True]}&{m2_d[m3_d][SBC][P1][True][True]}&{m2_d[m4_d][SBC][P1][True][True]}&{m2_d[m1_d][SBC][P1][False][True]}&{m2_d[m2_d][SBC][P1][False][True]}&{m2_d[m3_d][SBC][P1][False][True]}&{m2_d[m4_d][SBC][P1][False][True]} \\\\ \n\
  LUBW&{m3}&{m3_d[m1_d][SBC][P1][True][True]}&{m3_d[m2_d][SBC][P1][True][True]}&{m3_d[m3_d][SBC][P1][True][True]}&{m3_d[m4_d][SBC][P1][True][True]}&{m3_d[m1_d][SBC][P1][False][True]}&{m3_d[m2_d][SBC][P1][False][True]}&{m3_d[m3_d][SBC][P1][False][True]}&{m3_d[m4_d][SBC][P1][False][True]} \\\\ \n\
  &{m4}&{m4_d[m1_d][SBC][P1][True][True]}&{m4_d[m2_d][SBC][P1][True][True]}&{m4_d[m3_d][SBC][P1][True][True]}&{m4_d[m4_d][SBC][P1][True][True]}&{m4_d[m1_d][SBC][P1][False][True]}&{m3_d[m2_d][SBC][P1][False][True]}&{m4_d[m3_d][SBC][P1][False][True]}&{m4_d[m4_d][SBC][P1][False][True]} \\\\ \n\
  \\hline     \n\
  PM10&{m1}&{m1_d[m1_d][SBC][P1][True][False]}&{m1_d[m2_d][SBC][P1][True][False]}&{m1_d[m3_d][SBC][P1][True][False]}&{m1_d[m4_d][SBC][P1][True][False]}&{m1_d[m1_d][SBC][P1][False][False]}&{m1_d[m2_d][SBC][P1][False][False]}&{m1_d[m3_d][SBC][P1][False][False]}&{m1_d[m4_d][SBC][P1][False][False]} \\\\ \n\
  without&{m2}&{m2_d[m1_d][SBC][P1][True][False]}&{m2_d[m2_d][SBC][P1][True][False]}&{m2_d[m3_d][SBC][P1][True][False]}&{m2_d[m4_d][SBC][P1][True][False]}&{m2_d[m1_d][SBC][P1][False][False]}&{m2_d[m2_d][SBC][P1][False][False]}&{m2_d[m3_d][SBC][P1][False][False]}&{m2_d[m4_d][SBC][P1][False][False]} \\\\ \n\
  LUBW&{m3}&{m3_d[m1_d][SBC][P1][True][False]}&{m3_d[m2_d][SBC][P1][True][False]}&{m3_d[m3_d][SBC][P1][True][False]}&{m3_d[m4_d][SBC][P1][True][False]}&{m3_d[m1_d][SBC][P1][False][False]}&{m3_d[m2_d][SBC][P1][False][False]}&{m3_d[m3_d][SBC][P1][False][False]}&{m3_d[m4_d][SBC][P1][False][False]} \\\\ \n\
  &{m4}&{m4_d[m1_d][SBC][P1][True][False]}&{m4_d[m2_d][SBC][P1][True][False]}&{m4_d[m3_d][SBC][P1][True][False]}&{m4_d[m4_d][SBC][P1][True][False]}&{m4_d[m1_d][SBC][P1][False][False]}&{m4_d[m2_d][SBC][P1][False][False]}&{m4_d[m3_d][SBC][P1][False][False]}&{m4_d[m4_d][SBC][P1][False][False]} \\\\ \n\
  \\hline     \n\
  \\hline     \n\
\\textbf{{SNTR}}&&&&&&& \\\\ \n\
  PM10&{m1}&{m1_d[m1_d][SNTR][P1][True][True]}&{m1_d[m2_d][SNTR][P1][True][True]}&{m1_d[m3_d][SNTR][P1][True][True]}&{m1_d[m4_d][SNTR][P1][True][True]}&{m1_d[m1_d][SNTR][P1][False][True]}&{m1_d[m2_d][SNTR][P1][False][True]}&{m1_d[m3_d][SNTR][P1][False][True]}&{m1_d[m4_d][SNTR][P1][False][True]} \\\\ \n\
  with&{m2}&{m2_d[m1_d][SNTR][P1][True][True]}&{m2_d[m2_d][SNTR][P1][True][True]}&{m2_d[m3_d][SNTR][P1][True][True]}&{m2_d[m4_d][SNTR][P1][True][True]}&{m2_d[m1_d][SNTR][P1][False][True]}&{m2_d[m2_d][SNTR][P1][False][True]}&{m2_d[m3_d][SNTR][P1][False][True]}&{m2_d[m4_d][SNTR][P1][False][True]} \\\\ \n\
  LUBW&{m3}&{m3_d[m1_d][SNTR][P1][True][True]}&{m3_d[m2_d][SNTR][P1][True][True]}&{m3_d[m3_d][SNTR][P1][True][True]}&{m3_d[m4_d][SNTR][P1][True][True]}&{m3_d[m1_d][SNTR][P1][False][True]}&{m3_d[m2_d][SNTR][P1][False][True]}&{m3_d[m3_d][SNTR][P1][False][True]}&{m3_d[m4_d][SNTR][P1][False][True]} \\\\ \n\
  &{m4}&{m4_d[m1_d][SNTR][P1][True][True]}&{m4_d[m2_d][SNTR][P1][True][True]}&{m4_d[m3_d][SNTR][P1][True][True]}&{m4_d[m4_d][SNTR][P1][True][True]}&{m4_d[m1_d][SNTR][P1][False][True]}&{m3_d[m2_d][SNTR][P1][False][True]}&{m4_d[m3_d][SNTR][P1][False][True]}&{m4_d[m4_d][SNTR][P1][False][True]} \\\\ \n\
  \\hline     \n\
  PM10&{m1}&{m1_d[m1_d][SNTR][P1][True][False]}&{m1_d[m2_d][SNTR][P1][True][False]}&{m1_d[m3_d][SNTR][P1][True][False]}&{m1_d[m4_d][SNTR][P1][True][False]}&{m1_d[m1_d][SNTR][P1][False][False]}&{m1_d[m2_d][SNTR][P1][False][False]}&{m1_d[m3_d][SNTR][P1][False][False]}&{m1_d[m4_d][SNTR][P1][False][False]} \\\\ \n\
  without&{m2}&{m2_d[m1_d][SNTR][P1][True][False]}&{m2_d[m2_d][SNTR][P1][True][False]}&{m2_d[m3_d][SNTR][P1][True][False]}&{m2_d[m4_d][SNTR][P1][True][False]}&{m2_d[m1_d][SNTR][P1][False][False]}&{m2_d[m2_d][SNTR][P1][False][False]}&{m2_d[m3_d][SNTR][P1][False][False]}&{m2_d[m4_d][SNTR][P1][False][False]} \\\\ \n\
  LUBW&{m3}&{m3_d[m1_d][SNTR][P1][True][False]}&{m3_d[m2_d][SNTR][P1][True][False]}&{m3_d[m3_d][SNTR][P1][True][False]}&{m3_d[m4_d][SNTR][P1][True][False]}&{m3_d[m1_d][SNTR][P1][False][False]}&{m3_d[m2_d][SNTR][P1][False][False]}&{m3_d[m3_d][SNTR][P1][False][False]}&{m3_d[m4_d][SNTR][P1][False][False]} \\\\ \n\
  &{m4}&{m4_d[m1_d][SNTR][P1][True][False]}&{m4_d[m2_d][SNTR][P1][True][False]}&{m4_d[m3_d][SNTR][P1][True][False]}&{m4_d[m4_d][SNTR][P1][True][False]}&{m4_d[m1_d][SNTR][P1][False][False]}&{m4_d[m2_d][SNTR][P1][False][False]}&{m4_d[m3_d][SNTR][P1][False][False]}&{m4_d[m4_d][SNTR][P1][False][False]} \\\\ \n\
  \\hline     \n\
  \\hline     \n\
\\textbf{{SAKP}}&&&&&&& \\\\ \n\
  PM10&{m1}&{m1_d[m1_d][SAKP][P1][True][True]}&{m1_d[m2_d][SAKP][P1][True][True]}&{m1_d[m3_d][SAKP][P1][True][True]}&{m1_d[m4_d][SAKP][P1][True][True]}&{m1_d[m1_d][SAKP][P1][False][True]}&{m1_d[m2_d][SAKP][P1][False][True]}&{m1_d[m3_d][SAKP][P1][False][True]}&{m1_d[m4_d][SAKP][P1][False][True]} \\\\ \n\
  with&{m2}&{m2_d[m1_d][SAKP][P1][True][True]}&{m2_d[m2_d][SAKP][P1][True][True]}&{m2_d[m3_d][SAKP][P1][True][True]}&{m2_d[m4_d][SAKP][P1][True][True]}&{m2_d[m1_d][SAKP][P1][False][True]}&{m2_d[m2_d][SAKP][P1][False][True]}&{m2_d[m3_d][SAKP][P1][False][True]}&{m2_d[m4_d][SAKP][P1][False][True]} \\\\ \n\
  LUBW&{m3}&{m3_d[m1_d][SAKP][P1][True][True]}&{m3_d[m2_d][SAKP][P1][True][True]}&{m3_d[m3_d][SAKP][P1][True][True]}&{m3_d[m4_d][SAKP][P1][True][True]}&{m3_d[m1_d][SAKP][P1][False][True]}&{m3_d[m2_d][SAKP][P1][False][True]}&{m3_d[m3_d][SAKP][P1][False][True]}&{m3_d[m4_d][SAKP][P1][False][True]} \\\\ \n\
  &{m4}&{m4_d[m1_d][SAKP][P1][True][True]}&{m4_d[m2_d][SAKP][P1][True][True]}&{m4_d[m3_d][SAKP][P1][True][True]}&{m4_d[m4_d][SAKP][P1][True][True]}&{m4_d[m1_d][SAKP][P1][False][True]}&{m3_d[m2_d][SAKP][P1][False][True]}&{m4_d[m3_d][SAKP][P1][False][True]}&{m4_d[m4_d][SAKP][P1][False][True]} \\\\ \n\
  \\hline     \n\
  PM10&{m1}&{m1_d[m1_d][SAKP][P1][True][False]}&{m1_d[m2_d][SAKP][P1][True][False]}&{m1_d[m3_d][SAKP][P1][True][False]}&{m1_d[m4_d][SAKP][P1][True][False]}&{m1_d[m1_d][SAKP][P1][False][False]}&{m1_d[m2_d][SAKP][P1][False][False]}&{m1_d[m3_d][SAKP][P1][False][False]}&{m1_d[m4_d][SAKP][P1][False][False]} \\\\ \n\
  without&{m2}&{m2_d[m1_d][SAKP][P1][True][False]}&{m2_d[m2_d][SAKP][P1][True][False]}&{m2_d[m3_d][SAKP][P1][True][False]}&{m2_d[m4_d][SAKP][P1][True][False]}&{m2_d[m1_d][SAKP][P1][False][False]}&{m2_d[m2_d][SAKP][P1][False][False]}&{m2_d[m3_d][SAKP][P1][False][False]}&{m2_d[m4_d][SAKP][P1][False][False]} \\\\ \n\
  LUBW&{m3}&{m3_d[m1_d][SAKP][P1][True][False]}&{m3_d[m2_d][SAKP][P1][True][False]}&{m3_d[m3_d][SAKP][P1][True][False]}&{m3_d[m4_d][SAKP][P1][True][False]}&{m3_d[m1_d][SAKP][P1][False][False]}&{m3_d[m2_d][SAKP][P1][False][False]}&{m3_d[m3_d][SAKP][P1][False][False]}&{m3_d[m4_d][SAKP][P1][False][False]} \\\\ \n\
  &{m4}&{m4_d[m1_d][SAKP][P1][True][False]}&{m4_d[m2_d][SAKP][P1][True][False]}&{m4_d[m3_d][SAKP][P1][True][False]}&{m4_d[m4_d][SAKP][P1][True][False]}&{m4_d[m1_d][SAKP][P1][False][False]}&{m4_d[m2_d][SAKP][P1][False][False]}&{m4_d[m3_d][SAKP][P1][False][False]}&{m4_d[m4_d][SAKP][P1][False][False]} \\\\ \n\
  \\hline     \n\
  \\hline     \n\
\end{{tabular}} \n\
\\captionof{{table}}{{{m1} : \\texttt{{{m1_n}}} \\\\ {m2} : \\texttt{{{m2_n}}} \\\\ {m3} : \\texttt{{{m3_n}}}\\\\ {m4} : \\texttt{{{m4_n}}} }}"
    
    stations = ['SBC', 'SAKP', 'SNTR']
    values = ['P1']
    lu_bws  = [True, False]
    rules = ['CRPS']
    formater = dict()
    mod_names = defaultdict(dict)
    i=1
    mdn_i=1
    bnn_i=1
    
    for rule in rules:
        for stat in stations:
            for val in values:
                for lu_bw1 in lu_bws:
                    for lu_bw2 in lu_bws:
                        
                        res1 = sec.query_l(stat, lu_bw1, val, rule)
                        res2 = sec.query_l(stat, lu_bw2, val, rule)                        
                        for mod, value in res1.items():
                            if mod not in mod_names.keys():
                                if "mdn" in mod:
                                    mod_names["m"+str(i)] = "MDN$_"+str(mdn_i)+"$"
                                    mdn_i = mdn_i + 1
                                elif "bnn" in mod:
                                    mod_names["m"+str(i)] = "BNN$_"+str(bnn_i)+"$"
                                    bnn_i = bnn_i + 1
                                else:
                                    mod_names["m"+str(i)] = "Emp."
                                    
                                mod_names[mod] = "m"+str(i)                         
                                i = i + 1 
                                mod_name = mod_names[mod]
                                mod_names[mod_name+"_n"] = mod.replace("_","\\_")
                                
                        for mod1, values1 in res1.items():
                            for mod2, values2 in res2.items():
                                test = dm_test(values1, values2)
                                if round(test[1],3) < 0.05:
                                    test = (round(test[0],2), "$^*$")
                                else:
                                    # test = (round(test[0],2), round(test[1],2))
                                    test = (round(test[0],2), "")
                                    
                                if lu_bw1 != lu_bw2:
                                    test = (float("nan"), float("nan"))

                                mod1_name = mod_names[mod1]
                                mod2_name = mod_names[mod2]
                                formater.setdefault(
                                    mod1_name+"_d" ,{}).setdefault(
                                        mod2_name+"_d",{}).setdefault(
                                            stat, {}).setdefault(
                                                val, {}).setdefault(
                                                    str(lu_bw1), {}).setdefault(
                                                        str(lu_bw2),str(test[0])+""+str(test[1]))

    # [m1_d][m1_d][SBC][P1][True]
    # res = sec.query_l("SBC",True,"P1","CRPS")
    # print(mod_names)
    # print("---------")
    # print(formater)
    # print(latex_table.format_map({**formater, **mod_names}))
    
    with open(os.path.join(dest, "latex_pred_table.tex"),"w") as tex_file:
        tex_file.write(latex_table.format_map({**formater, **mod_names}).replace("nannan","\\texttt{\\--}").replace("nan","\\texttt{\\--}"))
    
    
    

    
    pass 
    




def main():

    parser = argparse.ArgumentParser(description='Generate result plots')    

    parser.add_argument('folders', metavar='FOLDs',
                        help='The main folders to process', nargs='+')
    
    parser.add_argument('--basic-plots', dest='basic', action='store_true',
                    help='Generate the basic comparasion plots')

    parser.add_argument('--feature-imp', dest='imp', action='store_true',
                        help='Generate plots of feature importance')

    parser.add_argument('--table-tex', dest='table', action='store_true',
                        help='Generate latex table with the results')

    parser.add_argument('--pred-check', dest='check', action='store_true',
                    help='Generate latex table with predictive checks between models acording to \
the Diebold-Mariano test')

    parser.add_argument('--dest', dest='dest', action='store', default="./",
                        help='Folder for the results', required=False)

    args = parser.parse_args()
    
    

    print("Processing fodlers: ", args.folders)
    sec = ResSelector(args.folders)

    
    if not os.path.isdir(args.dest):
        os.makedirs(args.dest)
    
    if args.basic:
        print("Generating results plots")
        basic_res(sec, args.dest)

    if args.imp:
        print("Generating feature importance")
        feat_importance(sec, args.dest)

    if args.table:
        print("Generating LaTeX Table with summery results")
        gen_tables(sec, args.dest)

    if args.check:
        print("Generating predictive checks")
        predictive_check(sec, args.dest)


if __name__ == '__main__':
    main()
