#!/home/arnaud/anaconda3/bin/python3.6



import argparse
import sys, os
import utils as ut
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re

lu_bw_data_files_dir = None
description_files_dir = None
time_column = None
reindex_period = None
integration_type = None
integration_freq = None

def _read_config(config_data):
    global lu_bw_data_files_dir, description_files_dir, time_column, reindex_period, integration_type, integration_freq

    lu_bw_data_files_dir = os.path.expanduser(config_data["lu_bw_data_files_dir"])
    description_files_dir = os.path.expanduser(config_data["description_files_dir"])

    time_column = config_data["lu_bw_prepreocesor"]["time_column"]
    reindex_period = config_data["lu_bw_prepreocesor"]["reindex_period"]
    integration_type = config_data["lu_bw_prepreocesor"]["integration_type"]
    integration_freq = config_data["lu_bw_prepreocesor"]["integration_freq"]
    
    
def _main():
    print("Loading LU-BW sensorts data")
    config_data = ut.get_config_data(sys.argv[1]) 
    print("Configuration file loaded")
    ut.sanity_cahecks(config_data)
    print("Sanity checks performed. Everything is ok.")
    _read_config(config_data)
    print("Relavent Configuration data has been loaded")    
        
    file_name = sys.argv[2]
    print("Loading from " + file_name)
    frames = pd.ExcelFile(file_name, index_col="timestamp",parse_dates=True)
    print(frames.sheet_names)
    for name in frames.sheet_names:
        df = frames.parse(name,index_col="timestamp",parse_dates=True)


        df["P1"].replace(to_replace = -999.0, value = np.NaN, inplace=True)
        df["P2"].replace(to_replace = -999.0, value = np.NaN, inplace=True)

        
        df = df.sort_index()
        day = df.groupby(pd.Grouper(freq=integration_freq))
        if integration_type == "MEAN":
            df = day.mean()
        elif integration_type == "MEADIAN":             
            df = day.median()
        elif integration_type == "MIN":                 
            df = day.min()
        elif integration_type == "MAX":                 
            df = day.max()



        date_range = pd.date_range(start=reindex_period[0], end=reindex_period[1], freq=integration_freq)
        date_range = date_range[0:date_range.shape[0] - 1]
        df = df.reindex(date_range)
        
        df["P1"] =  pd.to_numeric(df["P1"])
        df["P2"] =  pd.to_numeric(df["P2"])
        id = name.split(" ")[0]
        rename_dict = {}
        rename_dict["P1"] = "P1_" + str(id)
        rename_dict["P2"] = "P2_" + str(id)
        df.rename(index=str, columns=rename_dict, inplace=True)
        

        desc_file = open(description_files_dir+"/"+id+".txt","w")
        desc_file.write("Description before missing data correction\n-----------------------\n")
        desc_file.write(str(df.describe())+"\n\n")
        desc_file.write(str(df.corr())+"\n\n")


        df.interpolate(method="linear",inplace=True)
        df.fillna(df.mean(), inplace=True)

        plt.cla()
        plt.close()
        plt.clf()
        df.plot(linewidth=1.0, style = ['r-', 'b--'], grid = True,figsize=(13, 11), title="P1 and P2 values of " + id)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.savefig(os.path.join(description_files_dir, "plots/" + id + "_plot_P1_P2.png"), bbox_inches='tight')

        plt.cla()
        plt.close()
        plt.clf()
        df.rolling(100).mean().plot(linewidth=1.0, style = ['r-', 'b--'], grid = True,figsize=(13, 11), title="Rolling averages over P1 and P2 of " + id)
        plt.savefig(os.path.join(description_files_dir, "plots/" + id + "_rolling_avg.png"), bbox_inches='tight')
                
        desc_file.write("Description after missing data correction\n-----------------------\n")
        desc_file.write(str(df.describe())+"\n\n")
        desc_file.write(str(df.corr())+"\n\n")


        

        df.to_csv(
            os.path.join(lu_bw_data_files_dir, name.split(" ")[0]+".csv"),
            sep=";",
            index_label="timestamp")

        
    
        


def execute(config_data):
    _read_config(config_data)





    
if __name__ == '__main__':
    _main()
