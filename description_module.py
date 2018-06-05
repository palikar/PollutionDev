#!/home/arnaud/anaconda3/bin/python3.6



import argparse
import sys, os
import utils as ut
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re


data_files_dir = None
bad_missing_data_sensors = None
description_files_dir = None
env_dir = None

def _read_config(config_data):
    global data_files_dir, bad_missing_data_sensors, description_files_dir,env_dir

    data_files_dir = os.path.expanduser(config_data["data_files_dir"])
    bad_missing_data_sensors = os.path.expanduser(config_data["preprocess_module"]["bad_missing_data_sensors"])
    description_files_dir = os.path.expanduser(config_data["description_files_dir"])
    env_dir = os.path.expanduser(config_data["env_dir"])


def generate_info_dict(id,col, df):
    return {
        "id": id,
        "min":df[col].min(),
        "max":df[col].max(),
        "mean":df[col].mean(),
        "var":df[col].var(),
        "std":df[col].std(),
        "skew":df[col].skew(),
        "kurt":df[col].kurt(),
        "count":df[col].count()   
    } 




    
def _main():
    print("Starting the proprocess module from the command line")
    config_data = ut.get_config_data(sys.argv[1]) 
    print("Configuration file loaded")
    ut.sanity_cahecks(config_data)
    print("Sanity checks performed. Everything is ok.")
    _read_config(config_data)
    print("Relavent Configuration data has been loaded")


    df_before_p1 = pd.DataFrame(columns = ["id", "min", "max", "mean","var", "std", "skew", "kurt", "count"])
    df_before_p2 = pd.DataFrame(columns = ["id", "min", "max", "mean","var", "std", "skew", "kurt", "count"])
    df_after_p1 = pd.DataFrame(columns = ["id", "min", "max", "mean","var", "std", "skew", "kurt", "count"])
    df_after_p2 = pd.DataFrame(columns = ["id", "min", "max", "mean","var", "std", "skew", "kurt", "count"])
    
    data_files = [os.path.join(data_files_dir, f) for f in os.listdir(data_files_dir)
                 if os.path.isfile(os.path.join(data_files_dir, f))]


    missig_data_sensors = np.loadtxt(bad_missing_data_sensors, dtype='str',ndmin=1)
    print("Sensors with missing data "+str(len(missig_data_sensors))+": " + str(missig_data_sensors))

    size = len(data_files)
    for indx,f in enumerate(data_files):
        
        id = str(re.search('end_data_frame_(\d+)\.csv', f, re.IGNORECASE).group(1))
        if id  in  missig_data_sensors:
            print("Skipping" + id + "; " + str(indx) + "/" + str(size))
            continue
        print("Processing " + id + "; " + str(indx) + "/" + str(size))
        df = pd.read_csv(f,sep=';',parse_dates=True, index_col="timestamp")


        desc_file = open(description_files_dir+"/"+id+".txt","w")
        # Save those to file
        desc_file.write("Description before missing data correction\n-----------------------\n")
        desc_file.write(str(df.describe())+"\n\n")
        desc_file.write(str(df.corr())+"\n\n")

        
        df_before_p1 = df_before_p1.append(generate_info_dict(id,"P1_"+id, df),ignore_index=True)
        df_before_p2 = df_before_p2.append(generate_info_dict(id,"P2_"+id, df),ignore_index=True)

        


        df = df[~df.index.duplicated()]
        date_range = pd.date_range(start='2017-1-1',end='2018-01-01' , freq='30T')
        date_range = date_range[0:date_range.shape[0] - 1]
        df = df.reindex(date_range)
        df.fillna(df.mean(), inplace=True)                
        
        desc_file.write("Description after missing data correction\n-----------------------\n")
        desc_file.write(str(df.describe())+"\n\n")
        desc_file.write(str(df.corr())+"\n\n")
        
        df_after_p1 = df_before_p1.append(generate_info_dict(id,"P1_"+id, df),ignore_index=True)
        df_after_p2 = df_before_p2.append(generate_info_dict(id,"P2_"+id, df),ignore_index=True)

        
        plt.cla()
        plt.close()
        plt.clf()

        df.iloc[:,1].plot(linewidth=2)
        df.iloc[:,1].rolling(100).mean().plot(linewidth=2.5)
        plt.savefig(os.path.join(description_files_dir, "plots/" + id + "_plot_P1.png"), bbox_inches='tight')

        plt.cla()
        plt.close()
        plt.clf()
        
        df.iloc[:,0].plot(linewidth=2)
        df.iloc[:,0].rolling(100).mean().plot(linewidth=2.5)        
        plt.savefig(os.path.join(description_files_dir, "plots/" + id + "_plot_P2.png"), bbox_inches='tight')
        df.to_csv(f,sep=";",index_label="timestamp")
        desc_file.close()

    df_before_p1.to_csv(env_dir+"description_frame_p1_before.csv", sep=";")
    df_before_p2.to_csv(env_dir+"description_frame_p2_before.csv", sep=";")

    df_after_p1.to_csv(env_dir+"description_frame_p2_after.csv", sep=";")
    df_after_p2.to_csv(env_dir+"description_frame_p2_after.csv", sep=";")

if __name__ == '__main__':
    _main()




        
