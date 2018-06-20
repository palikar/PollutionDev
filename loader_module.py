#!/home/arnaud/anaconda3/bin/python3.6



import argparse
import sys, os
import utils as ut
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re



data_files_dir = None
period = None
freq = None
ignored_sensors_files = None
final_df_name = None
def _read_config(config_data):
    "Loading config variables"
    global data_files_dir, period, freq, final_df_name, folder, ignored_sensors_files



    data_files_dir = os.path.expanduser(config_data["data_files_dir"])
    period = config_data["loader_module"]["period"]
    freq = config_data["loader_module"]["freq"]
    ignored_sensors_files = config_data["loader_module"]["ingored_sensors_files"]
    folder = os.path.expanduser(config_data["loader_module"]["folder"])
    final_df_name = config_data["loader_module"]["final_df_name"]
    

    
    
def _main():
    print("Loading LU-BW sensorts data")
    config_data = ut.get_config_data(sys.argv[1]) 
    print("Configuration file loaded")
    ut.sanity_cahecks(config_data)
    print("Sanity checks performed. Everything is ok.")
    _read_config(config_data)
    print("Relavent Configuration data has been loaded")

    global data_files_dir
    
    

    df_end = pd.DataFrame()
    date_range = pd.date_range(start=period[1], end=period[1] , freq=freq)
    date_range = date_range[0:date_range.shape[0] - 1]
    df_end = df_end.reindex(date_range)

    ignored_sens = np.array([])
    
    for f in ignored_sensors_files:
        with open(os.path.expanduser(f)) as sens_file:
            sens = sens_file.read().split('\n')
            ignored_sens = np.append(ignored_sens,sens)        

    data_files = [os.path.join(folder, f) for f in os.listdir(folder)
                  if os.path.isfile(os.path.join(folder, f))]




    count = 0
    for f in data_files:
        res = re.search('end_data_frame_(\S+)\.csv', f, re.IGNORECASE)
        id = res.group(1)
        if str(id) not in ignored_sens:
            df = pd.read_csv(f,sep=';',parse_dates=True, index_col="timestamp")
            df_end = pd.concat([df_end , df], axis=1)
        

        
                



    lu_bw_folder = os.path.join(folder, "lu_bw")
    print(lu_bw_folder)
    data_files = [os.path.join(lu_bw_folder, f) for f in os.listdir(lu_bw_folder)
                  if os.path.isfile(os.path.join(lu_bw_folder, f))]

    for f in data_files:
        print(f)
        res = re.search('(\S+)\.csv', f, re.IGNORECASE)
        id = res.group(1)
        print("Loading " + id + " from lu bw")
        df = pd.read_csv(f,sep=';',parse_dates=True, index_col="timestamp")
        df_end = pd.concat([df_end, df], axis=1)


    print("Final shape of the DF: " + str(df_end.shape))
    df_end.to_csv(folder + "/" + final_df_name, sep=";",index_label="timestamp")


def execute(config_data):
    _read_config(config_data)





    
if __name__ == '__main__':
    _main()
