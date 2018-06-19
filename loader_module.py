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
    global data_files_dir, period, freq, final_df_name, folder


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

    data_files = [os.path.join(folder, f) for f in os.listdir(folder)
                  if os.path.isfile(os.path.join(folder, f))]
    for f in data_files:
        df = pd.read_csv(f,sep=';',parse_dates=True, index_col="timestamp")
        df_end = pd.concat([df_end , df], axis=1)




    lu_bw_folder = os.path.join(folder, "lu_bw")
    data_files = [os.path.join(data_files_dir, f) for f in os.listdir(data_files_dir)
                  if os.path.isfile(os.path.join(data_files_dir, f))]

    for f in data_files:
        df = pd.read_csv(f,sep=';',parse_dates=True, index_col="timestamp")
        df_end = pd.concat([df_end, df], axis=1)

    
    df_end.to_csv(folder + "/" + final_df_name, sep=";",index_label="timestamp")


def execute(config_data):
    _read_config(config_data)





    
if __name__ == '__main__':
    _main()
