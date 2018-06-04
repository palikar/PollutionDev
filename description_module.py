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

def _read_config(config_data):
    global data_files_dir, bad_missing_data_sensors

    data_files_dir = os.path.expanduser(config_data["data_files_dir"])
    bad_missing_data_sensors = os.path.expanduser(config_data["preprocess_module"]["bad_missing_data_sensors"])




    
def _main():
    print("Starting the proprocess module from the command line")
    config_data = ut.get_config_data(sys.argv[1]) 
    print("Configuration file loaded")
    ut.sanity_cahecks(config_data)
    print("Sanity checks performed. Everything is ok.")
    _read_config(config_data)
    print("Relavent Configuration data has been loaded")




    data_files = [os.path.join(data_files_dir, f) for f in os.listdir(data_files_dir)
                 if os.path.isfile(os.path.join(data_files_dir, f))]
    missig_data_sensors = np.loadtxt(bad_missing_data_sensors, dtype='str',ndmin=1)

    print("Sensors with missing data "+str(len(missig_data_sensors))+": " + str(missig_data_sensors))

    for f in data_files[0:1]:
        id = str(re.search('end_data_frame_(\d+)\.csv', f, re.IGNORECASE).group(0))
        if id  in  missig_data_sensors:
            continue
        print("Processing: " + id)
        df = pd.read_csv(f,sep=';',parse_dates=True, index_col="timestamp")
        date_range = pd.date_range(start='2017-1-1',end='2018-01-01' , freq='30T')
        date_range = date_range[0:date_range.shape[0]-1]
        df = df.reindex(date_range)
        df.fillna(df.mean(), inplace=True)
        
        
        # df = df.rolling(window=100).mean()
        
        print(df.describe())
        print(df.corr())
        df.iloc[:,1].plot(linewidth=2)
        df.iloc[:,1].rolling(100).mean().plot(linewidth=2.5)
        plt.show()
        

    



    
    

if __name__ == '__main__':
    _main()
