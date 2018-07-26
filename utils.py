#!/usr/bin/python

import json
import os
import sys
import numpy as np
import subprocess
import pandas as pd
from math import cos, sin, atan2, sqrt


if __name__ == '__main__':
    unittest.main()


def get_config_data(config_file_name):
    
    with open(config_file_name, "r") as config_file:
        config_data = json.load(config_file)
        return config_data



def sanity_cahecks(config):
    # directory for the data of all scripts
    if not "env_dir" in config:
        print("Envinroment directory not set in the config file (env_dir)")
        sys.exit("Exiting with error")

        if not os.path.isdir(os.path.expanduser(config["env_dir"])):
            os.makedirs(os.path.expanduser(config["env_dir"]))

    if not "env_dir" in config:
        print("Download directory not set in the config file (raw_down_dir)")
        sys.exit("Exiting with error")
    if not os.path.isdir(os.path.expanduser(config["raw_down_dir"])):
        os.makedirs(os.path.expanduser(config["raw_down_dir"]))


    if not "data_files_dir" in config:
        print("Final data files directory not set in the config file (data_files_dir)")
        sys.exit("Exiting with error")
    if not os.path.isdir(os.path.expanduser(config["data_files_dir"])):
        os.makedirs(os.path.expanduser(config["data_files_dir"]))


    if not "description_files_dir" in config:
        print("Final description files directory not set in the config file (description_files_dir)")
        sys.exit("Exiting with error")
    if not os.path.isdir(os.path.expanduser(config["description_files_dir"])):
        os.makedirs(os.path.expanduser(config["description_files_dir"]))
    if not os.path.isdir(os.path.join(os.path.expanduser(config["description_files_dir"]),"plots")):
        os.makedirs(os.path.join(os.path.expanduser(config["description_files_dir"]),"plots"))

    if not "lu_bw_data_files_dir" in config:
        print("LU BW  files directory not set in the config file (lu_bw_data_files_dir)")
        sys.exit("Exiting with error")
    if not os.path.isdir(os.path.expanduser(config["lu_bw_data_files_dir"])):
        os.makedirs(os.path.expanduser(config["lu_bw_data_files_dir"]))


def generator(arrays, batch_size):
    """Generate batches, one with respect to each array's first axis."""
    starts = [0] * len(arrays)  # pointers to where we are in iteration
    while True:
        batches = []
        for i, array in enumerate(arrays):
            start = starts[i]
            stop = start + batch_size
            diff = stop - array.shape[0]
            if diff <= 0:
                batch = array[start:stop]
                starts[i] += batch_size
            else:
                batch = np.concatenate((array[start:], array[:diff]))
                starts[i] = diff
            batches.append(batch)
        yield batches



def degreesToRadians(degrees):
  return degrees * 3.14159265359 / 180;


def distanceInKmBetweenEarthCoordinates(lat1, lon1, lat2, lon2):
    earthRadiusKm = 6371

    dLat = degreesToRadians(lat2-lat1)
    dLon = degreesToRadians(lon2-lon1)
    
    lat1 = degreesToRadians(lat1)
    lat2 = degreesToRadians(lat2)

    a = sin(dLat/2) * sin(dLat/2) + sin(dLon/2) * sin(dLon/2) * cos(lat1) * cos(lat2); 
    c = 2 * atan2(sqrt(a), sqrt(1-a)); 
    return earthRadiusKm * c



def sensor_coord(s_id):
    findCMD = 'find ./env/raw_files/*_'+ str(s_id)+".csv"
    out = subprocess.Popen(findCMD,shell=True,stdin=subprocess.PIPE, 
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (stdout, stderr) = out.communicate()
    filelist = stdout.decode().split()
    
    df = pd.read_csv(filelist[0],sep=';')
    lat = float(df["lat"][0])
    lon = float(df["lon"][0])
    return (lat, lon)
    


def list_coordinates(sensors):

    center_lat = 48.781342
    center_lon = 9.173868
    
    for sen in sensors:
        s_id = sen.split("_")[1]
        findCMD = 'find ./env/raw_files/*_' + "" + str(s_id) + ".csv"
        out = subprocess.Popen(findCMD,shell=True,stdin=subprocess.PIPE, 
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        (stdout, stderr) = out.communicate()
        filelist = stdout.decode().split()

        df = pd.read_csv(filelist[0],sep=';')
        lat = float(df["lat"][0])
        lon = float(df["lon"][0])
        print(s_id + ";" + str(lat) + ";" + str(lon))




def test_train_split(X, y, train_size=0.75, random=False):
    if random:
        return train_test_split(X, y, train_size=train_size, random_state=42)
    else:
        train_cnt = int(round(X.shape[0]*0.75, 0))
        return X[0:train_cnt], X[train_cnt:], y[0:train_cnt], y[train_cnt:]
    

def select_data(station, value, period):
    df = None
    if period == "1D":
        df = pd.read_csv("./env/data_frames/final_data_frame_1D.csv", sep=";", index_col="timestamp", parse_dates=True)
    elif period == "12H":
        df = pd.read_csv("./env/data_frames/final_data_frame_12H.csv", sep=";", index_col="timestamp", parse_dates=True)
    elif period == "1H":
        df = pd.read_csv("./env/data_frames/final_data_frame_1H.csv", sep=";", index_col="timestamp", parse_dates=True)

    X, y = None, None
    if value == "P1":
        columns = list(filter(lambda col: "P1" in str(col),list(df.columns.values)))
    elif value == "P2":
        columns = list(filter(lambda col: "P2" in str(col),list(df.columns.values)))
    else:
        columns = list(filter(lambda col: "P2" in str(col) or "P1" in str(col),list(df.columns.values)))
        
    out_col = None
    if station == "SBC":
        out_col = -1
    else:
        out_col = -2

    y = df[columns[out_col]].values
    X = df[columns[0:-3]].values
    return X, y

    



    








        

if __name__ == '__main__':
    print("This file is not to be executed from the command line")
            

    
