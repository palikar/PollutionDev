#!/usr/bin/python

import json
import os
import sys
import numpy as np
import subprocess
import pandas as pd
from math import cos, sin, atan2, sqrt



def get_config_data(config_file_name):
    """Reads the configuration file and returns dictionary with its contents
    """
    with open(config_file_name, "r") as config_file:
        config_data = json.load(config_file)
        return config_data


def sanity_cahecks(config):
    """The function performs all necessary checks on the env direcotry and
    creates the folders that are not there. Checks on the
    configuration are also made. If certain direcoty needed for the
    system is in not metioned in the configuration, it will be cought
    here.
    """
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


def distanceInKmBetweenEarthCoordinates(lat1, lon1, lat2, lon2):
    """Calculates the distance in KM between two sets of coordinates.
    """
    def degreesToRadians(degrees):
        return degrees * 3.14159265359 / 180;
    earthRadiusKm = 6371

    dLat = degreesToRadians(lat2-lat1)
    dLon = degreesToRadians(lon2-lon1)
    
    lat1 = degreesToRadians(lat1)
    lat2 = degreesToRadians(lat2)

    a = sin(dLat/2) * sin(dLat/2) + sin(dLon/2) * sin(dLon/2) * cos(lat1) * cos(lat2); 
    c = 2 * atan2(sqrt(a), sqrt(1-a)); 
    return earthRadiusKm * c


def sensor_coord(s_id):
    """Finds the coordinates of a sensor with given id. The function
    searches through the direcotry with raw files and inspects the
    CSV-files of the sensor.
    """
    findCMD = 'find ./env/raw_files/*_'+ str(s_id)+".csv"
    out = subprocess.Popen(findCMD,shell=True,stdin=subprocess.PIPE, 
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (stdout, stderr) = out.communicate()
    filelist = stdout.decode().split()
    
    df = pd.read_csv(filelist[0],sep=';')
    lat = float(df["lat"][0])
    lon = float(df["lon"][0])
    return (lat, lon)
    

def test_train_split(X, y, train_size=0.75, random=False):
    """Splits tha data into train and test sets. This can be done either
    randomly or not. The latter case preserves the time dependence of
    the data and will generated train split form "the past" and test
    split with "feature" values.
    train_size: the ration of size of the train set to the whole data set.
    """
    if random:
        return train_test_split(X, y, train_size=train_size)
    else:
        train_cnt = int(round(X.shape[0]*0.75, 0))
        return X[0:train_cnt], X[train_cnt:], y[0:train_cnt], y[train_cnt:]
    

def select_data(station, value, period, include_lu_bw=False, output_value=None, base_dir=None):
    """Given base directory with the final dataframes, this function can
    select the apropriate data from them acoriding to the given
    parameters. The function fully abstract away the selection of the
    data used to training the models. It also return the feature name
    of each column as well as the name of the target value that is to
    be predicted .
    """
    if output_value is None:
        output_value = "P1"

    df = None
    if period == "1D":
        df = pd.read_csv(base_dir + "/final_data_frame_1D.csv", sep=";", index_col="timestamp", parse_dates=True)
    elif period == "12H":
        df = pd.read_csv(base_dir + "/final_data_frame_12H.csv", sep=";", index_col="timestamp", parse_dates=True)
    elif period == "1H":
        df = pd.read_csv(base_dir + "/final_data_frame_1H.csv", sep=";", index_col="timestamp", parse_dates=True)

    X, y = None, None
    both_vals = False
    if value == "P1":
        columns = list(filter(lambda col: "P1" in str(col),list(df.columns.values)))
    elif value == "P2":
        columns = list(filter(lambda col: "P2" in str(col),list(df.columns.values)))
    else:
        columns = list(filter(lambda col: "P2" in str(col) or "P1" in str(col),list(df.columns.values)))
        both_vals=True
        
        
    
    out_col = columns.index(output_value+"_"+station)
    out_name = columns[out_col]
    y = df[columns[out_col]].values
    names=None
    x=None
    
    if "P1_"+station in columns : columns.remove("P1_"+station)
    if "P2_"+station in columns : columns.remove("P2_"+station)
    
    if not include_lu_bw:
        if both_vals:
            names = columns[0:-4]
        else:
            names = columns[0:-2]
        X = df[names].values            
    else:
        names = columns    
        X = df[columns].values
        
    
    return X, y, names, out_name


