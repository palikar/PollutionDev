#!/home/arnaud/anaconda3/bin/python3.6



import argparse
import sys, os
import utils as ut
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re


args = None

files_dir = None
raw_files_dir = None 
keep_columns = None
time_column = None
duplicates_resolution = None
min_sensor_cnt = None
files_list_file = None
good_sensors_list_file = None
all_sensors_list_file = None
good_sensors_data_files_list = None
day_integration_type = None
day_integration_period = None
missing_data_resolution = None
missing_data_cnt_threshold = None
bad_missing_data_sensors = None
values_columns = None
id_column = None
id_columnreindex_period = None


sensors_on_date = {}

def _read_config(config_data):
    global files_dir, raw_files_dir, keep_columns, time_column, duplicates_resolution, min_sensor_cnt, files_list_file, all_sensors_list_file, good_sensors_list_file, good_sensors_data_files_list, day_integration_period, day_integration_type, missing_data_cnt_threshold, missing_data_resolution, bad_missing_data_sensors, values_columns, id_column,id_columnreindex_period
    print("Reading config data")

    files_dir = os.path.expanduser(config_data["data_files_dir"])
    raw_files_dir = os.path.expanduser(config_data["raw_down_dir"])
    keep_columns = config_data["preprocess_module"]["keep_columns"]
    time_column = config_data["preprocess_module"]["time_column"]
    duplicates_resolution = config_data["preprocess_module"]["duplicates_resolution"]
    min_sensor_cnt = config_data["preprocess_module"]["min_sensor_cnt"]
    files_list_file = os.path.expanduser(config_data["preprocess_module"]["files_list_file"])
    good_sensors_list_file = os.path.expanduser(config_data["preprocess_module"]["good_sensors_list_file"])
    all_sensors_list_file = os.path.expanduser(config_data["preprocess_module"]["all_sensors_list_file"])
    good_sensors_data_files_list = os.path.expanduser(config_data["preprocess_module"]["good_sensors_data_files_list"])
    day_integration_type = config_data["preprocess_module"]["day_integration_type"]
    day_integration_period = config_data["preprocess_module"]["day_integration_period"]
    missing_data_cnt_threshold = config_data["preprocess_module"]["missing_data_cnt_threshold"]
    missing_data_resolution = config_data["preprocess_module"]["missing_data_resolution"]
    bad_missing_data_sensors = config_data["preprocess_module"]["bad_missing_data_sensors"]

    values_columns = config_data["preprocess_module"]["values_columns"]
    id_column = config_data["preprocess_module"]["id_column"]
    reindex_period = config_data["preprocess_module"]["reindex_period"]
    
    

sensors = {}
def _check_days_for_sensors (raw_files):
    global sensors
    print("Reading days and sensors")
    for f in raw_files:
        sensor_id = re.search('sensor_(\d+)\.csv', f, re.IGNORECASE)
        if not sensor_id:
            continue
        sensor_id = str(sensor_id.group(0))
        date = re.search('(\d\d\d\d)-(\d\d)-(\d\d)', f, re.IGNORECASE)
        date = str(date.group(0))
        if not sensor_id in sensors:
            sensors[sensor_id] = list()
        sensors[sensor_id].append(date)    
    print("Done reading days and sensors")
    
    good_sensors = np.array([])
    all_sensors = np.array([])
    for sensor, dates in sensors.items():
        all_sensors = np.append(all_sensors, sensor)
        if len(dates) >= min_sensor_cnt:        
            good_sensors = np.append(good_sensors, str(sensor))


    print("There are raw files for overall " + str(len(all_sensors)) + " sensors" )
    print(str(len(good_sensors)) + " sensors have data files for more than " + str(min_sensor_cnt) + " days. Those are \'good\'" )

    
    
    np.savetxt(str(good_sensors_list_file), good_sensors,fmt='%s')
    np.savetxt(str(all_sensors_list_file), all_sensors,fmt='%s')
    return good_sensors
    


bad_data_sensors = np.array([])
def _main():
    
    print("Starting the proprocess module from the command line")
    config_data = ut.get_config_data(sys.argv[1]) 
    print("Configuration file loaded")
    ut.sanity_cahecks(config_data)
    print("Sanity checks performed. Everything is ok.")
    _read_config(config_data)
    print("Relavent Configuration data has been loaded")

    raw_files = None
    if "--read-good-files-from-list" in sys.argv:
        print("Files to be processed are read from: " + str(good_sensors_data_files_list))
        raw_files_np  = np.loadtxt(good_sensors_data_files_list, dtype='str',ndmin=1)
        raw_files = raw_files_np.tolist()
    else:
        print("Reading all raw files.")
        raw_files = [os.path.join(raw_files_dir, f) for f in os.listdir(raw_files_dir)
                 if os.path.isfile(os.path.join(raw_files_dir, f))]
        print(str(len(raw_files)) + " raw files found in the raw_files directory.("+ raw_files_dir +")")        

        if "--filter-raw-files" in sys.argv:
            good_sensors = None
            if "--check-day-for-sensors" in sys.argv:
                print("Processing the files and finding the good sensors")
                good_sensors = _check_days_for_sensors(raw_files)
            else:
                print("Reading good sensors from: " + str(good_sensors_list_file))
                good_sensors = open(good_sensors_list_file, "r").read().split('\n')
            raw_files = list(filter(lambda f: str(re.search('sensor_(\d+)\.csv', f, re.IGNORECASE).group(0)) in good_sensors, raw_files))
        if "--save-raw-files-list" in sys.argv:
            raw_files_np = np.array(raw_files)
            np.savetxt(str(good_sensors_data_files_list), raw_files_np, fmt='%s')

    if "--preprocess-files" in sys.argv:
        size = len(raw_files)
        print("Processing " + str(size) + " files")
        size = len(raw_files)
        for indx, f in enumerate(raw_files):
            _process_file(f)
            if indx % 25 == 0:
                print(str(indx) + "/" + str(size))
        print("Done processing files")
        print("Sensors with missing data above the threshold: " + str(bad_data_sensors))
        np.savetxt(bad_missing_data_sensors, bad_data_sensors, fmt='%s')

    

    
    if "--sort-end-frames" in sys.argv:
        data_files = [os.path.join(files_dir, f) for f in os.listdir(files_dir)
                      if os.path.isfile(os.path.join(files_dir, f))]
        size = len(data_files)
        for ind, f in enumerate(data_files):
            df = pd.read_csv(f,sep=';', parse_dates=True, index_col="timestamp")
            df.sort_index(ascending=True, inplace=True)
            df.to_csv(f, sep=";", )
        del(data_files)
        
        
    
    
def _process_file(f):
    global bad_data_sensors
    
    # print("Processing file: " + str(f))
    df = pd.read_csv(f,sep=';', parse_dates=[time_column])
    
    for key in df.keys():
        if not key in keep_columns:
            del(df[key])



    df[time_column] = pd.to_datetime(
        df[time_column].dt.strftime('%Y-%m-%d-%H-%M'),
        format='%Y-%m-%d-%H-%M')


    if df[time_column].unique().size != df[time_column].count():
        # print("Resolving duplicates")
        # print("Duplicates found. Perfoerming resolution on the duplicates")
        if duplicates_resolution == "MEAN":
            df = df.groupby(time_column, as_index=False,sort=True).mean().reset_index()
        elif duplicates_resolution == "MEADIAN":
            df = df.groupby(time_column, as_index=False,sort=True).median().reset_index()
        elif duplicates_resolution == "MIN":
            df = df.groupby(time_column, as_index=False,sort=True).min().reset_index()
        elif duplicates_resolution == "MAX":
            df = df.groupby(time_column, as_index=False,sort=True).max().reset_index()
 


    if day_integration_type == "MEAN":
        df = df.groupby(pd.Grouper(key=time_column, freq=day_integration_period)).mean()
    elif day_integration_type == "MEADIAN":             
        df = df.groupby(pd.Grouper(key=time_column, freq=day_integration_period)).median()
    elif day_integration_type == "MIN":                 
        df = df.groupby(pd.Grouper(key=time_column, freq=day_integration_period)).min()
    elif day_integration_type == "MAX":                 
        df = df.groupby(pd.Grouper(key=time_column, freq=day_integration_period)).max()
    

    

    

    #renaming for the final dataframe
    id = str(int(df[id_column].iloc[0]))
    rename_dict = {}
    for value_name in values_columns:
        rename_dict[value_name] = value_name + "_" + id
    df.rename(index=str, columns=rename_dict, inplace=True)

    
    #Information about missing data
    missing_count = df[list(rename_dict.values())[0]].isnull().sum()
    # print("Missing values: " + str(missing_count))
    if missing_count > 0 and missing_count <= missing_data_cnt_threshold: 
        if missing_data_resolution == "MEAN":
            for key,val in rename_dict.items():
                df[val].fillna(df[val].mean(), inplace=True)
        else:
            for key,val in rename_dict.items():
                df[val].interpolate(method="linear", inplace=True)
    elif missing_count > missing_data_cnt_threshold:
        bad_data_sensors = np.append(bad_data_sensors, str(id))
                
    
    # if the file for this sensor does not exist, create it
    # otherwise we just append the information at the end of the existing file
    if not os.path.isfile(files_dir + "/end_data_frame_"+id+".csv"):
        df.to_csv(
            os.path.join(files_dir + "end_data_frame_"+id+".csv"),
            sep=";",
            columns=list(rename_dict.values())
        )
    else:
        df.to_csv(
            os.path.join(files_dir + "end_data_frame_"+id+".cvs"),
            sep=";",
            columns=list(rename_dict.values()),
            mode="a",
            header=False
        )
        


def execute(config_data):
    _read_config(config_data)


if __name__ == '__main__':
    _main()
    
