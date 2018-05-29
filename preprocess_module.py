#!/home/arnaud/anaconda3/bin/python3.6



import sys, os
import utils as ut
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re


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
sensors_on_date = {}

def _read_config(config_data):
    global files_dir, raw_files_dir, keep_columns, time_column, duplicates_resolution, min_sensor_cnt, files_list_file, all_sensors_list_file, good_sensors_list_file, good_sensors_data_files_list
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
    print( str(len(all_sensors)) + " sensors have data files for more than " + str(min_sensor_cnt) + " days. Those are \'good\'" )

    
    
    np.savetxt(str(good_sensors_list_file), good_sensors,fmt='%s')
    np.savetxt(str(all_sensors_list_file), all_sensors,fmt='%s')




    
def _main():

    
    print("Starting the proprocess module from the command line")
    config_data = ut.get_config_data(sys.argv[1]) 
    print("Configuration file loaded")
    ut.sanity_cahecks(config_data)
    print("Sanity checks performed. Everything is ok.")
    _read_config(config_data)
    print("Relavent Configuration data has been loaded")

    




    good_sensors = open(good_sensors_list_file, "r").read().split('\n')




        
    if "read_good_files_from_list" in config_data["preprocess_module"] and config_data["preprocess_module"]["read_good_files_from_list"]:
        raw_files_np  = np.genfromtxt(good_sensors_data_files_list,dtype='str')
        raw_files = raw_files_np.tolist()
    else:
        def check(f):
            if str(re.search('sensor_(\d+)\.csv', f, re.IGNORECASE).group(0)) in good_sensors:
                return True
            else:
                return False
            
        raw_files = [os.path.join(raw_files_dir, f) for f in os.listdir(raw_files_dir)
                 if os.path.isfile(os.path.join(raw_files_dir, f))]

        print(str(len(raw_files)) + " raw files found in the raw_files directory.("+ raw_files_dir +")")

        if "check_day_for_sensors" in config_data["preprocess_module"] and config_data["preprocess_module"]["check_day_for_sensors"]:
            _check_days_for_sensors(raw_files)

        raw_files = list(filter(check, raw_files))
        raw_files_np = np.array(raw_files)
        np.savetxt(str(good_sensors_data_files_list), raw_files_np, fmt='%s')

    

    print(str(len(raw_files)) + " files form good sensors")
    
    for f in raw_files[0:1000]:
        _process_file(f)

    # print("done")


sensors = {}
    


def _process_file(f):
    global sensors
    
    df = pd.read_csv(f,sep=';', parse_dates=[time_column])
    
    # print("Processing file: " + str(f))
    for key in df.keys():
        if not key in keep_columns:
            del(df[key])



    df[time_column] = pd.to_datetime(
        df[time_column].dt.strftime('%Y-%m-%d-%H-%M'),
        format='%Y-%m-%d-%H-%M')
    

    # print("Resolving duplicates")
    #No duplicates
    if df[time_column].unique().size != df[time_column].count():
        # print("Duplicates found. Perfoerming resolution on the duplicates")
        if duplicates_resolution == "AVG":
            df = df.groupby(time_column, as_index=False,sort=True).mean().reset_index()
        elif duplicates_resolution == "MEADIAN":
            df = df.groupby(time_column, as_index=False,sort=True).median().reset_index()
        elif duplicates_resolution == "MIN":
            df = df.groupby(time_column, as_index=False,sort=True).min().reset_index()
        elif duplicates_resolution == "MAX":
            df = df.groupby(time_column, as_index=False,sort=True).max().reset_index()

    # size,fields_cnt = df.shape
    # print("Size after duplicates resolution: " + str(size))


    df = df.groupby(pd.Grouper(key="timestamp", freq="30T")).mean()

    
    # size,fields_cnt = df.shape
    # print("Size after date grouping: " + str(size))

    # df[["P1", "P2"]].plot()
    # plt.show()


def execute(config_data):
    _read_config(config_data)






    
if __name__ == '__main__':
    _main()
