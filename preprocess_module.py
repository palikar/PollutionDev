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

sensors_on_date = {}

def _read_config(config_data):
    global files_dir, raw_files_dir, keep_columns, time_column, duplicates_resolution, min_sensor_cnt, files_list_file, all_sensors_list_file, good_sensors_list_file
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
    
def _main():
    print("Starting the proprocess module from the command line")
    config_data = ut.get_config_data(sys.argv[1]) 
    print("Configuration file loaded")
    ut.sanity_cahecks(config_data)
    _read_config(config_data)


    raw_files = [os.path.join(raw_files_dir, f) for f in os.listdir(raw_files_dir)
                 if os.path.isfile(os.path.join(raw_files_dir, f))]


    
    _check_days_for_sensors(raw_files)
    # for f in raw_files:
    #     _process_file(f)




sensors = {}


def _check_days_for_sensors (raw_files):
    global sensors
    print("Reading days and sensors")
    file_list = open(files_list_file,"r")
    for f in file_list.readlines():
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
            print(str(sensor) + " - " + str(len(dates)))
            good_sensors = np.append(good_sensors, str(sensor))

    np.savetxt(str(good_sensors_list_file), good_sensors,fmt='%s')
    np.savetxt(str(all_sensors_list_file), all_sensors,fmt='%s')
    print("Count: " + str(len(good_sensors)))
    file_list.close()
    


def _process_file(f):
    print(f)

    global sensors
    
    df = pd.read_csv(f,sep=';', parse_dates=[time_column])


    
    for key in df.keys():
        if not key in keep_columns:
            print("Deleting column: " + str(key) )
            del(df[key])


    #Pretty date
    
    df[time_column] = pd.to_datetime(
        df[time_column].dt.strftime('%Y-%m-%d-%H-%M'),
        format='%Y-%m-%d-%H-%M')
    

    #No duplicates
    if df[time_column].unique().size != df[time_column].count():
        print("Duplicates found. Perfoerming resolution on the duplicates")
        if duplicates_resolution == "AVG":
            df = df.groupby(time_column, as_index=False,sort=True).mean().reset_index()
        elif duplicates_resolution == "MEADIAN":
            df = df.groupby(time_column, as_index=False,sort=True).median().reset_index()
        elif duplicates_resolution == "MIN":
            df = df.groupby(time_column, as_index=False,sort=True).min().reset_index()
        elif duplicates_resolution == "MAX":
            df = df.groupby(time_column, as_index=False,sort=True).max().reset_index()

    # fig, ax = plt.subplots()
    # df["P2"].plot(ax=ax)
    # plt.show() # plt in place of ax

    

            

def execute(config_data):
    print("helo")
    _read_config(config_data)






    
if __name__ == '__main__':
    _main()
