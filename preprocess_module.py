#!/home/arnaud/anaconda3/bin/python3.6



import sys, os
import utils as ut
import pandas as pd
from matplotlib import pyplot as plt
import re


files_dir = None
raw_files_dir = None 
keep_columns = None
time_column = None
duplicates_resolution = None
min_sensor_cnt = None


sensors_on_date = {}

def _read_config(config_data):
    global files_dir, raw_files_dir, keep_columns, time_column, duplicates_resolution, min_sensor_cnt
    print("Reading config data")

    files_dir = os.path.expanduser(config_data["data_files_dir"])
    raw_files_dir = os.path.expanduser(config_data["raw_down_dir"])
    keep_columns = config_data["preprocess_module"]["keep_columns"]
    time_column = config_data["preprocess_module"]["time_column"]
    duplicates_resolution = config_data["preprocess_module"]["duplicates_resolution"]
    min_sensor_cnt = config_data["prepocess_module"]["min_sensor_cnt"]

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
    for sensor, dates in sensors.items():
        print("Sensor: " + str(sensor) + " - " + str(len(dates)) + " dates")

    
    


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
