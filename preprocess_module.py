#!/home/arnaud/anaconda3/bin/python3.6



import sys, os
import utils as ut
import pandas as pd


files_dir = None
raw_files_dir = None 
keep_columns = None
time_column = None

def _read_config(config_data):
    global files_dir, raw_files_dir, keep_columns, time_column
    print("Reading config data")

    files_dir = os.path.expanduser(config_data["data_files_dir"])
    raw_files_dir = os.path.expanduser(config_data["raw_down_dir"])
    keep_columns = config_data["preprocess_module"]["keep_columns"]
    time_column = config_data["preprocess_module"]["time_column"]


def _main():
    print("Starting the proprocess module from the command line")
    config_data = ut.get_config_data(sys.argv[1]) 
    print("Configuration file loaded")
    ut.sanity_cahecks(config_data)
    _read_config(config_data)


    raw_files = [os.path.join(raw_files_dir, f) for f in os.listdir(raw_files_dir)
                 if os.path.isfile(os.path.join(raw_files_dir, f))]


    for f in raw_files:
        _process_file(f)



def _process_file(f):
    print(f)

    df = pd.read_csv(f,sep=';', parse_dates=[time_column])


    
    for key in df.keys():
        if not key in keep_columns:
            print("Deleting column: " + str(key) )
            del(df[key])


    df[time_column] = df[time_column].dt.strftime('%Y-%m-%d-%H-%M')
    df[time_column] = pd.to_datetime(df[time_column], format='%Y-%m-%d-%H-%M')

    #Check of there are consecetive measurements in a single minute


    print(str(df[time_column].unique().size))
    print(str(df[time_column].size))
    if df[time_column].unique().size != df[time_column].size:
        print("Bad")




    


def execute(config_data):
    print("helo")
    _read_config(config_data)






    
if __name__ == '__main__':
    _main()
